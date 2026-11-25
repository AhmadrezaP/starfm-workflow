"""Build a conservative cloud/quality mask for Landsat--VIIRS registration.

The output is a byte GeoTIFF on the exact Landsat grid: 1 means that both
inputs are suitable for estimating a geometric shift; 0 means exclude it.
It is deliberately *not* used to change a VIIRS NBAR image.  The mask is only
passed to the paper-style moving-window search.

For the VIIRS VNP09GA first observation layer, the mask excludes pixels with
probably/confidently cloudy ``SurfReflect_QF1_1`` states, the M5 bad-quality
flag in ``SurfReflect_QF5_1``, and the adjacent-cloud flag in
``SurfReflect_QF7_1``.  For Landsat Collection 2 QA_PIXEL, it excludes fill,
dilated cloud, cirrus, cloud, cloud shadow, and snow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from osgeo import gdal

from run_viirs_nbar_hdf import VNP09_GRID, _hdf5_dataset, _open_dataset


def viirs_m5_clear_mask(qf1: np.ndarray, qf5: np.ndarray, qf7: np.ndarray) -> np.ndarray:
    """Return usable VNP09GA M5 pixels from the documented bit fields."""
    cloud_state = (np.asarray(qf1, dtype=np.uint8) >> 2) & 0b11
    m5_bad_quality = (np.asarray(qf5, dtype=np.uint8) >> 6) & 0b1
    adjacent_to_cloud = (np.asarray(qf7, dtype=np.uint8) >> 1) & 0b1
    # QF1 cloud states 10 and 11 are probably and confidently cloudy.
    return (cloud_state < 0b10) & (m5_bad_quality == 0) & (adjacent_to_cloud == 0)


def landsat_clear_mask(qa_pixel: np.ndarray) -> np.ndarray:
    """Return Collection 2 pixels free from fill, cloud, shadow, and snow."""
    excluded_bits = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
    return (np.asarray(qa_pixel, dtype=np.uint16) & excluded_bits) == 0


def _open_single_band(path: Path) -> gdal.Dataset:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise ValueError(f"GDAL could not open: {path}")
    if dataset.RasterCount != 1:
        raise ValueError(f"Expected a single-band raster: {path}")
    return dataset


def _bounds(dataset: gdal.Dataset) -> tuple[float, float, float, float]:
    transform = dataset.GetGeoTransform()
    return (
        transform[0],
        transform[3] + transform[5] * dataset.RasterYSize,
        transform[0] + transform[1] * dataset.RasterXSize,
        transform[3],
    )


def _warp_mask_to_landsat(mask: np.ndarray, viirs_reference: gdal.Dataset, landsat: gdal.Dataset) -> np.ndarray:
    memory = gdal.GetDriverByName("MEM").Create(
        "", viirs_reference.RasterXSize, viirs_reference.RasterYSize, 1, gdal.GDT_Byte
    )
    if memory is None:
        raise RuntimeError("GDAL could not create an in-memory VIIRS quality mask.")
    memory.SetGeoTransform(viirs_reference.GetGeoTransform())
    memory.SetProjection(viirs_reference.GetProjection())
    band = memory.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(mask.astype(np.uint8))
    warped = gdal.Warp(
        "",
        memory,
        format="MEM",
        dstSRS=landsat.GetProjection(),
        outputBounds=_bounds(landsat),
        width=landsat.RasterXSize,
        height=landsat.RasterYSize,
        resampleAlg="near",
        srcNodata=0,
        dstNodata=0,
    )
    if warped is None:
        raise RuntimeError("GDAL could not warp the VIIRS mask onto the Landsat grid.")
    return warped.GetRasterBand(1).ReadAsArray().astype(bool)


def _write_mask(path: Path, values: np.ndarray, reference: gdal.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = gdal.GetDriverByName("GTiff").Create(
        str(path), reference.RasterXSize, reference.RasterYSize, 1, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if output is None:
        raise RuntimeError(f"GDAL could not create: {path}")
    output.SetGeoTransform(reference.GetGeoTransform())
    output.SetProjection(reference.GetProjection())
    band = output.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(values.astype(np.uint8))
    band.FlushCache()
    output.FlushCache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Landsat-grid, cloud/quality mask for VIIRS M5 coregistration."
    )
    parser.add_argument("--landsat-reference", type=Path, required=True, help="Paired-date Landsat B4 raster defining the output grid.")
    parser.add_argument("--landsat-qa", type=Path, required=True, help="Paired-date Landsat Collection 2 QA_PIXEL raster.")
    parser.add_argument(
        "--vnp09ga",
        type=Path,
        action="append",
        required=True,
        help="Paired-date VNP09GA HDF5 tile. Repeat for every tile intersecting the Landsat scene.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output 0/1 valid-pixel mask GeoTIFF.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    landsat = _open_single_band(args.landsat_reference)
    landsat_qa = _open_single_band(args.landsat_qa)
    if (landsat.RasterXSize, landsat.RasterYSize) != (landsat_qa.RasterXSize, landsat_qa.RasterYSize) or landsat.GetGeoTransform() != landsat_qa.GetGeoTransform():
        raise ValueError("--landsat-reference and --landsat-qa must have the same grid.")

    viirs_on_landsat_grid = np.zeros(
        (landsat.RasterYSize, landsat.RasterXSize), dtype=bool
    )
    for vnp09ga_path in args.vnp09ga:
        qf1 = _open_dataset(
            _hdf5_dataset(vnp09ga_path, VNP09_GRID, "SurfReflect_QF1_1")
        )
        qf5 = _open_dataset(
            _hdf5_dataset(vnp09ga_path, VNP09_GRID, "SurfReflect_QF5_1")
        )
        qf7 = _open_dataset(
            _hdf5_dataset(vnp09ga_path, VNP09_GRID, "SurfReflect_QF7_1")
        )
        viirs_clear = viirs_m5_clear_mask(
            qf1.GetRasterBand(1).ReadAsArray(),
            qf5.GetRasterBand(1).ReadAsArray(),
            qf7.GetRasterBand(1).ReadAsArray(),
        )
        viirs_on_landsat_grid |= _warp_mask_to_landsat(viirs_clear, qf1, landsat)
    combined = landsat_clear_mask(landsat_qa.GetRasterBand(1).ReadAsArray()) & viirs_on_landsat_grid
    _write_mask(args.output, combined, landsat)
    print(
        f"Wrote {args.output}: {np.count_nonzero(combined):,} valid of {combined.size:,} "
        f"pixels ({100 * np.mean(combined):.2f}%)."
    )


if __name__ == "__main__":
    main()
