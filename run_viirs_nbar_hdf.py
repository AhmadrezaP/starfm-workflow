"""Run VIIRS directional-reflectance to NBAR normalization directly from HDF5.

This command is the file-preparation companion to :mod:`viirs_nbar`.  It reads
one VNP09GA M-band tile and its three same-day Collection 2 VNP43D BRDF
parameter files, warps the global BRDF parameters onto the VNP09GA grid, and
writes the corrected NBAR GeoTIFF.  It currently supports the VIIRS M bands
for which the Collection 2 VNP43D BRDF-parameter products are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from osgeo import gdal

from viirs_nbar import normalize_to_nbar, _write_raster


# Collection 2 VNP09GA files name the 1 km grid ``VIIRS_Grid_1km_2D``.
# Collection 1 files used ``VNP_Grid_1km_2D`` instead.
VNP09_GRID = "VIIRS_Grid_1km_2D"
VNP43D_GRID = "VIIRS_CMG_BRDF"
SUPPORTED_M_BANDS = {"M1", "M2", "M3", "M4", "M5", "M7", "M8", "M10", "M11"}
IGBP_CLASS_MIN = 0
IGBP_CLASS_MAX = 17
FILL_SOURCE_NODATA = 255


def _hdf5_dataset(file_path: Path, grid: str, field: str) -> str:
    # GDAL's HDF5 driver exposes the physical HDF-EOS ``Data Fields`` group
    # with an underscore in subdataset paths.
    return f'HDF5:"{file_path.absolute()}"://HDFEOS/GRIDS/{grid}/Data_Fields/{field}'


def _open_dataset(dataset_path: str) -> gdal.Dataset:
    dataset = gdal.Open(dataset_path)
    if dataset is None:
        raise ValueError(f"GDAL could not open HDF5 science dataset: {dataset_path}")
    return dataset


def _read_scaled(dataset: gdal.Dataset, scale: float, fill_value: float) -> np.ndarray:
    values = dataset.GetRasterBand(1).ReadAsArray().astype(np.float64) * scale
    values[np.isclose(values, fill_value * scale)] = np.nan
    return values


def _warp_to_reference(
    dataset: gdal.Dataset,
    reference: gdal.Dataset,
    resampling: str,
    source_nodata: float = 32767,
) -> gdal.Dataset:
    # VNP43D is a global 30-arc-second CMG product.  GDAL's HDF5 driver
    # currently opens its parameter fields with an incorrect geotransform
    # (their nominal pixel size divided by the raster dimensions).  Restore
    # the documented global WGS84 extent in a VRT before warping to VNP09GA.
    source = gdal.Translate(
        "",
        dataset,
        format="VRT",
        outputSRS="EPSG:4326",
        outputBounds=(-180.0, 90.0, 180.0, -90.0),
        width=dataset.RasterXSize,
        height=dataset.RasterYSize,
    )
    if source is None:
        raise RuntimeError("GDAL could not assign VNP43D's documented global grid.")
    transform = reference.GetGeoTransform()
    min_x = transform[0]
    max_y = transform[3]
    max_x = min_x + transform[1] * reference.RasterXSize
    min_y = max_y + transform[5] * reference.RasterYSize
    warped = gdal.Warp(
        "",
        source,
        format="MEM",
        dstSRS=reference.GetProjection(),
        outputBounds=(min_x, min_y, max_x, max_y),
        width=reference.RasterXSize,
        height=reference.RasterYSize,
        resampleAlg=resampling,
        srcNodata=source_nodata,
        dstNodata=source_nodata,
    )
    if warped is None:
        raise RuntimeError("GDAL could not warp the BRDF parameter to the VNP09GA grid.")
    return warped


def _reference_bounds(reference: gdal.Dataset) -> tuple[float, float, float, float]:
    """Return a north-up reference dataset's bounds in its native CRS."""
    transform = reference.GetGeoTransform()
    min_x = transform[0]
    max_y = transform[3]
    max_x = min_x + transform[1] * reference.RasterXSize
    min_y = max_y + transform[5] * reference.RasterYSize
    return min_x, min_y, max_x, max_y


def _warp_native_grid_to_reference(
    dataset: gdal.Dataset,
    reference: gdal.Dataset,
    source_nodata: float,
) -> gdal.Dataset:
    """Warp a correctly georeferenced native product grid to VNP09GA."""
    warped = gdal.Warp(
        "",
        dataset,
        format="MEM",
        dstSRS=reference.GetProjection(),
        outputBounds=_reference_bounds(reference),
        width=reference.RasterXSize,
        height=reference.RasterYSize,
        resampleAlg="near",
        srcNodata=source_nodata,
        dstNodata=source_nodata,
    )
    if warped is None:
        raise RuntimeError("GDAL could not warp the land-cover grid to VNP09GA.")
    return warped


def _open_mcd12q1_land_cover(file_path: Path) -> gdal.Dataset:
    """Open the IGBP ``LC_Type1`` layer from an annual MCD12Q1 HDF4 file."""
    container = gdal.Open(str(file_path))
    if container is not None:
        for dataset_path, description in container.GetSubDatasets():
            if dataset_path.endswith(":LC_Type1") or "Land_Cover_Type_1" in description:
                return _open_dataset(dataset_path)

    # This fallback is used by older GDAL builds that do not enumerate HDF4
    # EOS subdatasets on the container dataset.
    fallback = f'HDF4_EOS:EOS_GRID:"{file_path.absolute()}":MOD12Q1:LC_Type1'
    dataset = gdal.Open(fallback)
    if dataset is None:
        raise ValueError(
            "GDAL could not open LC_Type1 from the MCD12Q1 file. "
            "Install GDAL's HDF4 driver and verify the file is not corrupted."
        )
    return dataset


def fill_brdf_parameters_by_land_cover(
    f_iso: np.ndarray,
    f_vol: np.ndarray,
    f_geo: np.ndarray,
    quality: np.ndarray,
    land_cover: np.ndarray,
    trusted_quality_values: set[int],
    fill_quality_values: set[int],
    minimum_class_samples: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill unusable BRDF parameters from a local annual IGBP-class lookup table.

    The table is calculated from same-day, trusted VNP43D BRDF parameters.  A
    class median is used when at least ``minimum_class_samples`` trusted
    pixels exist; otherwise the same-day global trusted median is a documented
    fallback.  The returned source map is 0 for direct parameters, 1 for a
    class-LUT fill, 2 for the global fallback, and 255 for no usable BRDF.
    """
    arrays = [np.asarray(value) for value in (f_iso, f_vol, f_geo, quality, land_cover)]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("BRDF parameters, quality, and land cover must have the same shape.")

    iso, vol, geo = (np.asarray(value, dtype=np.float64).copy() for value in arrays[:3])
    quality_array = np.asarray(quality)
    land_cover_array = np.asarray(land_cover)
    parameter_valid = np.isfinite(iso) & np.isfinite(vol) & np.isfinite(geo)
    valid_land_cover = (
        np.isfinite(land_cover_array)
        & (land_cover_array >= IGBP_CLASS_MIN)
        & (land_cover_array <= IGBP_CLASS_MAX)
    )
    direct = np.isin(quality_array, list(trusted_quality_values))
    sample_mask = direct & parameter_valid & valid_land_cover
    source = np.full(iso.shape, FILL_SOURCE_NODATA, dtype=np.uint8)
    source[direct & parameter_valid] = 0

    if not np.any(sample_mask):
        raise ValueError("No valid trusted pixels are available to construct the BRDF gap-fill lookup table.")

    global_parameters = tuple(np.median(parameter[sample_mask]) for parameter in (iso, vol, geo))
    class_parameters: dict[int, tuple[float, float, float]] = {}
    for class_id in np.unique(land_cover_array[sample_mask]).astype(int):
        class_mask = sample_mask & (land_cover_array == class_id)
        if np.count_nonzero(class_mask) >= minimum_class_samples:
            class_parameters[class_id] = tuple(
                np.median(parameter[class_mask]) for parameter in (iso, vol, geo)
            )

    fillable = np.isin(quality_array, list(fill_quality_values)) & valid_land_cover
    for class_id in np.unique(land_cover_array[fillable]).astype(int):
        class_mask = fillable & (land_cover_array == class_id)
        parameters = class_parameters.get(class_id)
        if parameters is None:
            parameters = global_parameters
            source[class_mask] = 2
        else:
            source[class_mask] = 1
        iso[class_mask], vol[class_mask], geo[class_mask] = parameters

    usable = (source != FILL_SOURCE_NODATA) & np.isfinite(iso) & np.isfinite(vol) & np.isfinite(geo)
    return iso, vol, geo, usable, source


def _quality_values(values: str) -> set[int]:
    try:
        return {int(value.strip()) for value in values.split(",")}
    except ValueError as error:
        raise argparse.ArgumentTypeError("Quality values must be comma-separated integers, e.g. '0,1'.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate VIIRS M-band NBAR from VNP09GA and VNP43D HDF5 files."
    )
    parser.add_argument("--vnp09ga", type=Path, required=True, help="Daily VNP09GA h5 tile.")
    parser.add_argument("--fiso", type=Path, required=True, help="VNP43D parameter-1 h5 file.")
    parser.add_argument("--fvol", type=Path, required=True, help="VNP43D parameter-2 h5 file.")
    parser.add_argument("--fgeo", type=Path, required=True, help="VNP43D parameter-3 h5 file.")
    parser.add_argument("--quality", type=Path, help="Optional VNP43D40 BRDF-quality h5 file.")
    parser.add_argument(
        "--valid-quality-values",
        type=_quality_values,
        default={0},
        help="VNP43D quality classes to retain; default: 0 (best-quality full inversion).",
    )
    parser.add_argument(
        "--land-cover",
        type=Path,
        help=(
            "Annual MCD12Q1 HDF file. When supplied with --quality, use its IGBP LC_Type1 "
            "classes to gap-fill non-trusted VNP43D BRDF parameters."
        ),
    )
    parser.add_argument(
        "--gap-fill-quality-values",
        type=_quality_values,
        default={1, 2, 3},
        help="VNP43D quality classes eligible for land-cover BRDF gap filling; default: 1,2,3.",
    )
    parser.add_argument(
        "--brdf-source-output",
        type=Path,
        help="Optional diagnostic raster: 0 direct, 1 IGBP-class fill, 2 global fallback, 255 unavailable.",
    )
    parser.add_argument("--band", choices=sorted(SUPPORTED_M_BANDS), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output Float32 NBAR GeoTIFF.")
    parser.add_argument(
        "--brdf-resampling",
        choices=("near", "bilinear"),
        default="bilinear",
        help="Resampling for global VNP43D parameter rasters; default: bilinear.",
    )
    parser.add_argument("--nodata", type=float, default=-9999.0)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.land_cover is not None and args.quality is None:
        raise ValueError("--land-cover requires --quality so the BRDF gaps can be identified.")
    reflectance = _open_dataset(
        _hdf5_dataset(args.vnp09ga, VNP09_GRID, f"SurfReflect_{args.band}_1")
    )
    solar_zenith = _open_dataset(
        _hdf5_dataset(args.vnp09ga, VNP09_GRID, "SolarZenith_1")
    )
    sensor_zenith = _open_dataset(
        _hdf5_dataset(args.vnp09ga, VNP09_GRID, "SensorZenith_1")
    )
    solar_azimuth = _open_dataset(
        _hdf5_dataset(args.vnp09ga, VNP09_GRID, "SolarAzimuth_1")
    )
    sensor_azimuth = _open_dataset(
        _hdf5_dataset(args.vnp09ga, VNP09_GRID, "SensorAzimuth_1")
    )
    f_iso_source = _open_dataset(
        _hdf5_dataset(args.fiso, VNP43D_GRID, f"BRDF_Albedo_Parameter1_{args.band}")
    )
    f_vol_source = _open_dataset(
        _hdf5_dataset(args.fvol, VNP43D_GRID, f"BRDF_Albedo_Parameter2_{args.band}")
    )
    f_geo_source = _open_dataset(
        _hdf5_dataset(args.fgeo, VNP43D_GRID, f"BRDF_Albedo_Parameter3_{args.band}")
    )

    f_iso = _warp_to_reference(f_iso_source, reflectance, args.brdf_resampling)
    f_vol = _warp_to_reference(f_vol_source, reflectance, args.brdf_resampling)
    f_geo = _warp_to_reference(f_geo_source, reflectance, args.brdf_resampling)
    f_iso_values = _read_scaled(f_iso, scale=0.001, fill_value=32767)
    f_vol_values = _read_scaled(f_vol, scale=0.001, fill_value=32767)
    f_geo_values = _read_scaled(f_geo, scale=0.001, fill_value=32767)
    valid_mask = None
    source_map = None
    if args.quality is not None:
        quality_source = _open_dataset(
            _hdf5_dataset(args.quality, VNP43D_GRID, f"BRDF_Albedo_Band_Quality_{args.band}")
        )
        quality = _warp_to_reference(quality_source, reflectance, "near", source_nodata=255)
        quality_values = quality.GetRasterBand(1).ReadAsArray()
        if args.land_cover is None:
            valid_mask = np.isin(quality_values, list(args.valid_quality_values))
        else:
            land_cover_source = _open_mcd12q1_land_cover(args.land_cover)
            land_cover = _warp_native_grid_to_reference(
                land_cover_source, reflectance, source_nodata=FILL_SOURCE_NODATA
            )
            (
                f_iso_values,
                f_vol_values,
                f_geo_values,
                valid_mask,
                source_map,
            ) = fill_brdf_parameters_by_land_cover(
                f_iso_values,
                f_vol_values,
                f_geo_values,
                quality_values,
                land_cover.GetRasterBand(1).ReadAsArray(),
                args.valid_quality_values,
                args.gap_fill_quality_values,
            )
    nbar = normalize_to_nbar(
        _read_scaled(reflectance, scale=0.0001, fill_value=-28672),
        _read_scaled(solar_zenith, scale=0.01, fill_value=-32768),
        _read_scaled(sensor_zenith, scale=0.01, fill_value=-32768),
        _read_scaled(solar_azimuth, scale=0.01, fill_value=-32768),
        _read_scaled(sensor_azimuth, scale=0.01, fill_value=-32768),
        f_iso_values,
        f_vol_values,
        f_geo_values,
        valid_mask,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_raster(args.output, nbar, reflectance, args.nodata)
    if source_map is not None:
        source_map[~np.isfinite(nbar)] = FILL_SOURCE_NODATA
        source_output = args.brdf_source_output or args.output.with_name(
            f"{args.output.stem}_BRDF_source.tif"
        )
        source_output.parent.mkdir(parents=True, exist_ok=True)
        _write_raster(source_output, source_map.astype(np.float64), reflectance, FILL_SOURCE_NODATA)


if __name__ == "__main__":
    main()
