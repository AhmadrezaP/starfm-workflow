"""Convert daily MOD09GA directional reflectance to MODIS NBAR.

The command follows the MODIS counterpart of the VIIRS correction used by the
pipeline: daily MOD09GA reflectance is normalized to nadir with the
RossThick-LiSparseReciprocal parameters in MCD43A1.  MCD43A2 quality and
MCD12Q1 IGBP land cover are used to fill non-trusted BRDF parameters from a
same-day class lookup table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from osgeo import gdal

from run_viirs_nbar_hdf import (
    FILL_SOURCE_NODATA,
    _open_mcd12q1_land_cover,
    _warp_native_grid_to_reference,
    fill_brdf_parameters_by_land_cover,
)
from viirs_nbar import _write_raster, normalize_to_nbar


MODIS_BANDS = tuple(range(1, 8))
MOD09_GRID_500M = "MOD_Grid_500m_Surface_Reflectance"
MOD09_GRID_1KM = "MODIS_Grid_1km_2D"


def _open_field(path: Path, field_names: Iterable[str]) -> gdal.Dataset:
    """Open one HDF4-EOS field, tolerating small GDAL field-name variants."""
    container = gdal.Open(str(path))
    if container is not None:
        for field_name in field_names:
            for dataset_path, description in container.GetSubDatasets():
                if dataset_path.endswith(f":{field_name}") or field_name in description:
                    dataset = gdal.Open(dataset_path)
                    if dataset is not None:
                        return dataset
    choices = ", ".join(field_names)
    raise ValueError(
        f"GDAL could not open any of [{choices}] in {path}. "
        "Install GDAL with HDF4-EOS support and verify the MODIS file."
    )


def _read_scaled(
    dataset: gdal.Dataset, scale: float, fill_value: float, band_index: int = 1
) -> np.ndarray:
    values = dataset.GetRasterBand(band_index).ReadAsArray().astype(np.float64) * scale
    values[np.isclose(values, fill_value * scale)] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def _bounds(dataset: gdal.Dataset) -> tuple[float, float, float, float]:
    transform = dataset.GetGeoTransform()
    return (
        transform[0],
        transform[3] + transform[5] * dataset.RasterYSize,
        transform[0] + transform[1] * dataset.RasterXSize,
        transform[3],
    )


def _warp_to_reference(
    source: gdal.Dataset,
    reference: gdal.Dataset,
    *,
    resampling: str,
    source_nodata: float,
) -> gdal.Dataset:
    warped = gdal.Warp(
        "",
        source,
        format="MEM",
        dstSRS=reference.GetProjection(),
        outputBounds=_bounds(reference),
        width=reference.RasterXSize,
        height=reference.RasterYSize,
        resampleAlg=resampling,
        srcNodata=source_nodata,
        dstNodata=source_nodata,
    )
    if warped is None:
        raise RuntimeError("GDAL could not warp the MODIS field onto the MOD09GA reflectance grid.")
    return warped


def _quality_values(values: str) -> set[int]:
    try:
        return {int(value.strip()) for value in values.split(",")}
    except ValueError as error:
        raise argparse.ArgumentTypeError("Quality values must be comma-separated integers.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate MODIS-band NBAR from MOD09GA, MCD43A1, and MCD43A2 HDF files."
    )
    parser.add_argument("--mod09ga", type=Path, required=True, help="Daily MOD09GA Collection 6.1 HDF tile.")
    parser.add_argument("--mcd43a1", type=Path, required=True, help="Same-day MCD43A1 BRDF-parameter HDF tile.")
    parser.add_argument("--mcd43a2", type=Path, required=True, help="Same-day MCD43A2 BRDF-quality HDF tile.")
    parser.add_argument("--land-cover", type=Path, required=True, help="Annual MCD12Q1 HDF tile.")
    parser.add_argument("--band", type=int, choices=MODIS_BANDS, required=True, help="MODIS reflective band, 1 through 7.")
    parser.add_argument("--output", type=Path, required=True, help="Output Float32 MODIS NBAR GeoTIFF.")
    parser.add_argument(
        "--valid-quality-values", type=_quality_values, default={0, 1},
        help="MCD43A2 quality values trusted directly; default: 0,1.",
    )
    parser.add_argument(
        "--gap-fill-quality-values", type=_quality_values, default={2, 3, 4, 5, 6, 7},
        help="MCD43A2 quality values eligible for land-cover BRDF filling; default: 2-7.",
    )
    parser.add_argument("--brdf-source-output", type=Path, help="Optional BRDF-source diagnostic GeoTIFF.")
    parser.add_argument("--nodata", type=float, default=-9999.0)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    band_token = f"b{args.band:02d}"
    reflectance = _open_field(
        args.mod09ga, [f"sur_refl_{band_token}_1", f"sur_refl_{band_token}"]
    )
    solar_zenith = _warp_to_reference(
        _open_field(args.mod09ga, ["SolarZenith_1"]), reflectance,
        resampling="bilinear", source_nodata=-32767,
    )
    sensor_zenith = _warp_to_reference(
        _open_field(args.mod09ga, ["SensorZenith_1"]), reflectance,
        resampling="bilinear", source_nodata=-32767,
    )
    solar_azimuth = _warp_to_reference(
        _open_field(args.mod09ga, ["SolarAzimuth_1"]), reflectance,
        resampling="bilinear", source_nodata=-32767,
    )
    sensor_azimuth = _warp_to_reference(
        _open_field(args.mod09ga, ["SensorAzimuth_1"]), reflectance,
        resampling="bilinear", source_nodata=-32767,
    )
    parameters = _warp_to_reference(
        _open_field(args.mcd43a1, [f"BRDF_Albedo_Parameters_Band{args.band}"]),
        reflectance, resampling="bilinear", source_nodata=32767,
    )
    if parameters.RasterCount < 3:
        raise ValueError("MCD43A1 BRDF parameter field must contain iso, vol, and geo bands.")
    quality = _warp_to_reference(
        _open_field(args.mcd43a2, [f"BRDF_Albedo_Band_Quality_Band{args.band}"]),
        reflectance, resampling="near", source_nodata=255,
    )
    land_cover = _warp_native_grid_to_reference(
        _open_mcd12q1_land_cover(args.land_cover), reflectance, source_nodata=FILL_SOURCE_NODATA
    )
    f_iso, f_vol, f_geo, valid_mask, source_map = fill_brdf_parameters_by_land_cover(
        _read_scaled(parameters, 0.001, 32767, 1),
        _read_scaled(parameters, 0.001, 32767, 2),
        _read_scaled(parameters, 0.001, 32767, 3),
        quality.GetRasterBand(1).ReadAsArray(),
        land_cover.GetRasterBand(1).ReadAsArray(),
        args.valid_quality_values,
        args.gap_fill_quality_values,
    )
    nbar = normalize_to_nbar(
        _read_scaled(reflectance, 0.0001, -28672),
        _read_scaled(solar_zenith, 0.01, -32767),
        _read_scaled(sensor_zenith, 0.01, -32767),
        _read_scaled(solar_azimuth, 0.01, -32767),
        _read_scaled(sensor_azimuth, 0.01, -32767),
        f_iso, f_vol, f_geo, valid_mask,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_raster(args.output, nbar, reflectance, args.nodata)
    source_map[~np.isfinite(nbar)] = FILL_SOURCE_NODATA
    source_output = args.brdf_source_output or args.output.with_name(f"{args.output.stem}_BRDF_source.tif")
    _write_raster(source_output, source_map.astype(np.float64), reflectance, FILL_SOURCE_NODATA)


if __name__ == "__main__":
    main()
