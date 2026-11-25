"""Normalize daily directional VIIRS reflectance to NBAR.

This module applies the RossThick-LiSparseReciprocal BRDF model used by the
VIIRS BRDF/Albedo products.  Given a daily VNP09GA reflectance image, its
per-pixel sun/view geometry, and the three matching VNP43 BRDF parameters, it
calculates the reflectance that would have been observed at nadir.

All input rasters must be on exactly the same grid.  In particular, use the
1 km VNP09GA M-band layers with the 1 km VNP43M parameters, and the 500 m
VNP09GA I-band layers with the 500 m VNP43I parameters.  The CLI deliberately
does not resample inputs: resampling is a separate geospatial operation and
must occur only after this physical normalization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


LI_SPARSE_RECIPROCAL_HEIGHT_TO_BASE = 2.0
EPSILON = 1.0e-8


def _as_radians(values: np.ndarray | float) -> np.ndarray:
    """Return floating-point angles in radians."""
    return np.deg2rad(np.asarray(values, dtype=np.float64))


def _relative_azimuth(solar_azimuth_deg: np.ndarray, sensor_azimuth_deg: np.ndarray) -> np.ndarray:
    """Return relative azimuth in the [0, 180] degree interval."""
    difference = np.abs(np.asarray(solar_azimuth_deg) - np.asarray(sensor_azimuth_deg)) % 360.0
    return np.minimum(difference, 360.0 - difference)


def ross_thick_kernel(
    solar_zenith_deg: np.ndarray | float,
    sensor_zenith_deg: np.ndarray | float,
    relative_azimuth_deg: np.ndarray | float,
) -> np.ndarray:
    """Evaluate the RossThick volumetric scattering kernel."""
    solar_zenith = _as_radians(solar_zenith_deg)
    sensor_zenith = _as_radians(sensor_zenith_deg)
    relative_azimuth = _as_radians(relative_azimuth_deg)

    cos_phase = (
        np.cos(solar_zenith) * np.cos(sensor_zenith)
        + np.sin(solar_zenith) * np.sin(sensor_zenith) * np.cos(relative_azimuth)
    )
    phase = np.arccos(np.clip(cos_phase, -1.0, 1.0))
    denominator = np.cos(solar_zenith) + np.cos(sensor_zenith)
    return (
        ((np.pi / 2.0 - phase) * np.cos(phase) + np.sin(phase)) / denominator
        - np.pi / 4.0
    )


def li_sparse_reciprocal_kernel(
    solar_zenith_deg: np.ndarray | float,
    sensor_zenith_deg: np.ndarray | float,
    relative_azimuth_deg: np.ndarray | float,
    height_to_base: float = LI_SPARSE_RECIPROCAL_HEIGHT_TO_BASE,
) -> np.ndarray:
    """Evaluate the LiSparseReciprocal geometric scattering kernel.

    ``height_to_base=2`` is the standard value used by the MODIS/VIIRS
    RossThick-LiSparseReciprocal BRDF model.
    """
    solar_zenith = _as_radians(solar_zenith_deg)
    sensor_zenith = _as_radians(sensor_zenith_deg)
    relative_azimuth = _as_radians(relative_azimuth_deg)

    tan_solar = np.tan(solar_zenith)
    tan_sensor = np.tan(sensor_zenith)
    sec_solar = 1.0 / np.cos(solar_zenith)
    sec_sensor = 1.0 / np.cos(sensor_zenith)
    distance = np.sqrt(
        tan_solar**2
        + tan_sensor**2
        - 2.0 * tan_solar * tan_sensor * np.cos(relative_azimuth)
    )
    cos_overlap = height_to_base * np.sqrt(
        distance**2 + (tan_solar * tan_sensor * np.sin(relative_azimuth)) ** 2
    ) / (sec_solar + sec_sensor)
    overlap_angle = np.arccos(np.clip(cos_overlap, -1.0, 1.0))
    overlap = (
        (overlap_angle - np.sin(overlap_angle) * np.cos(overlap_angle))
        * (sec_solar + sec_sensor)
        / np.pi
    )
    cos_phase = (
        np.cos(solar_zenith) * np.cos(sensor_zenith)
        + np.sin(solar_zenith) * np.sin(sensor_zenith) * np.cos(relative_azimuth)
    )
    return overlap - sec_solar - sec_sensor + 0.5 * (1.0 + cos_phase) * sec_solar * sec_sensor


def normalize_to_nbar(
    directional_reflectance: np.ndarray,
    solar_zenith_deg: np.ndarray,
    sensor_zenith_deg: np.ndarray,
    solar_azimuth_deg: np.ndarray,
    sensor_azimuth_deg: np.ndarray,
    f_iso: np.ndarray,
    f_vol: np.ndarray,
    f_geo: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert directional surface reflectance to nadir BRDF-adjusted reflectance.

    The correction is a multiplicative magnitude inversion:

    ``rho_nbar = rho_directional * BRDF(sun, nadir) / BRDF(sun, view)``.

    Invalid input pixels and pixels with a near-zero directional BRDF magnitude
    are returned as NaN.  Inputs are physical values: reflectance is unitless,
    angles are degrees, and BRDF parameters are already scale-factor adjusted.
    """
    arrays = [
        np.asarray(array, dtype=np.float64)
        for array in (
            directional_reflectance,
            solar_zenith_deg,
            sensor_zenith_deg,
            solar_azimuth_deg,
            sensor_azimuth_deg,
            f_iso,
            f_vol,
            f_geo,
        )
    ]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ValueError("All reflectance, geometry, and BRDF arrays must have the same shape.")

    (
        directional_reflectance,
        solar_zenith_deg,
        sensor_zenith_deg,
        solar_azimuth_deg,
        sensor_azimuth_deg,
        f_iso,
        f_vol,
        f_geo,
    ) = arrays

    relative_azimuth_deg = _relative_azimuth(solar_azimuth_deg, sensor_azimuth_deg)
    directional_magnitude = (
        f_iso
        + f_vol * ross_thick_kernel(solar_zenith_deg, sensor_zenith_deg, relative_azimuth_deg)
        + f_geo * li_sparse_reciprocal_kernel(solar_zenith_deg, sensor_zenith_deg, relative_azimuth_deg)
    )
    nadir_magnitude = (
        f_iso
        + f_vol * ross_thick_kernel(solar_zenith_deg, 0.0, 0.0)
        + f_geo * li_sparse_reciprocal_kernel(solar_zenith_deg, 0.0, 0.0)
    )

    valid = np.isfinite(directional_reflectance)
    valid &= np.isfinite(directional_magnitude) & np.isfinite(nadir_magnitude)
    valid &= np.abs(directional_magnitude) > EPSILON
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    normalized = np.full(shape, np.nan, dtype=np.float64)
    normalized[valid] = directional_reflectance[valid] * (
        nadir_magnitude[valid] / directional_magnitude[valid]
    )
    return normalized


def _require_gdal():
    try:
        from osgeo import gdal
    except ImportError as error:  # pragma: no cover - only relevant to CLI environments
        raise RuntimeError("GDAL's Python bindings are required to read and write rasters.") from error
    return gdal


def _read_raster(path: Path, scale: float, fill_value: float | None) -> tuple[np.ndarray, object]:
    gdal = _require_gdal()
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise ValueError(f"Could not open raster: {path}")
    if dataset.RasterCount != 1:
        raise ValueError(f"Expected a single-band raster, found {dataset.RasterCount}: {path}")

    band = dataset.GetRasterBand(1)
    values = band.ReadAsArray().astype(np.float64) * scale
    nodata = band.GetNoDataValue()
    invalid = ~np.isfinite(values)
    if nodata is not None:
        invalid |= np.isclose(values, nodata * scale)
    if fill_value is not None:
        invalid |= np.isclose(values, fill_value * scale)
    values[invalid] = np.nan
    return values, dataset


def _assert_same_grid(reference_dataset: object, dataset: object, path: Path) -> None:
    if (reference_dataset.RasterXSize, reference_dataset.RasterYSize) != (
        dataset.RasterXSize,
        dataset.RasterYSize,
    ):
        raise ValueError(f"Raster dimensions do not match the directional reflectance: {path}")
    if reference_dataset.GetGeoTransform() != dataset.GetGeoTransform():
        raise ValueError(f"Raster geotransform does not match the directional reflectance: {path}")
    if reference_dataset.GetProjection() != dataset.GetProjection():
        raise ValueError(f"Raster CRS does not match the directional reflectance: {path}")


def _read_aligned(
    path: Path,
    reference_dataset: object,
    scale: float,
    fill_value: float | None,
) -> np.ndarray:
    values, dataset = _read_raster(path, scale, fill_value)
    _assert_same_grid(reference_dataset, dataset, path)
    return values


def _write_raster(path: Path, values: np.ndarray, reference_dataset: object, nodata: float) -> None:
    gdal = _require_gdal()
    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        str(path),
        reference_dataset.RasterXSize,
        reference_dataset.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if output is None:
        raise ValueError(f"Could not create output raster: {path}")
    output.SetGeoTransform(reference_dataset.GetGeoTransform())
    output.SetProjection(reference_dataset.GetProjection())
    band = output.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(np.where(np.isfinite(values), values, nodata).astype(np.float32))
    band.FlushCache()
    output.FlushCache()


def _parse_quality_values(values: str) -> set[int]:
    try:
        return {int(value.strip()) for value in values.split(",")}
    except ValueError as error:
        raise argparse.ArgumentTypeError("Quality values must be comma-separated integers, e.g. '0,1'.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a daily directional VIIRS reflectance raster to VIIRS NBAR."
    )
    parser.add_argument("--reflectance", type=Path, required=True, help="Daily VNP09GA reflectance raster.")
    parser.add_argument("--solar-zenith", type=Path, required=True, help="Daily solar-zenith raster (degrees).")
    parser.add_argument("--sensor-zenith", type=Path, required=True, help="Daily sensor-zenith raster (degrees).")
    parser.add_argument("--solar-azimuth", type=Path, required=True, help="Daily solar-azimuth raster (degrees).")
    parser.add_argument("--sensor-azimuth", type=Path, required=True, help="Daily sensor-azimuth raster (degrees).")
    parser.add_argument("--fiso", type=Path, required=True, help="VNP43 BRDF isotropic-parameter raster.")
    parser.add_argument("--fvol", type=Path, required=True, help="VNP43 BRDF volumetric-parameter raster.")
    parser.add_argument("--fgeo", type=Path, required=True, help="VNP43 BRDF geometric-parameter raster.")
    parser.add_argument("--quality", type=Path, help="Optional VNP43 per-band mandatory-quality raster.")
    parser.add_argument(
        "--valid-quality-values",
        type=_parse_quality_values,
        default={0, 1},
        help="VNP43 quality values to retain; default: 0,1.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output Float32 NBAR GeoTIFF.")
    parser.add_argument("--reflectance-scale", type=float, default=0.0001)
    parser.add_argument("--reflectance-fill", type=float, default=-28672)
    parser.add_argument("--angle-scale", type=float, default=0.01)
    parser.add_argument("--angle-fill", type=float, default=-32768)
    parser.add_argument("--parameter-scale", type=float, default=0.001)
    parser.add_argument("--parameter-fill", type=float, default=32767)
    parser.add_argument("--quality-fill", type=float, default=255)
    parser.add_argument("--nodata", type=float, default=-9999.0)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """Run the command-line NBAR normalization."""
    args = build_parser().parse_args(argv)
    reflectance, reference = _read_raster(
        args.reflectance, args.reflectance_scale, args.reflectance_fill
    )
    solar_zenith = _read_aligned(args.solar_zenith, reference, args.angle_scale, args.angle_fill)
    sensor_zenith = _read_aligned(args.sensor_zenith, reference, args.angle_scale, args.angle_fill)
    solar_azimuth = _read_aligned(args.solar_azimuth, reference, args.angle_scale, args.angle_fill)
    sensor_azimuth = _read_aligned(args.sensor_azimuth, reference, args.angle_scale, args.angle_fill)
    f_iso = _read_aligned(args.fiso, reference, args.parameter_scale, args.parameter_fill)
    f_vol = _read_aligned(args.fvol, reference, args.parameter_scale, args.parameter_fill)
    f_geo = _read_aligned(args.fgeo, reference, args.parameter_scale, args.parameter_fill)

    valid_mask = None
    if args.quality is not None:
        quality = _read_aligned(args.quality, reference, 1.0, args.quality_fill)
        valid_mask = np.isin(quality, list(args.valid_quality_values))

    nbar = normalize_to_nbar(
        reflectance,
        solar_zenith,
        sensor_zenith,
        solar_azimuth,
        sensor_azimuth,
        f_iso,
        f_vol,
        f_geo,
        valid_mask,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_raster(args.output, nbar, reference, args.nodata)


if __name__ == "__main__":
    main()
