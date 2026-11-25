"""Helpers for using a valid-pixel mask only while estimating a shift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from osgeo import gdal


def _open(path: Path) -> gdal.Dataset:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise ValueError(f"GDAL could not open: {path}")
    if dataset.RasterCount != 1:
        raise ValueError(f"Expected one raster band: {path}")
    return dataset


def _same_grid(first: gdal.Dataset, second: gdal.Dataset) -> bool:
    return (
        (first.RasterXSize, first.RasterYSize) == (second.RasterXSize, second.RasterYSize)
        and first.GetGeoTransform() == second.GetGeoTransform()
        and first.GetProjection() == second.GetProjection()
    )


def masked_copy(source_path: Path, mask_path: Path, output_path: Path, nodata: float) -> Path:
    """Copy a grid-aligned raster, replacing invalid matching pixels with nodata.

    This function intentionally creates an estimation-only image.  It never
    modifies the source imagery used for the final registered output.
    """
    source = _open(source_path)
    mask = _open(mask_path)
    if not _same_grid(source, mask):
        raise ValueError(f"Mask grid does not match source raster: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = gdal.GetDriverByName("GTiff").Create(
        str(output_path), source.RasterXSize, source.RasterYSize, 1, source.GetRasterBand(1).DataType,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if output is None:
        raise RuntimeError(f"GDAL could not create: {output_path}")
    output.SetGeoTransform(source.GetGeoTransform())
    output.SetProjection(source.GetProjection())
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(nodata)
    source_band = source.GetRasterBand(1)
    mask_band = mask.GetRasterBand(1)
    for y_offset in range(0, source.RasterYSize, 512):
        height = min(512, source.RasterYSize - y_offset)
        values = source_band.ReadAsArray(0, y_offset, source.RasterXSize, height)
        valid = mask_band.ReadAsArray(0, y_offset, source.RasterXSize, height).astype(bool)
        values[~valid] = nodata
        output_band.WriteArray(values, 0, y_offset)
    output_band.FlushCache()
    output.FlushCache()
    return output_path
