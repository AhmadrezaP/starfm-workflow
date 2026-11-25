"""Align coarse NBAR to a Landsat grid without estimating a spatial shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from osgeo import gdal


def _open_single_band(path: Path) -> gdal.Dataset:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise ValueError(f"GDAL could not open: {path}")
    if dataset.RasterCount != 1:
        raise ValueError(f"Expected one raster band: {path}")
    return dataset


def _reference_bounds(dataset: gdal.Dataset) -> tuple[float, float, float, float]:
    transform = dataset.GetGeoTransform()
    min_x, max_y = transform[0], transform[3]
    max_x = min_x + transform[1] * dataset.RasterXSize
    min_y = max_y + transform[5] * dataset.RasterYSize
    return min_x, min_y, max_x, max_y


def _align_without_shift(
    source_path: Path, landsat: gdal.Dataset, output_path: Path, nodata: float
) -> Path:
    source = _open_single_band(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "GTiff",
        "dstSRS": landsat.GetProjection(),
        "outputBounds": _reference_bounds(landsat),
        "width": landsat.RasterXSize,
        "height": landsat.RasterYSize,
        "resampleAlg": "bilinear",
        "dstNodata": nodata,
        "creationOptions": ["COMPRESS=LZW", "TILED=YES"],
    }
    source_nodata = source.GetRasterBand(1).GetNoDataValue()
    if source_nodata is not None:
        options["srcNodata"] = source_nodata
    aligned = gdal.Warp(str(output_path), source, **options)
    if aligned is None:
        raise RuntimeError(f"GDAL could not align coarse raster to Landsat grid: {source_path}")
    aligned.FlushCache()
    aligned = None
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Landsat-grid alignment of coarse NBAR without geometric shift correction."
    )
    parser.add_argument("--landsat", type=Path, required=True)
    parser.add_argument("--viirs-nbar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--apply-viirs", action="append", nargs=2, metavar=("INPUT", "OUTPUT"), default=[]
    )
    parser.add_argument("--nodata", type=float, default=-9999.0)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    landsat = _open_single_band(args.landsat)
    outputs = [_align_without_shift(args.viirs_nbar, landsat, args.output, args.nodata)]
    for input_name, output_name in args.apply_viirs:
        outputs.append(
            _align_without_shift(Path(input_name), landsat, Path(output_name), args.nodata)
        )
    report = {
        "method": "landsat_grid_alignment_without_coregistration",
        "coregistration_skipped": True,
        "grid_alignment_only": True,
        "resampling": "bilinear",
        "landsat": str(args.landsat.absolute()),
        "coarse_nbar": str(args.viirs_nbar.absolute()),
        "outputs": [str(path.absolute()) for path in outputs],
        "x_shift_pixels": 0.0,
        "y_shift_pixels": 0.0,
        "map_x_shift": 0.0,
        "map_y_shift": 0.0,
    }
    report_path = args.report or args.output.with_suffix(".no_coregistration.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
