"""Coregister daily VIIRS NBAR to a Landsat 8/9 image using the paper method.

The implementation follows the global-shift procedure in Wang et al. (2014):
reproject the daily coarse NBAR image to the Landsat grid, test a bounded set
of integer row/column shifts, and select the shift having the highest Pearson
correlation in a representative matching window.  Landsat is always the fixed
reference; VIIRS is the image that is shifted.

This is deliberately a *registration* tool, not a cloud-gap-filling tool.  Use
a cloud-free paired date, or provide a precomputed valid-pixel mask for the
correlation calculation.  The corrected output retains every finite VIIRS
observation; the mask is used only to make the estimated shift robust.  Once
estimated from a matching band pair (normally Landsat B4 and VIIRS M5), the
same shift can be applied to the other VIIRS NBAR bands.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from osgeo import gdal


@dataclass(frozen=True)
class ShiftResult:
    """The result of an exhaustive integer-pixel correlation search."""

    x_pixels: int
    y_pixels: int
    correlation: float
    sample_count: int
    score_map: np.ndarray


def apply_integer_shift(values: np.ndarray, x_pixels: int, y_pixels: int) -> np.ndarray:
    """Translate an array, filling areas exposed at the edge with NaN.

    A positive x shift moves features right (east on north-up grids); a
    positive y shift moves features down (south on north-up grids).
    """
    source = np.asarray(values, dtype=np.float64)
    output = np.full(source.shape, np.nan, dtype=np.float64)
    height, width = source.shape
    out_y0, out_y1 = max(0, y_pixels), min(height, height + y_pixels)
    out_x0, out_x1 = max(0, x_pixels), min(width, width + x_pixels)
    if out_y0 >= out_y1 or out_x0 >= out_x1:
        return output
    output[out_y0:out_y1, out_x0:out_x1] = source[
        out_y0 - y_pixels : out_y1 - y_pixels,
        out_x0 - x_pixels : out_x1 - x_pixels,
    ]
    return output


def _pearson_correlation(reference: np.ndarray, moving: np.ndarray, valid: np.ndarray) -> tuple[float, int]:
    """Return Pearson correlation and number of jointly valid pixels."""
    pixels = valid & np.isfinite(reference) & np.isfinite(moving)
    count = int(np.count_nonzero(pixels))
    if count == 0:
        return np.nan, count
    reference_values = reference[pixels]
    moving_values = moving[pixels]
    reference_centered = reference_values - reference_values.mean()
    moving_centered = moving_values - moving_values.mean()
    denominator = np.sqrt(np.sum(reference_centered**2) * np.sum(moving_centered**2))
    if denominator == 0.0:
        return np.nan, count
    return float(np.sum(reference_centered * moving_centered) / denominator), count


def search_integer_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift_pixels: int,
    minimum_samples: int,
    valid_mask: np.ndarray | None = None,
) -> ShiftResult:
    """Exhaustively choose the VIIRS translation with maximum correlation.

    ``x_pixels`` and ``y_pixels`` describe the translation to apply to the
    moving VIIRS array so that it aligns to the fixed Landsat reference.
    """
    reference_array = np.asarray(reference, dtype=np.float64)
    moving_array = np.asarray(moving, dtype=np.float64)
    if reference_array.shape != moving_array.shape:
        raise ValueError("Reference and moving arrays must have the same shape.")
    if reference_array.ndim != 2:
        raise ValueError("Coregistration inputs must be two-dimensional single-band arrays.")
    if max_shift_pixels < 0:
        raise ValueError("max_shift_pixels must be non-negative.")
    if minimum_samples < 2:
        raise ValueError("minimum_samples must be at least 2.")

    valid = np.ones(reference_array.shape, dtype=bool)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
        if valid.shape != reference_array.shape:
            raise ValueError("The valid-pixel mask must have the same shape as the inputs.")

    search_width = 2 * max_shift_pixels + 1
    score_map = np.full((search_width, search_width), np.nan, dtype=np.float64)
    best: tuple[float, int, int, int] | None = None
    for y_pixels in range(-max_shift_pixels, max_shift_pixels + 1):
        for x_pixels in range(-max_shift_pixels, max_shift_pixels + 1):
            shifted = apply_integer_shift(moving_array, x_pixels, y_pixels)
            correlation, count = _pearson_correlation(reference_array, shifted, valid)
            score_map[y_pixels + max_shift_pixels, x_pixels + max_shift_pixels] = correlation
            if count < minimum_samples or not np.isfinite(correlation):
                continue
            if best is None or correlation > best[0]:
                best = (correlation, x_pixels, y_pixels, count)

    if best is None:
        raise ValueError(
            "No candidate shift had enough valid overlapping pixels. "
            "Use a larger window, a less restrictive mask, or a smaller minimum-samples value."
        )
    correlation, x_pixels, y_pixels, count = best
    return ShiftResult(x_pixels, y_pixels, correlation, count, score_map)


def _reference_bounds(dataset: gdal.Dataset) -> tuple[float, float, float, float]:
    transform = dataset.GetGeoTransform()
    min_x, max_y = transform[0], transform[3]
    max_x = min_x + transform[1] * dataset.RasterXSize
    min_y = max_y + transform[5] * dataset.RasterYSize
    return min_x, min_y, max_x, max_y


def _open(path: Path) -> gdal.Dataset:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise ValueError(f"GDAL could not open: {path}")
    if dataset.RasterCount != 1:
        raise ValueError(f"Expected one raster band: {path}")
    return dataset


def _warp_to_landsat_grid(source: gdal.Dataset, landsat: gdal.Dataset, resampling: str) -> gdal.Dataset:
    warped = gdal.Warp(
        "",
        source,
        format="MEM",
        dstSRS=landsat.GetProjection(),
        outputBounds=_reference_bounds(landsat),
        width=landsat.RasterXSize,
        height=landsat.RasterYSize,
        resampleAlg=resampling,
        srcNodata=source.GetRasterBand(1).GetNoDataValue(),
        dstNodata=np.nan,
    )
    if warped is None:
        raise RuntimeError("GDAL could not reproject VIIRS NBAR to the Landsat grid.")
    return warped


def _read_values(dataset: gdal.Dataset, scale: float, offset: float) -> np.ndarray:
    band = dataset.GetRasterBand(1)
    raw = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    values = raw * scale + offset
    if nodata is not None and np.isfinite(nodata):
        values[np.isclose(raw, nodata)] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def _central_window(shape: tuple[int, int], window_size: int) -> tuple[slice, slice]:
    height, width = shape
    window_height = min(window_size, height)
    window_width = min(window_size, width)
    y0 = (height - window_height) // 2
    x0 = (width - window_width) // 2
    return slice(y0, y0 + window_height), slice(x0, x0 + window_width)


def _parse_window(value: str) -> tuple[int, int, int, int]:
    try:
        x_offset, y_offset, width, height = (int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("A window must be x_offset,y_offset,width,height.") from error
    if min(x_offset, y_offset) < 0 or min(width, height) < 1:
        raise argparse.ArgumentTypeError("Window offsets must be non-negative and dimensions positive.")
    return x_offset, y_offset, width, height


def _window_slices(
    shape: tuple[int, int],
    window: tuple[int, int, int, int] | None,
    default_size: int,
) -> tuple[slice, slice, dict[str, int]]:
    if window is None:
        y_slice, x_slice = _central_window(shape, default_size)
    else:
        x_offset, y_offset, width, height = window
        if x_offset + width > shape[1] or y_offset + height > shape[0]:
            raise ValueError("The requested correlation window falls outside the Landsat image.")
        x_slice = slice(x_offset, x_offset + width)
        y_slice = slice(y_offset, y_offset + height)
    metadata = {
        "x_offset": x_slice.start,
        "y_offset": y_slice.start,
        "width": x_slice.stop - x_slice.start,
        "height": y_slice.stop - y_slice.start,
    }
    return y_slice, x_slice, metadata


def _read_mask(path: Path, landsat: gdal.Dataset) -> np.ndarray:
    mask = _open(path)
    if (mask.RasterXSize, mask.RasterYSize) != (landsat.RasterXSize, landsat.RasterYSize):
        raise ValueError("The valid-pixel mask dimensions must match Landsat exactly.")
    if mask.GetGeoTransform() != landsat.GetGeoTransform() or mask.GetProjection() != landsat.GetProjection():
        raise ValueError("The valid-pixel mask grid and CRS must match Landsat exactly.")
    return mask.GetRasterBand(1).ReadAsArray() != 0


def _write_output(path: Path, values: np.ndarray, reference: gdal.Dataset, nodata: float) -> None:
    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        str(path),
        reference.RasterXSize,
        reference.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if output is None:
        raise RuntimeError(f"GDAL could not create: {path}")
    output.SetGeoTransform(reference.GetGeoTransform())
    output.SetProjection(reference.GetProjection())
    band = output.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(np.where(np.isfinite(values), values, nodata).astype(np.float32))
    band.FlushCache()
    output.FlushCache()


def _write_score_map(path: Path, score_map: np.ndarray, max_shift_pixels: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["y_shift_pixels\\x_shift_pixels", *range(-max_shift_pixels, max_shift_pixels + 1)])
        for index, y_pixels in enumerate(range(-max_shift_pixels, max_shift_pixels + 1)):
            writer.writerow([y_pixels, *score_map[index].tolist()])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-style global coregistration of VIIRS NBAR to Landsat 8/9."
    )
    parser.add_argument("--landsat", type=Path, required=True, help="Single-band Landsat 8/9 paired-date GeoTIFF (usually red B4).")
    parser.add_argument("--viirs-nbar", type=Path, required=True, help="Single-band paired-date VIIRS NBAR GeoTIFF (usually M5).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit shifted-VIIRS GeoTIFF path. Overrides --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("COREGISTRATION_FILES/OUTPUTS/MOVING_WINDOW"),
        help="Directory for the default GeoTIFF, JSON report, and score CSV.",
    )
    parser.add_argument(
        "--apply-viirs",
        action="append",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        default=[],
        help="Additional VIIRS NBAR input/output pair to receive the M5-derived shift. Repeat for each band.",
    )
    parser.add_argument("--landsat-scale", type=float, default=0.0000275, help="Landsat C2 scale factor; default: 0.0000275.")
    parser.add_argument("--landsat-offset", type=float, default=-0.2, help="Landsat C2 offset; default: -0.2.")
    parser.add_argument("--viirs-scale", type=float, default=1.0, help="VIIRS NBAR scale factor; default: 1.")
    parser.add_argument("--viirs-offset", type=float, default=0.0, help="VIIRS NBAR offset; default: 0.")
    parser.add_argument("--valid-mask", type=Path, help="Optional 0/1 valid-pixel mask on the Landsat grid, used only while estimating the shift.")
    parser.add_argument("--max-shift-pixels", type=int, default=20, help="Maximum Landsat-grid shift tested in either direction; default: 20.")
    parser.add_argument("--minimum-samples", type=int, default=5000, help="Minimum jointly valid pixels for a candidate; default: 5000.")
    parser.add_argument("--window", type=_parse_window, help="Optional correlation window: x_offset,y_offset,width,height. Default: centered square.")
    parser.add_argument("--window-size", type=int, default=512, help="Default centered correlation-window size in Landsat pixels; default: 512.")
    parser.add_argument("--resampling", choices=("near", "bilinear", "cubic"), default="bilinear", help="VIIRS-to-Landsat resampling used before matching; default: bilinear.")
    parser.add_argument("--nodata", type=float, default=-9999.0)
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    parser.add_argument("--score-map", type=Path, help="Optional CSV path for all tested correlations.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.window_size < 1:
        raise ValueError("--window-size must be positive.")

    landsat = _open(args.landsat)
    viirs = _open(args.viirs_nbar)
    output_path = args.output or args.output_dir / f"{args.viirs_nbar.stem}_coregistered.tif"
    viirs_on_landsat_grid = _warp_to_landsat_grid(viirs, landsat, args.resampling)
    landsat_values = _read_values(landsat, args.landsat_scale, args.landsat_offset)
    viirs_values = _read_values(viirs_on_landsat_grid, args.viirs_scale, args.viirs_offset)
    valid_mask = _read_mask(args.valid_mask, landsat) if args.valid_mask else None

    y_slice, x_slice, window_metadata = _window_slices(landsat_values.shape, args.window, args.window_size)
    analysis_valid_mask = (
        valid_mask[y_slice, x_slice]
        if valid_mask is not None
        else np.ones(landsat_values[y_slice, x_slice].shape, dtype=bool)
    )
    result = search_integer_shift(
        landsat_values[y_slice, x_slice],
        viirs_values[y_slice, x_slice],
        args.max_shift_pixels,
        args.minimum_samples,
        analysis_valid_mask,
    )
    shifted_viirs = apply_integer_shift(viirs_values, result.x_pixels, result.y_pixels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_output(output_path, shifted_viirs, landsat, args.nodata)

    applied_outputs = [str(output_path.absolute())]
    for input_path, additional_output_path in args.apply_viirs:
        additional_input = Path(input_path)
        additional_output = Path(additional_output_path)
        additional_viirs = _warp_to_landsat_grid(
            _open(additional_input), landsat, args.resampling
        )
        additional_values = _read_values(
            additional_viirs, args.viirs_scale, args.viirs_offset
        )
        additional_output.parent.mkdir(parents=True, exist_ok=True)
        _write_output(
            additional_output,
            apply_integer_shift(additional_values, result.x_pixels, result.y_pixels),
            landsat,
            args.nodata,
        )
        applied_outputs.append(str(additional_output.absolute()))

    transform = landsat.GetGeoTransform()
    initial_correlation, initial_count = _pearson_correlation(
        landsat_values[y_slice, x_slice],
        viirs_values[y_slice, x_slice],
        analysis_valid_mask,
    )
    finite_scores = result.score_map[np.isfinite(result.score_map)]
    sorted_scores = np.sort(finite_scores)
    runner_up = float(sorted_scores[-2]) if sorted_scores.size > 1 else np.nan
    report = {
        "method": "exhaustive_integer_shift_maximum_pearson_correlation",
        "landsat": str(args.landsat.absolute()),
        "viirs_nbar": str(args.viirs_nbar.absolute()),
        "outputs": applied_outputs,
        "correlation_window_landsat_pixels": window_metadata,
        "max_shift_pixels": args.max_shift_pixels,
        "x_shift_pixels": result.x_pixels,
        "y_shift_pixels": result.y_pixels,
        "map_x_shift": result.x_pixels * transform[1],
        "map_y_shift": result.y_pixels * transform[5],
        "initial_correlation": initial_correlation,
        "initial_sample_count": initial_count,
        "best_correlation": result.correlation,
        "best_sample_count": result.sample_count,
        "correlation_improvement": result.correlation - initial_correlation,
        "runner_up_correlation": runner_up,
        "peak_separation": result.correlation - runner_up,
    }
    report_path = args.report or output_path.with_suffix(".coregistration.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    score_map_path = args.score_map or output_path.with_suffix(".correlation_scores.csv")
    score_map_path.parent.mkdir(parents=True, exist_ok=True)
    _write_score_map(score_map_path, result.score_map, args.max_shift_pixels)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
