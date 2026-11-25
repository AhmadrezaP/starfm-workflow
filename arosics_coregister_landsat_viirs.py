"""Run global AROSICS coregistration for a Landsat 8/9 and VIIRS NBAR pair.

This script is an independent comparison with
``paper_coregister_landsat_viirs.py``.  It holds the Landsat image fixed and
uses AROSICS' frequency-domain global-shift estimator to move daily VIIRS
NBAR onto the Landsat grid.  By default its output GeoTIFF and JSON diagnostics
are stored in ``COREGISTRATION_FILES/OUTPUTS/AROSICS``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

# The desktop environment's home directory is intentionally read-only.  Set a
# writable Matplotlib cache before importing AROSICS and its plotting stack.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "arosics_mpl"))

from arosics import COREG, __version__ as AROSICS_VERSION
from arosics.DeShifter import DESHIFTER
from osgeo import gdal

from coregistration_masking import masked_copy


def _open_single_band(path: Path) -> gdal.Dataset:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise ValueError(f"GDAL could not open: {path}")
    if dataset.RasterCount != 1:
        raise ValueError(f"Expected one raster band: {path}")
    return dataset


def _parse_window(value: str) -> tuple[int, int, int, int]:
    try:
        x_offset, y_offset, width, height = (int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("A window must be x_offset,y_offset,width,height.") from error
    if min(x_offset, y_offset) < 0 or min(width, height) < 1:
        raise argparse.ArgumentTypeError("Window offsets must be non-negative and dimensions positive.")
    return x_offset, y_offset, width, height


def _matching_window_center(
    landsat: gdal.Dataset,
    window_size: int,
    window: tuple[int, int, int, int] | None,
) -> tuple[tuple[float, float], tuple[int, int], dict[str, int]]:
    """Return the map-coordinate centre of the same central window used by the paper method."""
    if window_size < 1:
        raise ValueError("--window-size must be positive.")
    if window is None:
        width = min(window_size, landsat.RasterXSize)
        height = min(window_size, landsat.RasterYSize)
        x_offset = (landsat.RasterXSize - width) // 2
        y_offset = (landsat.RasterYSize - height) // 2
    else:
        x_offset, y_offset, width, height = window
        if x_offset + width > landsat.RasterXSize or y_offset + height > landsat.RasterYSize:
            raise ValueError("The requested matching window falls outside the Landsat image.")
    transform = landsat.GetGeoTransform()
    center_x = transform[0] + (x_offset + width / 2.0) * transform[1]
    center_y = transform[3] + (y_offset + height / 2.0) * transform[5]
    return (
        (center_x, center_y),
        (width, height),
        {"x_offset": x_offset, "y_offset": y_offset, "width": width, "height": height},
    )


def _reference_bounds(dataset: gdal.Dataset) -> tuple[float, float, float, float]:
    transform = dataset.GetGeoTransform()
    min_x, max_y = transform[0], transform[3]
    max_x = min_x + transform[1] * dataset.RasterXSize
    min_y = max_y + transform[5] * dataset.RasterYSize
    return min_x, min_y, max_x, max_y


def _warp_viirs_to_landsat_grid(
    viirs_path: Path,
    landsat: gdal.Dataset,
    output_path: Path,
    nodata: float,
) -> Path:
    """Persist a bilinearly resampled VIIRS input on the exact Landsat grid."""
    viirs = _open_single_band(viirs_path)
    already_aligned = (
        viirs.GetProjection() == landsat.GetProjection()
        and viirs.GetGeoTransform() == landsat.GetGeoTransform()
        and (viirs.RasterXSize, viirs.RasterYSize) == (landsat.RasterXSize, landsat.RasterYSize)
    )
    if already_aligned:
        return viirs_path

    warped = gdal.Warp(
        str(output_path),
        viirs,
        format="GTiff",
        dstSRS=landsat.GetProjection(),
        outputBounds=_reference_bounds(landsat),
        width=landsat.RasterXSize,
        height=landsat.RasterYSize,
        resampleAlg="bilinear",
        srcNodata=viirs.GetRasterBand(1).GetNoDataValue(),
        dstNodata=nodata,
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
    )
    if warped is None:
        raise RuntimeError("GDAL could not warp VIIRS NBAR to the Landsat grid for AROSICS.")
    warped.FlushCache()
    return output_path


def _apply_detected_shift(
    source_path: Path,
    landsat: gdal.Dataset,
    coregistration: COREG,
    output_path: Path,
    nodata: float,
) -> Path:
    """Apply the reference-pair AROSICS correction to another coarse raster."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_source = _warp_viirs_to_landsat_grid(
        source_path,
        landsat,
        output_path.with_name(f"{source_path.stem}_on_landsat_grid.tif"),
        nodata,
    )
    DESHIFTER(
        str(aligned_source),
        coregistration.coreg_info,
        path_out=str(output_path),
        fmt_out="GTIFF",
        out_crea_options=["COMPRESS=LZW", "TILED=YES"],
        nodata=nodata,
        align_grids=True,
        match_gsd=True,
        resamp_alg="bilinear",
        q=True,
    ).correct_shifts()
    if not output_path.is_file():
        raise RuntimeError(f"AROSICS did not create the shifted coarse raster: {output_path}")
    return output_path


def _json_safe(value: Any) -> Any:
    """Convert AROSICS/GDAL values in ``coreg_info`` to compact JSON data."""
    if isinstance(value, float):
        return value if isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Global AROSICS coregistration of VIIRS NBAR to a Landsat 8/9 paired image."
    )
    parser.add_argument("--landsat", type=Path, required=True, help="Single-band Landsat paired-date GeoTIFF (B2 for this comparison).")
    parser.add_argument("--viirs-nbar", type=Path, required=True, help="Single-band paired-date VIIRS NBAR GeoTIFF (M3 for this comparison).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("COREGISTRATION_FILES/OUTPUTS/AROSICS"),
        help="Directory for the registered GeoTIFF and JSON diagnostic report.",
    )
    parser.add_argument("--output", type=Path, help="Optional explicit output GeoTIFF path. Overrides --output-dir.")
    parser.add_argument(
        "--apply-viirs",
        action="append",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        default=[],
        help=(
            "Additional coarse NBAR input/output pair to receive the global shift "
            "estimated from --viirs-nbar. Repeat for each additional date or band."
        ),
    )
    parser.add_argument("--window-size", type=int, default=1024, help="Central matching-window size in Landsat pixels; default: 1024.")
    parser.add_argument("--window", type=_parse_window, help="Optional matching window: x_offset,y_offset,width,height on the Landsat grid.")
    parser.add_argument("--max-shift-pixels", type=int, default=20, help="Maximum shift AROSICS should accept; default: 20.")
    parser.add_argument("--reference-nodata", type=float, default=0.0, help="Landsat nodata value; default: 0.")
    parser.add_argument("--target-nodata", type=float, default=-9999.0, help="VIIRS NBAR nodata value; default: -9999.")
    parser.add_argument("--valid-mask", type=Path, help="Optional Landsat-grid 0/1 mask used only while estimating the shift.")
    parser.add_argument("--report", type=Path, help="Optional JSON diagnostic-report path.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    landsat = _open_single_band(args.landsat)
    _open_single_band(args.viirs_nbar)
    if args.max_shift_pixels < 0:
        raise ValueError("--max-shift-pixels must be non-negative.")

    output_path = args.output or args.output_dir / f"{args.viirs_nbar.stem}_coregistered_arosics.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_viirs_path = _warp_viirs_to_landsat_grid(
        args.viirs_nbar,
        landsat,
        output_path.with_name(f"{args.viirs_nbar.stem}_on_landsat_grid.tif"),
        args.target_nodata,
    )
    matching_landsat_path = args.landsat
    matching_viirs_path = aligned_viirs_path
    if args.valid_mask is not None:
        matching_landsat_path = masked_copy(
            args.landsat, args.valid_mask,
            output_path.with_name(f"{args.landsat.stem}_matching_masked.tif"), args.reference_nodata,
        )
        matching_viirs_path = masked_copy(
            aligned_viirs_path, args.valid_mask,
            output_path.with_name(f"{args.viirs_nbar.stem}_matching_masked.tif"), args.target_nodata,
        )
    window_center, matching_window, window_metadata = _matching_window_center(
        landsat, args.window_size, args.window
    )
    coregistration = COREG(
        im_ref=str(matching_landsat_path),
        im_tgt=str(matching_viirs_path),
        path_out=str(output_path),
        fmt_out="GTIFF",
        r_b4match=1,
        s_b4match=1,
        wp=window_center,
        ws=matching_window,
        max_iter=100,
        max_shift=args.max_shift_pixels,
        align_grids=True,
        match_gsd=True,
        resamp_alg_calc="bilinear",
        resamp_alg_deshift="bilinear",
        nodata=(args.reference_nodata, args.target_nodata),
        q=True,
        ignore_errors=False,
    )
    status = coregistration.calculate_spatial_shifts()
    if status != "success" or not coregistration.success:
        raise RuntimeError("AROSICS could not derive a reliable global shift for this image pair.")
    deshift_results = coregistration.correct_shifts()
    applied_outputs = [str(output_path.absolute())]
    for input_name, output_name in args.apply_viirs:
        additional_output = _apply_detected_shift(
            Path(input_name), landsat, coregistration, Path(output_name), args.target_nodata
        )
        applied_outputs.append(str(additional_output.absolute()))
    report = {
        "method": "arosics_global_coregistration",
        "arosics_version": AROSICS_VERSION,
        "landsat": str(args.landsat.absolute()),
        "viirs_nbar": str(args.viirs_nbar.absolute()),
        "viirs_nbar_on_landsat_grid": str(aligned_viirs_path.absolute()),
        "valid_mask": str(args.valid_mask.absolute()) if args.valid_mask else None,
        "output": str(output_path.absolute()),
        "outputs": applied_outputs,
        "matching_window_center_map_coordinates": {"x": window_center[0], "y": window_center[1]},
        "matching_window_landsat_pixels": window_metadata,
        "matching_window_size_landsat_pixels": {"width": matching_window[0], "height": matching_window[1]},
        "maximum_shift_pixels": args.max_shift_pixels,
        "x_shift_pixels": _json_safe(coregistration.x_shift_px),
        "y_shift_pixels": _json_safe(coregistration.y_shift_px),
        "map_x_shift": _json_safe(coregistration.x_shift_map),
        "map_y_shift": _json_safe(coregistration.y_shift_map),
        "shift_reliability_percent": _json_safe(coregistration.shift_reliability),
        "coreg_info": _json_safe(coregistration.coreg_info),
        "deshift_results": {
            key: _json_safe(deshift_results[key])
            for key in ("is shifted", "is resampled", "updated map info", "updated geotransform", "band")
            if key in deshift_results
        },
    }
    report_path = args.report or output_path.with_suffix(".arosics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
