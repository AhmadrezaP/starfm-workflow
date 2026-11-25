"""Add MAE and coefficient-of-determination R² to saved held-out runs.

No downloads, registration, STARFM, reprojection, or mask changes. Preview is
the default; --apply backs up and updates per-run JSON/CSV reports. Refresh
campaign tables afterwards with slurm/date_plan.py --collect PLAN.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from osgeo import gdal, osr


def resolve_file(value, root, near):
    """Support original absolute paths and relocated COREGISTRATION_FILES trees."""
    if not value:
        raise FileNotFoundError("Missing raster path in saved report")
    old = Path(value)
    candidates = [near / old.name]
    if "OUTPUTS" in old.parts:
        candidates.append(root.joinpath(*old.parts[old.parts.index("OUTPUTS"):]))
    candidates.append(old)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Required saved raster not found: {value}")


def calculate(prediction, actual, mask, block_rows=256):
    datasets = [gdal.Open(str(p)) for p in (prediction, actual, mask)]
    if any(d is None or d.RasterCount != 1 for d in datasets):
        raise ValueError("Expected three readable single-band rasters")
    reference = datasets[0]
    for d in datasets[1:]:
        if (d.RasterXSize, d.RasterYSize) != (reference.RasterXSize, reference.RasterYSize):
            raise ValueError("Raster dimensions differ")
        if not np.allclose(d.GetGeoTransform(), reference.GetGeoTransform(), rtol=0, atol=1e-8):
            raise ValueError("Raster transforms differ")
        a, b = osr.SpatialReference(), osr.SpatialReference()
        a.ImportFromWkt(reference.GetProjection())
        b.ImportFromWkt(d.GetProjection())
        if not a.IsSame(b):
            raise ValueError("Raster coordinate systems differ")
    n = 0
    mean = m2 = sse = absolute = bias = 0.0
    for y in range(0, reference.RasterYSize, block_rows):
        height = min(block_rows, reference.RasterYSize - y)
        arrays = [d.GetRasterBand(1).ReadAsArray(0, y, d.RasterXSize, height) for d in datasets]
        if not np.all(np.isin(arrays[2], [0, 1])):
            raise ValueError("Comparison mask must contain only 0 and 1")
        valid = arrays[2] == 1
        if not valid.any():
            continue
        p, a = [x[valid].astype(np.float64) for x in arrays[:2]]
        for values, d in zip((p, a), datasets[:2]):
            nd = d.GetRasterBand(1).GetNoDataValue()
            if not np.isfinite(values).all() or (nd is not None and np.any(np.isclose(values, nd))):
                raise ValueError("Saved comparison mask includes invalid raster values")
        # These pipeline products contain Landsat Collection 2 integer DN,
        # regardless of their GDAL scale/offset metadata (normally 1 and 0).
        p = p * 0.0000275 - 0.2
        a = a * 0.0000275 - 0.2
        if np.any((a < 0) | (a > 1) | (p < 0) | (p > 1)):
            raise ValueError("Masked reflectances outside [0,1]; verify saved DN products")
        count = a.size
        batch_mean = float(a.mean())
        delta = batch_mean - mean
        m2 += float(np.sum((a - batch_mean) ** 2)) + delta ** 2 * n * count / (n + count)
        mean += delta * count / (n + count)
        n += count
        error = p - a
        sse += float(np.sum(error ** 2))
        absolute += float(np.sum(np.abs(error)))
        bias += float(np.sum(error))
    if not n:
        raise ValueError("Comparison mask has no valid pixels")
    return dict(mae_surface_reflectance=absolute / n,
                r2_surface_reflectance=1 - sse / m2 if n > 1 and m2 > 0 else None,
                rmse_surface_reflectance=float(np.sqrt(sse / n)),
                mean_difference_predicted_minus_actual=bias / n,
                valid_comparison_pixels=n)


def write_report(path, data):
    """Keep the first pre-backfill copy; replace each report atomically."""
    if path.exists():
        backup = path.with_name(path.name + ".before_metrics_backfill")
        if not backup.exists():
            shutil.copy2(path, backup)
    if path.suffix == ".csv":
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(data))
        writer.writeheader()
        writer.writerow(data)
        content = stream.getvalue()
    else:
        content = json.dumps(data, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".backfill.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def process(test, root, apply=False):
    summary_path = test / "test_summary.json"
    summary = json.loads(summary_path.read_text())
    reports = list(test.glob("VALIDATION/**/heldout_validation_metrics.json"))
    if len(reports) != 1:
        raise ValueError(f"Expected exactly one validation report, found {len(reports)}")
    report_path = reports[0]
    report = json.loads(report_path.read_text())
    result = calculate(
        resolve_file(summary["final_output_geotiff"], root, test),
        resolve_file(report.get("actual_landsat_on_prediction_grid"), root, report_path.parent),
        resolve_file(report.get("comparison_mask"), root, report_path.parent),
    )
    # Never add metrics on a different pixel population or silently replace
    # contradictory existing scores. Float tolerance covers summation order.
    for source in (report, summary):
        for key in ("valid_comparison_pixels", "rmse_surface_reflectance", "mean_difference_predicted_minus_actual"):
            if key not in source or source[key] is None:
                raise ValueError(f"Missing verification metric: {key}")
            matches = (result[key] == source[key] if key == "valid_comparison_pixels"
                       else np.isclose(result[key], source[key], rtol=1e-7, atol=1e-10))
            if not matches:
                raise ValueError(f"Original {key} differs: saved={source[key]}, recalculated={result[key]}")
    additions = {k: result[k] for k in ("mae_surface_reflectance", "r2_surface_reflectance")}
    manifest_path = test / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None:
        manifest["outputs"]["heldout_landsat_validation"].update(additions)
    if apply:
        for path, data in ((report_path, report), (summary_path, summary)):
            data.update(additions)
            write_report(path, data)
            write_report(path.with_suffix(".csv"), data)
        if manifest is not None:
            write_report(manifest_path, manifest)
    return additions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("COREGISTRATION_FILES"))
    parser.add_argument("--test-pattern", default="*", help="Test-folder glob, e.g. '*_retry1'")
    parser.add_argument("--apply", action="store_true", help="Update reports; otherwise calculate and verify only")
    args = parser.parse_args()
    gdal.UseExceptions()
    root = args.output_dir.resolve()
    tests = sorted(p for p in (root / "OUTPUTS/TESTS").glob(args.test_pattern)
                   if p.is_dir() and (p / "test_summary.json").is_file())
    if not tests:
        parser.error("No completed test folders matched")
    failures = 0
    for i, test in enumerate(tests, 1):
        try:
            result = process(test, root, args.apply)
            print(f"[{i}/{len(tests)}] {'UPDATED' if args.apply else 'VERIFIED'} {test.name}: {result}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"[{i}/{len(tests)}] FAILED {test.name}: {exc}", file=sys.stderr, flush=True)
    print(f"Successful: {len(tests)-failures}; failed: {failures}; mode: {'apply' if args.apply else 'preview'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
