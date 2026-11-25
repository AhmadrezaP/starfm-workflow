"""Run a matched held-out STARFM comparison of MODIS Terra and VIIRS.

Both coarse sensors are run with exactly the same Landsat pair dates, target
date, STARFM settings, and coregistration settings.  The all-band mode uses
only Landsat bands supported by *both* sensors so the resulting accuracy table
is a like-for-like comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from run_nearest_pair_pipeline import BAND_INFO, _band, _parse_date, _test_id


SENSORS = ("MODIS", "VIIRS")
COMPARABLE_BANDS = tuple(band for band, info in BAND_INFO.items() if info.modis_band is not None)
COMPARISON_METRICS = (
    "rmse_surface_reflectance",
    "mae_surface_reflectance",
    "r2_surface_reflectance",
    "mean_difference_predicted_minus_actual",
    "relative_error_percent",
    "valid_comparison_percent",
    "coregistration_quality_percent",
)


def _comparison_id(value: str) -> str:
    """Validate an ID while reserving room for sensor and band suffixes."""
    normalized = _test_id(value)
    if len(normalized) > 65:
        raise argparse.ArgumentTypeError(
            "--comparison-id must be at most 65 characters so sensor/band test IDs remain portable."
        )
    return normalized


def _test_id_for_sensor_band(comparison_id: str, sensor: str, band: str) -> str:
    return _test_id(f"{comparison_id}_{sensor.lower()}_{band.lower()}")


def _selected_bands(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all_bands:
        return COMPARABLE_BANDS
    if args.band not in COMPARABLE_BANDS:
        raise ValueError(
            f"{args.band} cannot be compared because MODIS has no matching daily reflectance band. "
            f"Choose one of: {', '.join(COMPARABLE_BANDS)}."
        )
    return (args.band,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched MODIS and VIIRS STARFM validations, then write long-form and "
            "per-band paired comparison tables."
        )
    )
    parser.add_argument("--prediction-date", type=_parse_date, required=True)
    parser.add_argument("--path", type=int, required=True, help="WRS-2 path.")
    parser.add_argument("--row", type=int, required=True, help="WRS-2 row.")
    band_selection = parser.add_mutually_exclusive_group(required=True)
    band_selection.add_argument("--band", type=_band, help="One Landsat output band to compare.")
    band_selection.add_argument(
        "--all-bands",
        action="store_true",
        help="Compare every band available from both MODIS and VIIRS: B2, B3, B4, B5, and B7.",
    )
    parser.add_argument("--comparison-id", type=_comparison_id, required=True, help="Unique label for this comparison.")
    parser.add_argument("--coregistration-method", choices=("moving-window", "arosics", "none"), default="moving-window")
    parser.add_argument(
        "--coregistration-band",
        type=_band,
        default=None,
        help="AROSICS band (default B5); ignored for moving-window, which uses each prediction band.",
    )
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--pair-count", choices=(1, 2), type=int, default=2)
    parser.add_argument("--pair-dates", type=_parse_date, nargs="+", metavar="YYYY-MM-DD")
    parser.add_argument("--max-cloud-percent", type=float, default=10.0)
    parser.add_argument("--starfm-max-search-distance", type=int, default=750)
    parser.add_argument("--starfm-uncertainty", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("COREGISTRATION_FILES"))
    parser.add_argument("--starfm-executable", type=Path, default=Path("StarFM_original/source/StarFM.exe"))
    intermediate_files = parser.add_mutually_exclusive_group()
    intermediate_files.add_argument(
        "--keep-intermediate-outputs",
        action="store_true",
        help="Keep bulky co-registration and STARFM intermediate files for each run.",
    )
    intermediate_files.add_argument(
        "--remove-unnecessary-files",
        action="store_true",
        help=(
            "Remove generated NBAR, co-registration, and temporary STARFM input/log files "
            "after each successful run. This is already the default."
        ),
    )
    return parser


def _pipeline_command(args: argparse.Namespace, sensor: str, band: str, test_id: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("run_nearest_pair_pipeline.py")),
        "--prediction-date", args.prediction_date.isoformat(),
        "--path", str(args.path),
        "--row", str(args.row),
        "--band", band,
        "--coarse-sensor", sensor,
        "--coregistration-method", args.coregistration_method,
        "--coregistration-band", (args.coregistration_band or "B5") if args.coregistration_method == "arosics" else band,
        "--window-size", str(args.window_size),
        "--max-cloud-percent", str(args.max_cloud_percent),
        "--starfm-max-search-distance", str(args.starfm_max_search_distance),
        "--starfm-uncertainty", str(args.starfm_uncertainty),
        "--output-dir", str(args.output_dir),
        "--starfm-executable", str(args.starfm_executable),
        "--validate-held-out-landsat",
        "--test-id", test_id,
        # NBAR products are sensor/date/band-specific and may safely be reused
        # among the individual bands of this comparison.
        "--reuse-existing-nbar",
    ]
    if args.pair_dates is not None:
        command.extend(["--pair-dates", *(pair_date.isoformat() for pair_date in args.pair_dates)])
    else:
        command.extend(["--pair-count", str(args.pair_count)])
    if args.keep_intermediate_outputs:
        command.append("--keep-intermediate-outputs")
    else:
        command.append("--discard-intermediate-outputs-after-validation")
    return command


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paired_rows(rows: list[dict[str, Any]], bands: tuple[str, ...]) -> list[dict[str, Any]]:
    by_sensor_band = {(row["coarse_sensor"], row["predicted_landsat_band"]): row for row in rows}
    paired: list[dict[str, Any]] = []
    for band in bands:
        modis, viirs = by_sensor_band[("MODIS", band)], by_sensor_band[("VIIRS", band)]
        comparison: dict[str, Any] = {
            "predicted_landsat_band": band,
            "modis_test_id": modis["test_id"],
            "viirs_test_id": viirs["test_id"],
            "prediction_date": modis["prediction_date"],
            "pair_dates": modis["pair_dates"],
            "coregistration_method": modis["coregistration_method"],
        }
        for metric in COMPARISON_METRICS:
            modis_value, viirs_value = modis.get(metric), viirs.get(metric)
            comparison[f"modis_{metric}"] = modis_value
            comparison[f"viirs_{metric}"] = viirs_value
            comparison[f"viirs_minus_modis_{metric}"] = (
                float(viirs_value) - float(modis_value)
                if modis_value is not None and viirs_value is not None
                else None
            )
        paired.append(comparison)
    return paired


def _write_results(output_directory: Path, rows: list[dict[str, Any]], bands: tuple[str, ...]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    paired = _paired_rows(rows, bands) if len(rows) == 2 * len(bands) else []
    _write_table(output_directory / "modis_viirs_results.csv", rows)
    _write_table(output_directory / "modis_viirs_paired_comparison.csv", paired)
    (output_directory / "modis_viirs_results.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "modis_viirs_paired_comparison.json").write_text(
        json.dumps(paired, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    bands = _selected_bands(args)
    comparison_directory = args.output_dir / "OUTPUTS" / "SENSOR_COMPARISONS" / args.comparison_id
    if comparison_directory.exists():
        raise RuntimeError(
            f"Comparison output directory already exists: {comparison_directory}. Choose a new --comparison-id."
        )

    test_root = args.output_dir / "OUTPUTS" / "TESTS"
    plan = [
        (sensor, band, _test_id_for_sensor_band(args.comparison_id, sensor, band))
        for band in bands
        for sensor in SENSORS
    ]
    existing = [test_id for _, _, test_id in plan if (test_root / test_id).exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite existing test results: " + ", ".join(existing))

    rows: list[dict[str, Any]] = []
    for sensor, band, test_id in plan:
        command = _pipeline_command(args, sensor, band, test_id)
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
        summary_path = test_root / test_id / "test_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"Pipeline completed but did not create: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "completed"
        rows.append(summary)
        _write_results(comparison_directory, rows, bands)

    print(json.dumps({
        "comparison_id": args.comparison_id,
        "bands": bands,
        "long_form_csv": str((comparison_directory / "modis_viirs_results.csv").absolute()),
        "paired_csv": str((comparison_directory / "modis_viirs_paired_comparison.csv").absolute()),
    }, indent=2))


if __name__ == "__main__":
    main()
