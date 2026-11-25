"""Run a reproducible STARFM input-uncertainty parameter sweep."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from run_nearest_pair_pipeline import BAND_INFO, _band, _parse_date, _test_id


DEFAULT_UNCERTAINTIES = (20, 35, 50, 65, 80)


def _uncertainties(values: Iterable[int]) -> list[int]:
    result = list(values)
    if not result or any(value < 0 for value in result):
        raise ValueError("--uncertainties requires one or more non-negative integers.")
    if len(set(result)) != len(result):
        raise ValueError("--uncertainties must not contain duplicates.")
    return result


def _selected_bands(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all_bands:
        return tuple(
            band for band, info in BAND_INFO.items()
            if args.coarse_sensor == "VIIRS" or info.modis_band is not None
        )
    return (args.band,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run held-out STARFM validations for selected uncertainty values."
    )
    parser.add_argument("--prediction-date", type=_parse_date, required=True)
    parser.add_argument("--path", type=int, required=True)
    parser.add_argument("--row", type=int, required=True)
    bands = parser.add_mutually_exclusive_group(required=True)
    bands.add_argument("--band", type=_band)
    bands.add_argument("--all-bands", action="store_true")
    parser.add_argument("--sweep-id", type=_test_id, required=True)
    parser.add_argument("--coarse-sensor", choices=("VIIRS", "MODIS"), type=str.upper, default="VIIRS")
    parser.add_argument("--coregistration-method", choices=("moving-window", "arosics", "none"), default="moving-window")
    parser.add_argument("--coregistration-band", type=_band, help="AROSICS band; ignored by moving-window.")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--pair-count", choices=(1, 2), type=int, default=2)
    parser.add_argument("--pair-dates", type=_parse_date, nargs="+", metavar="YYYY-MM-DD")
    parser.add_argument("--max-cloud-percent", type=float, default=10.0)
    parser.add_argument("--starfm-max-search-distance", type=int, default=750)
    parser.add_argument("--uncertainties", type=int, nargs="+", default=list(DEFAULT_UNCERTAINTIES))
    parser.add_argument("--output-dir", type=Path, default=Path("COREGISTRATION_FILES"))
    parser.add_argument("--starfm-executable", type=Path, default=Path("StarFM_original/source/StarFM.exe"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an existing sweep: reuse tests that already have test_summary.json "
            "and run only missing tests. An incomplete test directory must first be moved "
            "aside or removed."
        ),
    )
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument("--keep-intermediate-outputs", action="store_true")
    cleanup.add_argument("--remove-unnecessary-files", action="store_true")
    return parser


def _pipeline_command(args: argparse.Namespace, band: str, uncertainty: int, test_id: str) -> list[str]:
    coregistration_band = (
        (args.coregistration_band or "B5")
        if args.coregistration_method == "arosics"
        else band
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("run_nearest_pair_pipeline.py")),
        "--prediction-date", args.prediction_date.isoformat(),
        "--path", str(args.path), "--row", str(args.row), "--band", band,
        "--coarse-sensor", args.coarse_sensor,
        "--coregistration-method", args.coregistration_method,
        "--coregistration-band", coregistration_band,
        "--window-size", str(args.window_size),
        "--max-cloud-percent", str(args.max_cloud_percent),
        "--starfm-max-search-distance", str(args.starfm_max_search_distance),
        "--starfm-uncertainty", str(uncertainty),
        "--output-dir", str(args.output_dir),
        "--starfm-executable", str(args.starfm_executable),
        "--validate-held-out-landsat", "--test-id", test_id, "--reuse-existing-nbar",
    ]
    if args.pair_dates is not None:
        command.extend(["--pair-dates", *(day.isoformat() for day in args.pair_dates)])
    else:
        command.extend(["--pair-count", str(args.pair_count)])
    command.append(
        "--keep-intermediate-outputs"
        if args.keep_intermediate_outputs
        else "--discard-intermediate-outputs-after-validation"
    )
    return command


def _write_results(directory: Path, rows: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["starfm_uncertainty_scaled", "status"]
    with (directory / "uncertainty_sweep_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (directory / "uncertainty_sweep_results.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    values = _uncertainties(args.uncertainties)
    bands = _selected_bands(args)
    sweep_directory = args.output_dir / "OUTPUTS" / "UNCERTAINTY_SWEEPS" / args.sweep_id
    if sweep_directory.exists() and not args.resume:
        raise RuntimeError(f"Sweep output directory already exists: {sweep_directory}")
    test_root = args.output_dir / "OUTPUTS" / "TESTS"
    plan = [
        (band, value, _test_id(f"{args.sweep_id}_{band.lower()}_u{value:04d}"))
        for band in bands for value in values
    ]
    existing = [test_id for _, _, test_id in plan if (test_root / test_id).exists()]
    incomplete = [
        test_id for test_id in existing
        if not (test_root / test_id / "test_summary.json").is_file()
    ]
    if args.resume and incomplete:
        raise RuntimeError(
            "Cannot resume while incomplete test directories exist. Move aside or remove: "
            + ", ".join(incomplete)
        )
    if existing:
        if not args.resume:
            raise RuntimeError("Refusing to overwrite existing test results: " + ", ".join(existing))
    rows: list[dict[str, Any]] = []
    for band, value, test_id in plan:
        summary_path = test_root / test_id / "test_summary.json"
        if args.resume and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["status"] = "completed"
            rows.append(summary)
            _write_results(sweep_directory, rows)
            print(f"Reusing completed test: {test_id}", flush=True)
            continue
        command = _pipeline_command(args, band, value, test_id)
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        if not summary_path.is_file():
            raise RuntimeError(f"Pipeline completed but did not create: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "completed"
        rows.append(summary)
        _write_results(sweep_directory, rows)
    print(json.dumps({
        "sweep_id": args.sweep_id,
        "bands": bands,
        "uncertainties_scaled": values,
        "aggregate_csv": str((sweep_directory / "uncertainty_sweep_results.csv").absolute()),
        "aggregate_json": str((sweep_directory / "uncertainty_sweep_results.json").absolute()),
    }, indent=2))


if __name__ == "__main__":
    main()
