"""Run a reproducible STARFM maximum-search-distance parameter sweep.

Each distance is executed as a separately named pipeline test. After every
successful held-out validation, this script collects the compact test summary
into one CSV/JSON table that is ready for spreadsheet analysis.
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


def _distances(start: int, stop: int, step: int) -> list[int]:
    if start <= 0 or stop <= 0 or step <= 0:
        raise ValueError("Search distances and step must be positive.")
    if start > stop:
        raise ValueError("--start-distance must not exceed --stop-distance.")
    if (stop - start) % step != 0:
        raise ValueError("The requested range must be exactly divisible by --step-distance.")
    return list(range(start, stop + 1, step))


def _test_id_for_distance(sweep_id: str, band: str, distance: int) -> str:
    return _test_id(f"{sweep_id}_{band.lower()}_d{distance:04d}")


def _selected_bands(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all_bands:
        return tuple(
            band for band, info in BAND_INFO.items()
            if args.coarse_sensor == "VIIRS" or info.modis_band is not None
        )
    return (args.band,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run STARFM at a series of maximum search distances, retain each named test's "
            "final/validation products, and write one aggregate comparison table."
        )
    )
    parser.add_argument("--prediction-date", type=_parse_date, required=True)
    parser.add_argument("--path", type=int, required=True, help="WRS-2 path.")
    parser.add_argument("--row", type=int, required=True, help="WRS-2 row.")
    band_selection = parser.add_mutually_exclusive_group(required=True)
    band_selection.add_argument("--band", type=_band, help="One final Landsat-like output band.")
    band_selection.add_argument(
        "--all-bands",
        action="store_true",
        help="Run the same sweep for every reflective band supported by the selected coarse sensor.",
    )
    parser.add_argument("--sweep-id", type=_test_id, required=True, help="Unique label for this whole sweep.")
    parser.add_argument("--coarse-sensor", choices=("VIIRS", "MODIS"), type=str.upper, default="MODIS")
    parser.add_argument("--coregistration-method", choices=("moving-window", "arosics", "none"), default="arosics")
    parser.add_argument("--coregistration-band", type=_band, help="AROSICS band (default B5); ignored for moving-window, which uses each prediction band.")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--pair-count", choices=(1, 2), type=int, default=1)
    parser.add_argument("--pair-dates", type=_parse_date, nargs="+", metavar="YYYY-MM-DD")
    parser.add_argument("--max-cloud-percent", type=float, default=10.0)
    parser.add_argument("--starfm-uncertainty", type=int, default=50)
    parser.add_argument("--start-distance", type=int, default=550)
    parser.add_argument("--stop-distance", type=int, default=950)
    parser.add_argument("--step-distance", type=int, default=100)
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
    intermediate_files = parser.add_mutually_exclusive_group()
    intermediate_files.add_argument(
        "--keep-intermediate-outputs",
        action="store_true",
        help="Keep bulky co-registration and STARFM intermediate files for every distance.",
    )
    intermediate_files.add_argument(
        "--remove-unnecessary-files",
        action="store_true",
        help=(
            "Remove generated NBAR, co-registration, and temporary STARFM input/log files "
            "after each successful test. This is already the default."
        ),
    )
    return parser


def _pipeline_command(
    args: argparse.Namespace,
    band: str,
    distance: int,
    test_id: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("run_nearest_pair_pipeline.py")),
        "--prediction-date", args.prediction_date.isoformat(),
        "--path", str(args.path),
        "--row", str(args.row),
        "--band", band,
        "--coarse-sensor", args.coarse_sensor,
        "--coregistration-method", args.coregistration_method,
        "--coregistration-band", (args.coregistration_band or "B5") if args.coregistration_method == "arosics" else band,
        "--window-size", str(args.window_size),
        "--max-cloud-percent", str(args.max_cloud_percent),
        "--starfm-max-search-distance", str(distance),
        "--starfm-uncertainty", str(args.starfm_uncertainty),
        "--output-dir", str(args.output_dir),
        "--starfm-executable", str(args.starfm_executable),
        "--validate-held-out-landsat",
        "--test-id", test_id,
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


def _write_aggregate_results(
    output_directory: Path,
    rows: list[dict[str, Any]],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "search_distance_sweep_results.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["starfm_max_search_distance_m", "status", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_directory / "search_distance_sweep_results.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    distances = _distances(args.start_distance, args.stop_distance, args.step_distance)
    bands = _selected_bands(args)
    sweep_directory = args.output_dir / "OUTPUTS" / "SWEEPS" / args.sweep_id
    if sweep_directory.exists() and not args.resume:
        raise RuntimeError(f"Sweep output directory already exists: {sweep_directory}. Choose a new --sweep-id.")

    test_root = args.output_dir / "OUTPUTS" / "TESTS"
    test_plan = [
        (band, distance, _test_id_for_distance(args.sweep_id, band, distance))
        for band in bands
        for distance in distances
    ]
    test_ids = [test_id for _, _, test_id in test_plan]
    existing = [test_id for test_id in test_ids if (test_root / test_id).exists()]
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
            raise RuntimeError(
            "Refusing to overwrite existing test results: " + ", ".join(existing)
            )

    rows: list[dict[str, Any]] = []
    for band, distance, test_id in test_plan:
        summary_path = test_root / test_id / "test_summary.json"
        if args.resume and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["status"] = "completed"
            rows.append(summary)
            _write_aggregate_results(sweep_directory, rows)
            print(f"Reusing completed test: {test_id}", flush=True)
            continue
        command = _pipeline_command(args, band, distance, test_id)
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
        if not summary_path.is_file():
            raise RuntimeError(f"Pipeline completed but did not create: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "completed"
        rows.append(summary)
        _write_aggregate_results(sweep_directory, rows)

    print(json.dumps({
        "sweep_id": args.sweep_id,
        "bands": bands,
        "distances_m": distances,
        "aggregate_csv": str((sweep_directory / "search_distance_sweep_results.csv").absolute()),
        "aggregate_json": str((sweep_directory / "search_distance_sweep_results.json").absolute()),
    }, indent=2))


if __name__ == "__main__":
    main()
