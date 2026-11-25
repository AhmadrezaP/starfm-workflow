"""Collect completed per-band validation summaries for one Slurm experiment."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    # Literal prefix matching, not a user-supplied glob.
    root = args.output_dir / "OUTPUTS"
    rows = []
    for folder in sorted((root / "TESTS").iterdir()):
        if folder.name.startswith(args.experiment_id + "_"):
            summary = folder / "test_summary.json"
            if summary.is_file():
                rows.append(json.loads(summary.read_text()))
    if not rows:
        raise RuntimeError("No completed test summaries found for this experiment.")
    destination = root / "EXPERIMENTS" / args.experiment_id
    destination.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (destination / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Collected {len(rows)} completed summaries in {destination}")


if __name__ == "__main__":
    main()
