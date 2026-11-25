"""Score simple temporal baselines on each completed STARFM run's own validation pixels.

For one run, the two Landsat pair images are aligned to the prediction grid and
compared with the held-out target using:

* persistence from the earlier pair, the later pair, and the nearer pair
  (ties use the earlier pair, which is available first);
* linear interpolation in time between the two pairs.

All methods are scored on one common set: the run's saved comparison mask
(target clear and prediction valid) further restricted to pixels that are clear
and in range in both pairs. STARFM is also rescored on that set, and on its
full saved mask to confirm the stored metrics are reproduced.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import tempfile

import numpy as np

SCALE, OFFSET = 0.0000275, -0.2


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    """RMSE, MAE, bias and R² (1-SSE/SST) for already-masked reflectances."""
    errors = predicted - actual
    sse = float(np.sum(errors ** 2))
    sst = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "n": int(actual.size),
        "rmse": float(np.sqrt(sse / actual.size)),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "r2": float(1.0 - sse / sst) if actual.size >= 2 and sst > 0.0 else None,
    }


def score_methods(
    actual: np.ndarray,
    starfm: np.ndarray,
    saved_mask: np.ndarray,
    pairs: list[tuple[date, np.ndarray, np.ndarray]],
    target: date,
) -> dict:
    """Score STARFM and the baselines.

    ``pairs`` holds (date, reflectance, clear QA mask) for the two pair images.
    """
    (d1, early, clear1), (d2, late, clear2) = sorted(pairs, key=lambda item: item[0])
    if not d1 < target < d2:
        raise ValueError(f"Prediction date {target} is not bracketed by {d1} and {d2}.")

    def usable(values: np.ndarray) -> np.ndarray:
        return np.isfinite(values) & (values >= 0.0) & (values <= 1.0)

    common = saved_mask & clear1 & clear2 & usable(early) & usable(late)
    if not np.any(common):
        raise RuntimeError("No pixels are clear in the target and both pairs.")
    weight = (target - d1).days / (d2 - d1).days
    nearest = early if (target - d1) <= (d2 - target) else late
    candidates = {
        "starfm": starfm,
        "persistence_earlier": early,
        "persistence_later": late,
        "persistence_nearest": nearest,
        "linear_interpolation": (1.0 - weight) * early + weight * late,
    }
    reference = actual[common]
    return {
        "days_from_earlier_pair": (target - d1).days,
        "days_to_later_pair": (d2 - target).days,
        "interpolation_weight_later": weight,
        "saved_mask_pixels": int(np.count_nonzero(saved_mask)),
        "common_pixels": int(np.count_nonzero(common)),
        "common_fraction_of_saved_mask": float(np.count_nonzero(common) / max(1, np.count_nonzero(saved_mask))),
        "starfm_on_saved_mask": metrics(actual[saved_mask], starfm[saved_mask]),
        "common_mask": {name: metrics(reference, values[common]) for name, values in candidates.items()},
    }


def evaluate_run(manifest_path: Path, work_directory: Path) -> dict:
    import run_nearest_pair_pipeline as pipeline
    from build_coregistration_mask import landsat_clear_mask

    manifest = json.loads(manifest_path.read_text())
    band = manifest["requested_landsat_band"]
    validation = manifest["outputs"]["heldout_landsat_validation"]
    prediction = Path(manifest["outputs"]["final_output_geotiff"])

    def reflectance(path: Path) -> np.ndarray:
        raw, _ = pipeline._read_float_raster(path)
        return raw * SCALE + OFFSET

    starfm = reflectance(prediction)
    actual = reflectance(Path(validation["actual_landsat_on_prediction_grid"]))
    saved_mask = pipeline._open_single_band(Path(validation["comparison_mask"])).ReadAsArray() == 1
    pairs = []
    for pair_date, files in manifest["inputs"]["landsat"].items():
        aligned = {}
        for asset in (band, "QA_PIXEL"):
            output = work_directory / f"{pair_date}_{asset}.tif"
            aligned[asset] = pipeline._align_landsat_to_reference(Path(files[asset]), prediction, output)
        qa = pipeline._open_single_band(aligned["QA_PIXEL"]).ReadAsArray()
        pairs.append((date.fromisoformat(pair_date), reflectance(aligned[band]), landsat_clear_mask(qa)))
    result = score_methods(actual, starfm, saved_mask, pairs, date.fromisoformat(manifest["prediction_date"]))
    stored = validation["rmse_surface_reflectance"]
    rescored = result["starfm_on_saved_mask"]["rmse"]
    if not np.isclose(stored, rescored, rtol=1e-6, atol=1e-9):
        raise RuntimeError(f"Rescored STARFM RMSE {rescored} does not reproduce stored {stored}.")
    return dict(test_id=manifest["test_id"], band=band, prediction_date=manifest["prediction_date"],
                wrs2_path=manifest["wrs2_path"], wrs2_row=manifest["wrs2_row"],
                coarse_sensor=manifest["coarse_sensor"], **result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-list", type=Path, required=True, help="Text file with one run_manifest.json path per line")
    parser.add_argument("--index", type=int, required=True, help="Zero-based line in --manifest-list")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest_list.read_text().split()[args.index])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="baselines-") as directory:
        result = evaluate_run(manifest, Path(directory))
    output = args.output_dir / f"{result['test_id']}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(result["common_mask"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
