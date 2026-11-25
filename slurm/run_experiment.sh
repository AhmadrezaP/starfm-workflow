#!/bin/bash
# Shared implementation; submit one of the four .sbatch files.
set -euo pipefail
MODE="${1:?experiment required}"
PREDICTION_DATE="${PREDICTION_DATE:-2023-09-18}"
WRS_PATH="${WRS_PATH:-29}"
WRS_ROW="${WRS_ROW:-32}"
PAIR_BEFORE="${PAIR_BEFORE:-2023-09-02}"
PAIR_AFTER="${PAIR_AFTER:-2023-09-26}"
SECOND_BEFORE="${SECOND_BEFORE:-2023-08-25}"
SECOND_AFTER="${SECOND_AFTER:-2023-10-04}"
ONE_PAIR_DATE="${ONE_PAIR_DATE:-$PAIR_AFTER}"
COARSE_SENSOR="${COARSE_SENSOR:-VIIRS}"
COREG_BAND="${COREG_BAND:-}"
WINDOW_SIZE="${WINDOW_SIZE:-1024}"
CLOUD_PERCENT="${CLOUD_PERCENT:-5}"
SEARCH_DISTANCE="${SEARCH_DISTANCE:-750}"
UNCERTAINTY="${UNCERTAINTY:-50}"
REMOVE_UNNECESSARY="${REMOVE_UNNECESSARY:-0}"
RESUME="${RESUME:-0}"
STARFM_EXECUTABLE="${STARFM_EXECUTABLE:-StarFM_original/source/StarFM.exe}"
OUTPUT_DIR="${OUTPUT_DIR:-COREGISTRATION_FILES}"
EXPERIMENT_ID="${EXPERIMENT_ID:-${MODE}_p${WRS_PATH}_r${WRS_ROW}_${PREDICTION_DATE//-/}_j${SLURM_JOB_ID:-local}}"
if [[ "$MODE" == sensors ]]; then
  BAND="${BAND:-B3}"
else
  BAND="${BAND:-B1}"
fi
if [[ "$BAND" == all ]]; then
  band_args=(--all-bands)
else
  band_args=(--band "$BAND")
fi
case "$REMOVE_UNNECESSARY" in
  0) runner_cleanup=(--keep-intermediate-outputs); pipeline_cleanup=(--keep-intermediate-outputs) ;;
  1) runner_cleanup=(--remove-unnecessary-files); pipeline_cleanup=(--discard-intermediate-outputs-after-validation) ;;
  *) echo "REMOVE_UNNECESSARY must be 0 or 1" >&2; exit 2 ;;
esac
case "$RESUME" in
  0) ;;
  1) runner_cleanup+=(--resume) ;;
  *) echo "RESUME must be 0 or 1" >&2; exit 2 ;;
esac

# Restrict OpenMP to the CPUs allocated by Slurm and allow the original C
# implementation's large per-thread candidate array.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_STACKSIZE="${OMP_STACKSIZE:-64M}"
export PYTHONUNBUFFERED=1
if [[ "${DRY_RUN:-0}" != 1 ]]; then
python -c 'from osgeo import gdal; import yaml; assert all(gdal.GetDriverByName(n) for n in ("HDF4", "HDF5")), "Missing GDAL HDF4/HDF5 drivers"'
[[ -x "$STARFM_EXECUTABLE" ]] || { echo "Build STARFM first: make -B -C StarFM_original/source" >&2; exit 2; }
if [[ "$(uname -s)" == Linux ]] && ! file "$STARFM_EXECUTABLE" | grep -q ELF; then
  echo "STARFM is not a Linux ELF binary. Build it on Swan first." >&2
  exit 2
fi
fi
common=(--prediction-date "$PREDICTION_DATE" --path "$WRS_PATH" --row "$WRS_ROW"
  "${band_args[@]}" --max-cloud-percent "$CLOUD_PERCENT"
  --window-size "$WINDOW_SIZE"
  --starfm-executable "$STARFM_EXECUTABLE"
  --output-dir "$OUTPUT_DIR")
if [[ -n "$COREG_BAND" ]]; then common+=(--coregistration-band "$COREG_BAND"); fi
run_command() {
  printf 'Running:'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "$@"; fi
}
pipeline() {
  local scenario="$1"; shift
  run_command python -u run_nearest_pair_pipeline.py "${common[@]}" \
    --coarse-sensor "$COARSE_SENSOR" --starfm-max-search-distance "$SEARCH_DISTANCE" \
    --starfm-uncertainty "$UNCERTAINTY" \
    --validate-held-out-landsat --reuse-existing-nbar "${pipeline_cleanup[@]}" \
    --test-id "${EXPERIMENT_ID}_${scenario}" "$@"
}
echo "Experiment: $EXPERIMENT_ID; band: $BAND; started: $(date -Is)"
case "$MODE" in
  parameters)
    run_command python -u run_search_distance_sweep.py "${common[@]}" \
      --coarse-sensor "$COARSE_SENSOR" --coregistration-method moving-window \
      --pair-dates "$PAIR_BEFORE" "$PAIR_AFTER" --sweep-id "$EXPERIMENT_ID" \
      --starfm-uncertainty "$UNCERTAINTY" \
      --start-distance 550 --stop-distance 950 --step-distance 100 \
      "${runner_cleanup[@]}"
    run_command python -u run_uncertainty_sweep.py "${common[@]}" \
      --coarse-sensor "$COARSE_SENSOR" --coregistration-method moving-window \
      --pair-dates "$PAIR_BEFORE" "$PAIR_AFTER" --sweep-id "$EXPERIMENT_ID" \
      --starfm-max-search-distance "$SEARCH_DISTANCE" \
      --uncertainties 20 35 50 65 80 "${runner_cleanup[@]}"
    ;;
  sensors)
    run_command python -u run_modis_viirs_comparison.py "${common[@]}" \
      --coregistration-method moving-window --starfm-max-search-distance "$SEARCH_DISTANCE" \
      --starfm-uncertainty "$UNCERTAINTY" \
      --pair-dates "$PAIR_BEFORE" "$PAIR_AFTER" --comparison-id "$EXPERIMENT_ID" \
      "${runner_cleanup[@]}"
    ;;
  coregistration)
    COREGISTRATION_METHODS="${COREGISTRATION_METHODS:-moving-window arosics none}"
    for method in $COREGISTRATION_METHODS; do
      case "$method" in
        moving-window|arosics|none) ;;
        *) echo "Unsupported coregistration method in COREGISTRATION_METHODS: $method" >&2; exit 2 ;;
      esac
      pipeline "$method" --coregistration-method "$method" --pair-dates "$PAIR_BEFORE" "$PAIR_AFTER"
    done
    ;;
  pairs)
    pipeline one --coregistration-method moving-window --pair-dates "$ONE_PAIR_DATE"
    pipeline bracket --coregistration-method moving-window --pair-dates "$PAIR_BEFORE" "$PAIR_AFTER"
    pipeline before --coregistration-method moving-window --pair-dates "$SECOND_BEFORE" "$PAIR_BEFORE"
    pipeline after --coregistration-method moving-window --pair-dates "$PAIR_AFTER" "$SECOND_AFTER"
    ;;
  *) echo "Unknown experiment: $MODE" >&2; exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  python slurm/collect_results.py --output-dir "$OUTPUT_DIR" --experiment-id "$EXPERIMENT_ID"
  echo "Completed: $EXPERIMENT_ID; finished: $(date -Is)"
else
  echo "Dry run complete; no pipeline commands executed."
fi
