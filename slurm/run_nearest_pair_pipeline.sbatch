#!/bin/bash
#SBATCH --job-name=run_starfm_pipeline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.edu
#SBATCH --mem=32000
#SBATCH --partition=batch
#SBATCH --error=logs/thesis_%j.err
#SBATCH --output=logs/thesis_%j.out

set -eo pipefail

# DRY_RUN=1 makes this script print the pipeline command without loading the
# Swan environment, downloading data, or running STARFM.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  module load anaconda
  conda activate thesis_paper_env
fi

set -u

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-COREGISTRATION_FILES}"
PREDICTION_DATE="${PREDICTION_DATE:-2023-09-18}"
WRS_PATH="${WRS_PATH:-29}"
WRS_ROW="${WRS_ROW:-32}"
BAND="${BAND:-B1}"
COARSE_SENSOR="${COARSE_SENSOR:-VIIRS}"
PAIR_DATE_1="${PAIR_DATE_1:-2023-09-02}"
PAIR_DATE_2="${PAIR_DATE_2:-2023-09-26}"
MAXIMUM_DAYS="${MAXIMUM_DAYS:-32}"
CLOUD_PERCENT="${CLOUD_PERCENT:-5}"
VALIDATE_HELD_OUT="${VALIDATE_HELD_OUT:-1}"
COREGISTRATION_METHOD="${COREGISTRATION_METHOD:-moving-window}"
COREGISTRATION_BAND="${COREGISTRATION_BAND:-}"
WINDOW_SIZE="${WINDOW_SIZE:-1024}"
MINIMUM_WINDOW_VALID_FRACTION="${MINIMUM_WINDOW_VALID_FRACTION:-0.30}"
SEARCH_DISTANCE="${SEARCH_DISTANCE:-950}"
UNCERTAINTY="${UNCERTAINTY:-50}"
USE_QUALITY_MASK="${USE_QUALITY_MASK:-0}"
STARFM_USE_MASK_FILES="${STARFM_USE_MASK_FILES:-0}"
REUSE_EXISTING_NBAR="${REUSE_EXISTING_NBAR:-0}"
INTERMEDIATE_OUTPUTS="${INTERMEDIATE_OUTPUTS:-default}"
STARFM_EXECUTABLE="${STARFM_EXECUTABLE:-StarFM_original/source/StarFM.exe}"
TEST_ID="${TEST_ID:-pipeline_p${WRS_PATH}_r${WRS_ROW}_${PREDICTION_DATE//-/}_j${SLURM_JOB_ID:-local}}"

case "$COARSE_SENSOR" in
  VIIRS|MODIS) ;;
  *) echo "COARSE_SENSOR must be VIIRS or MODIS" >&2; exit 2 ;;
esac
case "$COREGISTRATION_METHOD" in
  moving-window|arosics|none) ;;
  *) echo "COREGISTRATION_METHOD must be moving-window, arosics, or none" >&2; exit 2 ;;
esac
for setting in VALIDATE_HELD_OUT USE_QUALITY_MASK STARFM_USE_MASK_FILES REUSE_EXISTING_NBAR; do
  value="${!setting}"
  [[ "$value" == 0 || "$value" == 1 ]] || {
    echo "$setting must be 0 or 1" >&2
    exit 2
  }
done
case "$INTERMEDIATE_OUTPUTS" in
  default|keep|discard) ;;
  *) echo "INTERMEDIATE_OUTPUTS must be default, keep, or discard" >&2; exit 2 ;;
esac
[[ -n "$PAIR_DATE_1" ]] || { echo "PAIR_DATE_1 must be set" >&2; exit 2; }

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  cd "$PROJECT_DIR"
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
  export OMP_STACKSIZE="${OMP_STACKSIZE:-64M}"
  export PYTHONUNBUFFERED=1
  python -c 'from osgeo import gdal; import yaml; assert all(gdal.GetDriverByName(n) for n in ("HDF4", "HDF5")), "Missing GDAL HDF4/HDF5 drivers"'
  [[ -x "$STARFM_EXECUTABLE" ]] || {
    echo "STARFM executable not found. Build it with: make -B -C StarFM_original/source" >&2
    exit 2
  }
  if [[ "$(uname -s)" == Linux ]] && ! file "$STARFM_EXECUTABLE" | grep -q ELF; then
    echo "STARFM_EXECUTABLE is not a Linux ELF binary. Build it on Swan first." >&2
    exit 2
  fi
fi

if [[ "$BAND" == all ]]; then
  band_args=(--all-bands)
else
  band_args=(--band "$BAND")
fi

pair_args=(--pair-dates "$PAIR_DATE_1")
if [[ -n "$PAIR_DATE_2" ]]; then
  pair_args+=("$PAIR_DATE_2")
fi

command=(
  python -u run_nearest_pair_pipeline.py
  --prediction-date "$PREDICTION_DATE"
  --path "$WRS_PATH"
  --row "$WRS_ROW"
  "${band_args[@]}"
  --coarse-sensor "$COARSE_SENSOR"
  "${pair_args[@]}"
  --maximum-days "$MAXIMUM_DAYS"
  --max-cloud-percent "$CLOUD_PERCENT"
  --coregistration-method "$COREGISTRATION_METHOD"
  --window-size "$WINDOW_SIZE"
  --minimum-window-valid-fraction "$MINIMUM_WINDOW_VALID_FRACTION"
  --starfm-max-search-distance "$SEARCH_DISTANCE"
  --starfm-uncertainty "$UNCERTAINTY"
  --starfm-executable "$STARFM_EXECUTABLE"
  --output-dir "$OUTPUT_DIR"
  --test-id "$TEST_ID"
)

[[ "$VALIDATE_HELD_OUT" == 1 ]] && command+=(--validate-held-out-landsat)
[[ -n "$COREGISTRATION_BAND" ]] && command+=(--coregistration-band "$COREGISTRATION_BAND")
[[ "$USE_QUALITY_MASK" == 1 ]] && command+=(--use-quality-mask)
[[ "$STARFM_USE_MASK_FILES" == 1 ]] && command+=(--starfm-use-mask-files)
[[ "$REUSE_EXISTING_NBAR" == 1 ]] && command+=(--reuse-existing-nbar)
case "$INTERMEDIATE_OUTPUTS" in
  keep) command+=(--keep-intermediate-outputs) ;;
  discard) command+=(--discard-intermediate-outputs-after-validation) ;;
esac

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  "${command[@]}"
fi
