#!/bin/bash
#SBATCH --job-name=starfm_baselines
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=24000
#SBATCH --partition=batch
#SBATCH --no-requeue

# Submit with: sbatch --array=0-<N-1>%30 --export=ALL,MANIFEST_LIST=...,OUTPUT_DIR=... slurm/baselines_array.sbatch
set -eo pipefail
module load anaconda
conda activate thesis_paper_env
set -u
cd "${PROJECT_DIR:?PROJECT_DIR required}"
exec python -u evaluate_baselines.py --manifest-list "${MANIFEST_LIST:?}" --index "${SLURM_ARRAY_TASK_ID:?}" --output-dir "${OUTPUT_DIR:?}"
