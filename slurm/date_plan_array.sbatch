#!/bin/bash
#SBATCH --job-name=starfm_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --mem=32000
#SBATCH --partition=batch
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.edu

set -eo pipefail
module load anaconda
conda activate thesis_paper_env
set -u
cd "${PROJECT_DIR:?PROJECT_DIR required}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec python -u slurm/date_plan.py --execute "${PLAN_FILE:?PLAN_FILE required}" --index "${SLURM_ARRAY_TASK_ID:?}"
