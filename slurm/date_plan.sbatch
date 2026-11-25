#!/bin/bash
#SBATCH --job-name=starfm_dates
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --partition=batch
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.edu

set -eo pipefail
module load anaconda
conda activate thesis_paper_env
set -u
cd "${PROJECT_DIR:?PROJECT_DIR required}"
exec python -u slurm/date_plan.py --execute "${PLAN_FILE:?PLAN_FILE required}" --index "${PLAN_INDEX:?}"
