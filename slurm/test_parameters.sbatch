#!/bin/bash
#SBATCH --job-name=test_parameters
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
module load anaconda
conda activate thesis_paper_env
set -u
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
exec bash slurm/run_experiment.sh parameters
