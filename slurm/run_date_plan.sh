#!/bin/bash
# Run with bash, not sbatch. The Python submitter only prepares/submits jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python3}" slurm/date_plan.py "$@"
