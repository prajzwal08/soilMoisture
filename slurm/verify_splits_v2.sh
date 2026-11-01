#!/bin/bash
#SBATCH --job-name=verify_splits_v2
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/verify_splits_v2_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §47.9 — the pre-registered gate. Recomputes the geometry independently of the script
# that wrote the flags, then builds the train/val/oot/oost datasets for real.
#
#   sbatch slurm/verify_splits_v2.sh                  full, opens every zarr store
#   sbatch slurm/verify_splits_v2.sh --skip-datasets  checks 1-4 and 7 only, ~1 min

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
# The dataset builds are I/O bound and single-threaded: job 27287679 ran 28 min at 18.5% CPU
# in state D and had not finished the first of four. Unbuffered, or a walltime kill takes the
# buffer with it and even the checks that already passed are lost.
export PYTHONUNBUFFERED=1
conda run -n terramind --no-capture-output python -u verify_splits_v2.py "$@"
