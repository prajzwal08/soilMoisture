#!/bin/bash
#SBATCH --job-name=verify_restage
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verify_restage_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Counts chunk files on both sides and demands the required keys exist as real files.
# Never looks at `.complete` -- the sentinel is copied along with everything else.
#
#   sbatch slurm/verify_restage.sh tokens
#   sbatch slurm/verify_restage.sh imagery

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python verify_restage.py --store "${1:-tokens}" --workers 64
