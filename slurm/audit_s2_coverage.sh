#!/bin/bash
#SBATCH --job-name=s2cov
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/s2cov_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Read-only audit: S2 scenes the catalogue offers vs scenes obtained, per station-year.
#   sbatch slurm/audit_s2_coverage.sh --stations ISMN_SCAN_Price ...   # smoke
#   sbatch slurm/audit_s2_coverage.sh                                  # all stations
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
conda activate terramind
python audit_s2_coverage.py --dump-store
conda activate soilmoisture
python audit_s2_coverage.py "$@"
