#!/bin/bash
#SBATCH --job-name=bf_phase0
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bf_phase0_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §50 phase 0: catalogue scan (soilmoisture) -> existing-store stats (terramind) -> report +
# target list (terramind). Read-only on every store.
#   sbatch slurm/backfill_phase0.sh --stations ISMN_SCAN_Price ...   # smoke
#   sbatch slurm/backfill_phase0.sh                                  # all stations
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
conda activate soilmoisture
python backfill_catalogue.py --catalogue "$@"
conda activate terramind
python backfill_catalogue.py --store "$@"
python backfill_catalogue.py --report "$@"
