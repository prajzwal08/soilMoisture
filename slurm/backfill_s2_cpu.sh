#!/bin/bash
#SBATCH --job-name=bf_dl
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bf_dl_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50 phases 1-2: download the target scenes, then harmonise by baseline. Staging only.
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture
cd /gpfs/work3/0/prjs1968/soilMoisture
python backfill_s2_download.py --download "$@"
python backfill_s2_download.py --harmonise "$@"
