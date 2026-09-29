#!/bin/bash
#SBATCH --job-name=bf_dl
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bf_dl_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -uo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
R=$1   # run dir: stations.txt, backup.ok, *.ok
grep -Fxf csvs/backfill_runs/backup.ok $R/stations.txt > $R/go.txt
conda activate soilmoisture
python backfill_s2_download.py --download --stations-file $R/go.txt --workers 16 &&
python backfill_s2_download.py --harmonise --stations-file $R/go.txt --workers 16
