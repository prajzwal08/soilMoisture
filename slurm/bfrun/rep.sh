#!/bin/bash
#SBATCH --job-name=bf_rep
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bf_rep_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -uo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
R=$1   # run dir: stations.txt, backup.ok, *.ok
grep -Fxf csvs/backfill_runs/backup.ok $R/stations.txt > $R/go.txt
conda activate terramind
python backfill_repair.py --raw --stations-file $R/go.txt --ok-out $R/raw.ok --workers 16
conda activate sensei
python cloud_masking_inference.py --scratch-dir /gpfs/scratch1/shared/pkhanal/s2_repair \
  --data-dir /gpfs/scratch1/shared/pkhanal/s2_repair_cm --stations-file $R/raw.ok --batch-size 16 --io-workers 8
conda activate terramind
python backfill_repair.py --encode --stations-file $R/raw.ok --ok-out $R/enc.ok
