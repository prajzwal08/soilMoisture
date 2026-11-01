#!/bin/bash
#SBATCH --job-name=bf_cm
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bf_cm_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -uo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
R=$1   # run dir: stations.txt, backup.ok, *.ok
conda activate sensei
python cloud_masking_inference.py --scratch-dir /gpfs/scratch1/shared/pkhanal/s2_backfill \
  --data-dir /gpfs/scratch1/shared/pkhanal/s2_backfill_cm --stations-file $R/go.txt --batch-size 16 --io-workers 8 &&
python backfill_s2_download.py --filter --stations-file $R/go.txt --workers 16
