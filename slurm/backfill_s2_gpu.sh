#!/bin/bash
#SBATCH --job-name=bf_cm
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bf_cm_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50 phase 3: SEnSeIv2 on the staged scenes (existing script, redirected), then the filter.
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate sensei
cd /gpfs/work3/0/prjs1968/soilMoisture
python cloud_masking_inference.py --scratch-dir /gpfs/scratch1/shared/pkhanal/s2_backfill \
    --data-dir /gpfs/scratch1/shared/pkhanal/s2_backfill_cm --batch-size 16 --io-workers 3
python backfill_s2_download.py --filter "$@"
