#!/bin/bash
#SBATCH --job-name=lst_dT_idx_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lst_tmean_diff/smoke_idx_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Smoke for train_lst_dT_idx.sh (§57): same flags, 20 stations, 4 epochs, warmup 20.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
rm -rf /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/lst_dT_idx_smoke_20261001
bash slurm/train.sh --run-name lst_dT_idx_smoke_20261001 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0.003 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --fine-inputs indices \
    --max-stations 20 --max-epochs 4 --warmup-steps 20 --max-val-batches 50
