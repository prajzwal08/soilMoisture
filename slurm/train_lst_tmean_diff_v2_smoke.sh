#!/bin/bash
#SBATCH --job-name=lst_tmean_diff_v2_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lst_tmean_diff/smoke_v2_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Smoke for train_lst_tmean_diff_v2.sh: same flags, 20 stations, 4 epochs, warmup 20.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
rm -rf /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/lst_tmean_diff_v2_smoke_20260930
bash slurm/train.sh --run-name lst_tmean_diff_v2_smoke_20260930 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0.003 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --max-stations 20 --max-epochs 4 --warmup-steps 20 --max-val-batches 50
