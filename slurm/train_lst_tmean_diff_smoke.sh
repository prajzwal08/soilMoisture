#!/bin/bash
#SBATCH --job-name=lst_tmean_diff_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lst_tmean_diff/smoke_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Smoke for the LST - T2mean difference run (dT_pixel + era5-dropout 0.3 + frac 1): 20 stations,
# 4 epochs, warmup 20. The full run is submitted separately, only with the user's OK.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
rm -rf /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/lst_tmean_diff_smoke_20260930
bash slurm/train.sh --run-name lst_tmean_diff_smoke_20260930 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --era5-dropout 0.3 --sif-twsa-dropout 0 --lambda-frac 1 \
    --max-stations 20 --max-epochs 4 --warmup-steps 20 --max-val-batches 50
