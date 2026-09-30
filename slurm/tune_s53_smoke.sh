#!/bin/bash
#SBATCH --job-name=s53_smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:00:00
#SBATCH --output=logs/s48_tune/smoke_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §53 smoke for the stage-1 config (dT_pixel + era5-dropout 0.3 + lambda-frac 1): 20 stations,
# 4 epochs, warmup 20. Stage 1 is submitted afterok on this job.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
rm -rf /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/s48_tune/s53_smoke_20260930
bash slurm/train.sh --run-name s53_smoke_20260930 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/s48_tune \
    --lst-target dT_pixel --era5-dropout 0.3 --lambda-frac 1 \
    --max-stations 20 --max-epochs 4 --warmup-steps 20 --max-val-batches 50
