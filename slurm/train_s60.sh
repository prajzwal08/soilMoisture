#!/bin/bash
#SBATCH --job-name=s60_sd05_wd05
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_s60_sd05_wd05_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# §60: §59 (no-LST, 3 layers, warmup 200, every epoch kept) + --static-dropout 0.5 (DEM, LULC, soil each
# withheld independently, trunk + fine tied) + --weight-decay 0.5 (was 0.05). Max 15 epochs.
# Anti-memorisation run (user 2026-10-05). Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name nolst_L3_wu200_sd05_wd05_20261005 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --n-layers 3 --warmup-steps 200 --save-every-epoch \
    --static-dropout 0.5 --weight-decay 0.5 --max-epochs 15
