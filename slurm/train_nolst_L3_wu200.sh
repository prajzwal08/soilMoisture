#!/bin/bash
#SBATCH --job-name=nolst_L3_wu200
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_nolst_L3_wu200_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# No-LST baseline (train_nolst_v2cfg.sh, 27417115) with 3 transformer layers instead of 6 and a
# 200-step warmup (~1 epoch) instead of 1000 (~5 epochs). --save-every-epoch keeps
# epochs/epoch_NNN.pt so any epoch can be chosen later (user 2026-10-05). Per epoch the log has
# SELECT, the val/train gap line and the two [diag] fine-path lines. Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name nolst_L3_wu200_20261005 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --n-layers 3 --warmup-steps 200 --save-every-epoch
