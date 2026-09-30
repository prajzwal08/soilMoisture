#!/bin/bash
#SBATCH --job-name=lst_tmean_diff_v2
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_v2_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# LST - T2mean difference run v2 (user 2026-09-30). Changes vs v1 (lst_tmean_diff_era5do03):
#   dT from the START: fixed lambda 0.003 (~ equal push, measured g_sm/g_lst 0.0018-0.0046 in v1)
#   ERA5 whole-window dropout 0.3 -> 0.5   (+ the usual 15% per-day mask)
#   NEW coarse-path dropout 0.3: all 160 m satellite tokens (anchor -> bottleneck, S2/S1 history)
#     withheld, so spatial structure must come from the 20 m fine path
# Unchanged: --lst-target dT_pixel (knee 7.215 K frozen), SIF/TWSA dropout 0, rest = s48_full.
# Selected on val SELECT ubRMSE only. Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name lst_tmean_diff_era5do05_coarse03_20260930 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0.003 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0
