#!/bin/bash
#SBATCH --job-name=lst_tmean_diff
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# LST - T2mean difference run (§53, user 2026-09-30; a single run, not a tuning sweep):
#   thermal target   --lst-target dT_pixel   Huber vs LST_obs - t2m_mean per valid cell, knee 7.215 K
#   ERA5             whole-window dropout 0.3 + the usual 15% per-day masking
#   SIF / TWSA       no dropout (0)
#   lambda_frac      1  (LST pulls on the shared map as hard as SM; user rejected 0.3)
# Everything else = s48_full. Selected on val SELECT ubRMSE only.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name lst_tmean_diff_era5do03_20260930 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --era5-dropout 0.3 --sif-twsa-dropout 0 --lambda-frac 1
