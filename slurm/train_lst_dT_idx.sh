#!/bin/bash
#SBATCH --job-name=lst_dT_idx
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_idx_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# §57 final run (user 2026-10-01). v2 (lst_tmean_diff_era5do05_coarse03_20260930) with ONE change:
#   --fine-inputs indices: the 20 m path sees NDVI, NDMI, VV, VH/VV (dB) instead of 10 S2 bands
#   + VV/VH. DEM (asinh relief) and the LULC embedding unchanged. Converted on the GPU from the
#   same fine cache. ERA5 whole dropout 0.5, coarse 0.3, dT_pixel, lambda 0.003 as v2.
# Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name lst_dT_idx_era5do05_coarse03_20261001 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0.003 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --fine-inputs indices
