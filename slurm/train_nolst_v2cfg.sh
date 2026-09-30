#!/bin/bash
#SBATCH --job-name=nolst_v2cfg
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_nolst_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# No-LST control (user 2026-09-30: "may be we are better without LST"). IDENTICAL to v2
# (lst_tmean_diff_era5do05_coarse03_20260930) except --lambda-lst 0 = no thermal gradient at all.
# --lst-target dT_pixel kept so the (untrained) LST head's val stats still print as a reference.
# Compare val SELECT with v2's best. Own run name + checkpoint dir. Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name nolst_era5do05_coarse03_20260930 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0
