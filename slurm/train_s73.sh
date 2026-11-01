#!/bin/bash
#SBATCH --job-name=s73_shapready
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_s73_shapready_%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §73 SHAP-ready retrain = §64 (train_s64.sh: §59 + static dropout 0.5, wd 0.05) plus
#   --coarse-dropout-mode per_sensor (S2, S1 160 m each 0.3, anchor re-picked from the kept sensor),
#   --fine-group-dropout 0.1 (fine S2+S1 together; one-of modality dropout 0.2 kept),
#   --sif-twsa-dropout 0.5 (was 0 in §59).
# Every Shapley player (era5, s2, s1, fine, dem, lulc, soil, sif, twsa) is seen absent in training.
# GATE (runbook §73): eval vs baseline_selected_20261005 before ANY SHAP work; pass = SELECT <= 0.0498
# at ep10 and no split/depth ubRMSE > ~2 % worse. Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name nolst_L3_wu200_shapready_20261012 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --coarse-dropout-mode per_sensor \
    --fine-group-dropout 0.1 --sif-twsa-dropout 0.5 \
    --n-layers 3 --fine-inputs bands --warmup-steps 200 --save-every-epoch \
    --static-dropout 0.5 --weight-decay 0.05 --max-epochs 10
