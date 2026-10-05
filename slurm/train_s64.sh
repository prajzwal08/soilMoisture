#!/bin/bash
#SBATCH --job-name=nolst_L3_wu200_sd05
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_nolst_L3_wu200_sd05_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# §64: §59 (train_nolst_L3_wu200.sh) + --static-dropout 0.5 ONLY (DEM, LULC, soil each withheld independently,
# trunk + fine tied). Weight decay stays at 0.05 (written explicitly), unlike §60 which also set 0.5. Max 10 epochs.
# Fine inputs = raw bands (the §59 default, written explicitly). Every epoch kept in epochs/epoch_NNN.pt.
# Ablating an epoch later needs --n-layers 3 on the CLI. Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name nolst_L3_wu200_sd05_20261005 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --n-layers 3 --fine-inputs bands --warmup-steps 200 --save-every-epoch \
    --static-dropout 0.5 --weight-decay 0.05 --max-epochs 10
