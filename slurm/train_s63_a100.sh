#!/bin/bash
#SBATCH --job-name=nolst_L1_wu200_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_nolst_L1_wu200_a100_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# §63: §59 (train_nolst_L3_wu200.sh) with 1 transformer layer instead of 3; nothing else changed. Max 10 epochs (user).
# Fine inputs = raw bands (the §59 default, written explicitly). Every epoch kept in epochs/epoch_NNN.pt.
# A100 variant (H100 queue, 2026-10-05): 480G max node RAM, --num-workers 8, own run folder; raced against the H100 job.
# Ablating an epoch later needs --n-layers 1 on the CLI. Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name nolst_L1_wu200_a100_20261005 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --n-layers 1 --fine-inputs bands --warmup-steps 200 --save-every-epoch --max-epochs 10 --num-workers 8
