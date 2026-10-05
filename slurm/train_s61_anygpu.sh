#!/bin/bash
#SBATCH --job-name=s61_idx_any
#SBATCH --partition=gpu_h100,gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_s61_anygpu_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# §61: §59 (no-LST, 3 layers, warmup 200, every epoch kept) + fine inputs = indices (NDVI, NDMI, VV,
# VH/VV + DEM, LULC) + no zero-init of the decoder skips + coarse dropout 0.5 + wd 0.2 + lr 1e-4, max 10
# epochs. Static dropout off. Goal: make the decoder use the 20 m path. Needs the user's OK to submit.
# ANY-GPU variant (2026-10-05, user option B): partition gpu_h100,gpu_a100 -> starts on whichever frees first;
# A100-safe settings everywhere (node RAM 480 GB) -> --mem=480G (max) and
# --num-workers 8 (was 12) for memory headroom; previous full runs peaked ~230 GB RSS + ~186 GB /dev/shm.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name nolst_L3_idx_nzi_cd05_wd02_lr1e4_anygpu_20261005 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.5 --sif-twsa-dropout 0 \
    --n-layers 3 --warmup-steps 200 --save-every-epoch \
    --fine-inputs indices --no-skip-zero-init --weight-decay 0.2 --lr 1e-4 --max-epochs 10 --num-workers 8
