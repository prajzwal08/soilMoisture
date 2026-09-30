#!/bin/bash
#SBATCH --job-name=fine_ablation_v2
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lst_tmean_diff/fine_ablation_v2_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# EVAL ONLY (user 2026-09-30: "does the fine path drive LST or not"). Loads v2's best.pt and
# scores val three times: fine path intact / shuffled across stations / zeroed. Writes no
# checkpoint and no W&B run. Flags must match v2 (the resume guard checks lst_target).
# Run only AFTER the v2 training job has finished (best.pt must be final).
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name lst_tmean_diff_era5do05_coarse03_20260930 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0.003 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --eval-fine-ablation --eval-ckpt best.pt
