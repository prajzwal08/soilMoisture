#!/bin/bash
#SBATCH --job-name=scale_ablation_nolst
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:30:00
#SBATCH --output=logs/lst_tmean_diff/scale_ablation_nolst_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# EVAL ONLY (user 2026-09-30: tests 1 + 3, "does the SM map use 160 m or 20 m info"). Loads the no-LST best.pt and
# scores val 5 ways: fine intact / shuffled / zeroed / 14x14 bottleneck flattened / both, plus a
# between- vs within-160 m-cell split of the SM map. Writes no checkpoint and no W&B run.
# Run only AFTER the no-LST training job (27417115) has finished (best.pt must be final).
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name nolst_era5do05_coarse03_20260930 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --eval-fine-ablation --eval-ckpt best.pt
