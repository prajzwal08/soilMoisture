#!/bin/bash
#SBATCH --job-name=nolst_L3_wu200_sd03_rng2
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lst_tmean_diff/train_nolst_L3_wu200_sd03_rng2_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# §74 (2026-10-13, user OK on the design): §64 (train_s64.sh) with TWO changes:
#   --static-dropout 0.3 (was 0.5) = equal to --coarse-dropout 0.3, so the less-dropped group does not
#     absorb the shared site-level information (§64: sat ablation 37% -> 73-94%, statics -> ~0);
#   --lambda-range 2 (soft hinge outside [0, 0.9] on the whole SM map; review: lambda ~2).
# Max 10 epochs (compare at ep10); early stop 4, LR patience 2 as §64; every epoch saved; user picks the epoch.
# Runs from branch feat/sm-range-penalty: do NOT switch branches in this checkout while it is queued.
# Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

bash slurm/train.sh --run-name nolst_L3_wu200_sd03_rng2_20261013 \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff \
    --lst-target dT_pixel --lambda-lst 0 \
    --era5-dropout 0.5 --coarse-dropout 0.3 --sif-twsa-dropout 0 \
    --n-layers 3 --fine-inputs bands --warmup-steps 200 --save-every-epoch \
    --static-dropout 0.3 --weight-decay 0.05 --max-epochs 10 --early-stop-patience 4 --lr-patience 2 \
    --lambda-range 2 --sm-range-lo 0.0 --sm-range-hi 0.9
