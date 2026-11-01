#!/bin/bash
#SBATCH --job-name=sm_nolst
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=30:00:00
#SBATCH --output=logs/s48_lst_ablation/train_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# LST-aux control for s48_full_20260929: identical code/config/seed, only --lambda-lst 0.
# Checkpoints and logs go to their own folders so they never mix with the main runs.
# Compare against s48_full_20260929: SELECT ubRMSE per depth, within-r, fine input-grad
# ratio per epoch (does it still fall ~0.15 -> ~0.05 at ep7 without LST?), TxSON CR200-18.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

RUN="${RUN:-s48_nolst_20260930}"
CKPT_ROOT=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/s48_lst_ablation

bash slurm/train.sh --run-name "${RUN}" --lambda-lst 0 --checkpoint-dir "${CKPT_ROOT}"
