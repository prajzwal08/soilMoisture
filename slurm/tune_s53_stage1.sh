#!/bin/bash
#SBATCH --job-name=s53_frac
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/s48_tune/stage1_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

# §53 tuning STAGE 1: lambda_frac in {0.3, 1, 3}. Fixed: --lst-target dT_pixel (knee 7.215 K
# frozen in csvs/lst_dT_stats.json), --era5-dropout 0.3, everything else = s48_full.
# Judged on val SELECT ubRMSE only (OOS/OOT untouched). One 4-GPU node per task, so two tasks
# never share a node's /dev/shm staging (train.sh clears /dev/shm/s48_* on start).
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

FRACS=(0.3 1 3)
FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}
RUN="s53_s1_frac${FRAC}_20260930"
echo "array task ${SLURM_ARRAY_TASK_ID}: lambda_frac=${FRAC} run=${RUN}"

bash slurm/train.sh --run-name "${RUN}" \
    --checkpoint-dir /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/s48_tune \
    --lst-target dT_pixel --era5-dropout 0.3 --lambda-frac "${FRAC}"
