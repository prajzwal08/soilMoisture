#!/bin/bash
#SBATCH --job-name=sm_check_data
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/check_dataset_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail

cd /gpfs/work3/0/prjs1968/soilMoisture

CATEGORIES=(sm_only sm_only sm_and_flux sm_and_flux flux_only flux_only)
SPLITS=(train val train val train val)

CATEGORY=${CATEGORIES[$SLURM_ARRAY_TASK_ID]}
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

echo "Job ID:    $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:      $SLURM_NODELIST"
echo "Category:  $CATEGORY  |  Split: $SPLIT"

conda run -n terramind python -u check_dataset.py \
    --workers 64 \
    --batch-size 8 \
    --category "$CATEGORY" \
    --split "$SPLIT"
