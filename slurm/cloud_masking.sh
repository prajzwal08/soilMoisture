#!/bin/bash
#SBATCH --job-name=cloud_mask
#SBATCH --array=0-1047
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/cloud_mask_%A_%a.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/cloud_mask_%A_%a.err

# ── env ───────────────────────────────────────────────────────────────────────
PYTHON=/gpfs/home5/pkhanal/miniforge3/envs/sensei/bin/python

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job       : $SLURM_JOB_ID  array task: $SLURM_ARRAY_TASK_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo ""

# ── cloud masking (1 station per array task) ──────────────────────────────────
$PYTHON cloud_masking_inference.py \
    --start-idx $SLURM_ARRAY_TASK_ID \
    --end-idx   $((SLURM_ARRAY_TASK_ID + 1))

echo ""
echo "Finished  : $(date)"
