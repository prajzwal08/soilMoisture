#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --array=0-1047
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/tokenize_%A_%a.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/tokenize_%A_%a.err

# ── env ───────────────────────────────────────────────────────────────────────
PYTHON=/gpfs/home5/pkhanal/miniforge3/envs/terramind/bin/python
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job       : $SLURM_JOB_ID  array task: $SLURM_ARRAY_TASK_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started   : $(date)"
echo ""

# ── tokenization (1 station per array task) ───────────────────────────────────
$PYTHON precompute_terramind.py \
    --batch-size 8 \
    --start-idx $SLURM_ARRAY_TASK_ID \
    --end-idx   $((SLURM_ARRAY_TASK_ID + 1))

echo ""
echo "Finished  : $(date)"
