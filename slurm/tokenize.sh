#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --array=0-99
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/tokenize_%A_%a.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/tokenize_%A_%a.err

# ── env ───────────────────────────────────────────────────────────────────────
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
PYTHON=$(which python)
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job       : $SLURM_JOB_ID  array task: $SLURM_ARRAY_TASK_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started   : $(date)"
echo ""

# ── tokenization (10 stations per array task) ─────────────────────────────────
# GPU is I/O-bound (~80% time saving .pt files, only 3% SM util measured).
# Grouping 10 stations amortises torch.compile warmup and reduces job overhead
# by 10x. Tasks 0-99 handle 10 stations each; task 100 handles the last 10
# (indices 1000-1009), giving 1010 stations total.
START=$(( SLURM_ARRAY_TASK_ID * 10 ))
END=$(( START + 10 ))

$PYTHON precompute_terramind.py \
    --batch-size  8       \
    --num-workers 3       \
    --compile             \
    --start-idx $START    \
    --end-idx   $END

echo ""
echo "Finished  : $(date)"
