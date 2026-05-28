#!/bin/bash
#SBATCH --job-name=cleanup_t1
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/cleanup_tier1_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/cleanup_tier1_%j.err

# ── env ───────────────────────────────────────────────────────────────────────
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate sensei
PYTHON=$(which python)

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "Started  : $(date)"
echo ""

# ── cleanup ───────────────────────────────────────────────────────────────────
# Phase A: delete 17 excluded station directories entirely (scratch + data)
# Phase B: delete remaining individual tier-1 failing tiles
$PYTHON cleanup_tier1_failures.py --workers 14

echo ""
echo "Finished : $(date)"
