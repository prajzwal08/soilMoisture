#!/bin/bash
#SBATCH --job-name=nan_analysis
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/nan_analysis_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/nan_analysis_%j.err

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

# ── analysis ─────────────────────────────────────────────────────────────────
$PYTHON analyze_nan_filtering.py --workers 14

echo ""
echo "Finished : $(date)"
