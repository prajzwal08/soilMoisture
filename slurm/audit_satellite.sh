#!/bin/bash
#SBATCH --job-name=audit_sat
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/audit_satellite_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/audit_satellite_%j.err

# ── env ───────────────────────────────────────────────────────────────────────
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate sensei
PYTHON=$(which python)

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job     : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "CPUs    : $SLURM_CPUS_PER_TASK"
echo "Started : $(date)"
echo ""

# ── audit: coverage + nodata (100% sample, 16 workers) ────────────────────────
$PYTHON audit_satellite_data.py \
    --workers 16 \
    --sample-frac 1.0

echo ""
echo "Finished : $(date)"
