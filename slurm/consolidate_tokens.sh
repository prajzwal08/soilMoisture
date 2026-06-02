#!/bin/bash
#SBATCH --job-name=consolidate_tokens
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/consolidate_tokens_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/consolidate_tokens_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# ── env ───────────────────────────────────────────────────────────────────────
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
PYTHON=$(which python)

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job     : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "CPUs    : $SLURM_CPUS_PER_TASK"
echo "Started : $(date)"
echo ""

# ── consolidation ─────────────────────────────────────────────────────────────
$PYTHON consolidate_tokens.py --execute

echo ""
echo "Finished : $(date)"
