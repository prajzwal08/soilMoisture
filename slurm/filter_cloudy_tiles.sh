#!/bin/bash
#SBATCH --job-name=filter_cloudy
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/filter_cloudy_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/filter_cloudy_%j.err

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

# ── analyze all 413k tiles, save rejection manifest ───────────────────────────
# No files are deleted. Review text/cloudy_tile_manifest.csv before running --delete.
$PYTHON filter_cloudy_tiles.py --analyze \
    --manifest text/cloudy_tile_manifest.csv

echo ""
echo "Finished : $(date)"
