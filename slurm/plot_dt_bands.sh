#!/bin/bash
#SBATCH --job-name=eco_dtplot
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# Re-pair, dump the per-station band table, then draw the bias diagnostics.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "=== eco_dtplot job=$SLURM_JOB_ID $(date) ==="
conda run -n soilmoisture --no-capture-output python count_pairs_dt12.py
echo "--- plotting ---"
conda run -n soilmoisture --no-capture-output python plot_dt_bands.py
echo "=== done $(date) ==="
