#!/bin/bash
#SBATCH --job-name=eco_dt12
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# Re-pair ECOSTRESS under clear-first + dt<=12 h and count images.  No COG is opened.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "=== eco_dt12 job=$SLURM_JOB_ID $(date) ==="
conda run -n soilmoisture --no-capture-output python count_pairs_dt12.py
echo "=== done $(date) ==="
