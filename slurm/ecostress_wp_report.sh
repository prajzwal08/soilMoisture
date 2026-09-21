#!/bin/bash
#SBATCH --job-name=eco_wpreport
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# Merge every shard CSV and print the granule / pair tables.  No COG is opened.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
EXTRA="${1:-}"
echo "=== eco_wpreport job=$SLURM_JOB_ID $(date) ==="
conda run -n soilmoisture --no-capture-output python qc_wellphased_pairs.py \
    --out-tag wp --report-only $EXTRA
echo "=== done $(date) ==="
