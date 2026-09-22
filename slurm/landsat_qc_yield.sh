#!/bin/bash
#SBATCH --job-name=ls_qc_yield
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Step 0 of the 993-station Landsat ST pull -- QC yield curves.
# NO DOWNLOAD: reads the ~14.7k tifs already on disk from the cancelled job 27015709.
# Sets the ST_QA and 100 m cell-mask thresholds before the pull bakes them in.
#
# soilmoisture env (rasterio/pandas). Never combined with analysis in one job.

set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "=== landsat qc yield  job=$SLURM_JOB_ID  $(date) ==="

conda run -n soilmoisture --no-capture-output python -m py_compile qc_landsat_yield.py
echo "--- syntax OK ---"

conda run -n soilmoisture --no-capture-output python qc_landsat_yield.py \
    --workers 64 "$@"

echo "=== done $(date) ==="
