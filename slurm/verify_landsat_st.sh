#!/bin/bash
#SBATCH --job-name=ls_st30_verify
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Reads every Landsat ST bundle back. Exits 1 if any station fails, so the job turns red.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "=== verify landsat st30  job=$SLURM_JOB_ID  $(date) ==="
conda run -n soilmoisture --no-capture-output python -m py_compile verify_landsat_st.py
conda run -n soilmoisture --no-capture-output python verify_landsat_st.py --workers 64 "$@"
echo "=== done $(date) ==="
