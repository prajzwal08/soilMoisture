#!/bin/bash
#SBATCH --job-name=ls_st30_check
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# Visual check of the Landsat ST bundles. Downloads nothing; runs in soilmoisture only so it
# can import the reference qa_decode from download_landsat_st30 instead of copying it.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n soilmoisture --no-capture-output python -m py_compile plot_landsat_st30_check.py
conda run -n soilmoisture --no-capture-output python plot_landsat_st30_check.py "$@"
