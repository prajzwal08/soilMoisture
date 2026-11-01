#!/bin/bash
#SBATCH --job-name=cdist_clim
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n soilmoisture --no-capture-output python probe_cdist_climate.py
