#!/bin/bash
#SBATCH --job-name=era5_rad_check
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §43.12 run-order step 2 -- is the radiation physical, or did we fetch the
# ACCUMULATED band?  Run after every smoke and before the splice.
set -eo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture
cd /gpfs/work3/0/prjs1968/soilMoisture
python check_era5_radiation.py "$@"
