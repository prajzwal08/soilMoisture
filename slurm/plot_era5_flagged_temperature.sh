#!/bin/bash
#SBATCH --job-name=era5_flag_temp
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §43.12 -- annual temperature cycle of the stations the radiation check flagged.
# Falsifies (or confirms) the "too-tight threshold" reading before any constant
# is loosened.  Read-only: touches no store.
set -eo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
python plot_era5_flagged_temperature.py "$@"
