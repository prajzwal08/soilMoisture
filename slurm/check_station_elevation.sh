#!/bin/bash
#SBATCH --job-name=elev_check
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §45.12 -- cross-check station elevation_m against MERIT band 4 (elv).
# Read-only; writes only csvs/ and fig/.
set -eo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
python check_station_elevation.py "$@"
