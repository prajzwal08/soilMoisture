#!/bin/bash
#SBATCH --job-name=era5_verify18
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §43.12 -- read back every era5/values18 and prove it.  Run after the splice.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
python verify_era5_18.py --workers 64 "$@"
