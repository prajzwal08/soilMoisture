#!/bin/bash
#SBATCH --job-name=era5_splice
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §43.12 -- splice ssrd_sum/strd_sum onto era5/values, writing era5/values18
# BESIDE it.  Dry run by default; pass --execute to write.
#
#   sbatch slurm/splice_era5_radiation.sh                    # dry run first
#   sbatch slurm/splice_era5_radiation.sh --execute
#
# Read the gap report before executing.  zarr_tokens is the ONLY copy of the
# drivers, so nothing here overwrites era5/values.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
python splice_era5_radiation.py --workers 64 "$@"
