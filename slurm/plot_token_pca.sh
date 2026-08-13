#!/bin/bash
#SBATCH --job-name=token_pca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
# One tile, four dates per layer -- each read is a single index into
# (N,196,768) fp16, so the memory footprint is tens of MB.  64G is headroom
# for the raw 224x224 S2/S1/DEM/LULC rasters and matplotlib, not a measured need.
#SBATCH --time=00:30:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/token_pca_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §27a -- PCA->RGB token maps for one tile: every layer, every season, raw vs pooled.
# See text/training_runbook.md §27.7 (is the diversity the RIGHT diversity?) and §27.8
# (what is actually pooled: the history and DEM/LULC, NOT the anchor).
#
#   sbatch slurm/plot_token_pca.sh                            # CR200-18, 2019-2020
#   sbatch slurm/plot_token_pca.sh --tile ISMN_TxSON_CR200-3  # another dense tile

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate terramind

cd /gpfs/work3/0/prjs1968/soilMoisture

python plot_token_pca.py "$@"

echo "figures:"
ls -la figures/token_pca/ 2>/dev/null || echo "  (none written)"
