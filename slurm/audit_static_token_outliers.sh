#!/bin/bash
#SBATCH --job-name=static_outliers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/static_outliers_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §27a.3 -- what is the DEM token carrying 64.8% of one tile's embedding variance?
# Deep dive on one tile + sweep over every station on disk.
#
#   sbatch slurm/audit_static_token_outliers.sh
#   sbatch slurm/audit_static_token_outliers.sh --tile ISMN_TxSON_CR200-3

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate terramind

cd /gpfs/work3/0/prjs1968/soilMoisture

python audit_static_token_outliers.py --workers 64 "$@"
