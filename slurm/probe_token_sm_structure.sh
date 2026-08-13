#!/bin/bash
#SBATCH --job-name=token_sm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/token_sm_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §27b Scale A part 1 -- does the embedding space organise itself by wetness?
# Station centre token (index 105) -> mean soil moisture. kNN / k-means / UMAP.
#   sbatch slurm/probe_token_sm_structure.sh
#   sbatch slurm/probe_token_sm_structure.sh --limit 120 --max-dates 8   # smoke

set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
python probe_token_sm_structure.py --workers 64 "$@"
