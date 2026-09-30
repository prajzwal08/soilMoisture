#!/bin/bash
#SBATCH --job-name=probe_offsets
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=01:30:00
#SBATCH --output=logs/probe_offsets_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Step-0 feasibility (user 2026-09-30): can TerraMind cell embeddings predict which spots are
# wetter than their tile? Two ridge regressions, scored on val/oos and on ALL within-tile
# station pairs. Read-only; writes csvs/probe_offsets/.
#   sbatch slurm/probe_offsets.sh --max-tiles 80    # smoke
#   sbatch slurm/probe_offsets.sh                   # full
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
conda activate terramind
python probe_offsets.py --workers 64 "$@"
