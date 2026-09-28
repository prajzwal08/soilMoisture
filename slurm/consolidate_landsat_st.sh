#!/bin/bash
#SBATCH --job-name=lst22
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lst22_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §46.5-B / §48 — warp every station's st30 bundle to the 22x22 @ 100 m target once.
# Writes {DATA_ROOT}/{cat}/{folder}/LANDSAT_ST/{folder}_lst22.npz + csvs/lst22_index.csv.
#
#   sbatch slurm/consolidate_landsat_st.sh --dry-run --limit 8    # smoke
#   sbatch slurm/consolidate_landsat_st.sh

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python consolidate_landsat_st.py "$@"
