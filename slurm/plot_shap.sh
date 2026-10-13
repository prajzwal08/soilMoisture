#!/bin/bash
#SBATCH --job-name=plot_shap
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/plot_shap_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §73 SHAP figures A-I (mock-ups adapted to K = 5) from shap_<split>.parquet. Pool(64) reads the
# per-station ERA5 rain for E/H from the permanent zarr store. Needs the user's OK to submit.
#   sbatch slurm/plot_shap.sh [RUN] [SPLIT]
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
RUN="${1:-nolst_L3_wu200_sd03_rng2_20261013}"
SPLIT="${2:-oos}"
conda run -n terramind --no-capture-output python plot_shap.py \
    --shap eval_output/${RUN}_shap/shap_${SPLIT}.parquet \
    --per-station eval_output/${RUN}/per_station_${SPLIT}.csv \
    --out-dir figures/eval/${RUN}_paper/shap_${SPLIT} --workers 64
