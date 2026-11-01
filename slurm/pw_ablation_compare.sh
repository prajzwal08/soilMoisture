#!/bin/bash
#SBATCH --job-name=pw_abl_cmp
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/pw_ablation_compare_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Paired comparison of the pw_ablation array (run afterok the array). Uses the worktree's
# compare_ablation.py so the metric code matches the §24 numbers.
set -euo pipefail
OUT=/gpfs/work3/0/prjs1968/soilMoisture/eval_output/pw_stage2a_L3_ablation
cd /gpfs/work3/0/prjs1968/wt_pw_ablation
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
python compare_ablation.py \
    "$OUT/sat_cross/predictions_val_sat_cross_station_s0.parquet" \
    "$OUT/sat_within/predictions_val_sat_within_station_s0.parquet" \
    "$OUT/era5_cross/predictions_val_era5_cross_station_s0.parquet" \
    --base "$OUT/base/predictions_val.parquet" --csv "$OUT/compare_val.csv"
