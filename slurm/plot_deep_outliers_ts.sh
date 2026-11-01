#!/bin/bash
#SBATCH --job-name=s74_ts_raincloud
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/deep_outliers_ts_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §74 (2026-10-13): time series of the 8 OOS stations with the worst 30-100 cm NSE_anom
# (near-flat observed deep SM, model swings 2-6x the observed SD). Ranked from per_station_oos.csv.
# Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
RUN="${RUN:-nolst_L3_wu200_sd03_rng2_20261013}"
RUN_PY="conda run -n terramind --no-capture-output python"
$RUN_PY plot_eval_timeseries.py --in-dir "eval_output/${RUN}" \
    --out-dir "figures/eval/${RUN}_paper/timeseries_deep_outliers" --splits oos \
    --select named --style paper --stations \
        ISMN_SNOTEL_BurtsMillerRanch ISMN_SNOTEL_GrizzlyPeak ISMN_SNOTEL_HiltsCreek \
        ISMN_COSMOS-UK_HolmeLacy ISMN_SNOTEL_LamoilleUpper ISMN_SNOTEL_Sharkstooth \
        ISMN_SNOTEL_RockwoodGS ICOS_DE-RuS

# §74 sample raincloud plots (half violin + box + station dots, ideal line), user asked for all four metrics
OUT=figures/eval/${RUN}_paper/raincloud_sample
for M in R2_pearson ubRMSE NSE NSE_anom; do
    $RUN_PY plot_eval_boxplot.py --in-dir "eval_output/${RUN}" --out-dir "${OUT}" \
        --splits oos oot oost --metric ${M} --kind raincloud --style paper
done
