#!/bin/bash
#SBATCH --job-name=s74_raincloud
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/raincloud_sample_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §74 raincloud samples only (re-run after the agent figure review). Needs the user's OK to submit.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
RUN="${RUN:-nolst_L3_wu200_sd03_rng2_20261013}"
RUN_PY="conda run -n terramind --no-capture-output python"
OUT=figures/eval/${RUN}_paper/raincloud_sample
for M in R2_pearson ubRMSE NSE NSE_anom; do
    $RUN_PY plot_eval_boxplot.py --in-dir "eval_output/${RUN}" --out-dir "${OUT}" \
        --splits oos oot oost --metric ${M} --kind raincloud --style paper
done
