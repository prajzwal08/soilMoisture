#!/bin/bash
#SBATCH --job-name=slides_density_fig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/slides_density_fig_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Pred vs obs as a density-coloured scatter (results slide), CPU only, existing parquets.
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
RUN=baseline_selected_20261005
conda run -n terramind --no-capture-output python plot_eval_scatter.py \
    --in-dir eval_output/$RUN --out-dir figures/eval/${RUN}_paper --style paper \
    --only hist_pred_obs
echo "=== done $(date) ==="
