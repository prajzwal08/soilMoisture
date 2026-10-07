#!/bin/bash
#SBATCH --job-name=slides_results_figs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/slides_results_figs_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Results-slide figures for the frozen baseline (CPU only, reads existing parquets):
#   scatter_pred_obs_kde  -- pred vs obs as smoothed density (50/75/90/99 % bands)
#   txson_cr200-18_side   -- TxSON tile left, six 0-10 cm series in one column right
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
RUN=baseline_selected_20261005
PY="conda run -n terramind --no-capture-output python"
$PY plot_eval_scatter.py --in-dir eval_output/$RUN --out-dir figures/eval/${RUN}_paper \
    --style paper --only scatter_pred_obs_kde
$PY plot_tile_paper.py --tile-station ISMN_TxSON_CR200-18 \
    --stations ISMN_TxSON_CR200-18 ISMN_TxSON_CR200-25 ISMN_TxSON_CR1000-2 \
               ISMN_TxSON_CR200-24 ISMN_TxSON_CR200-15 ISMN_TxSON_CR200-6 \
    --name txson_cr200-18_side --layout side
echo "=== done $(date) ==="
