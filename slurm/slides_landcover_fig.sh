#!/bin/bash
#SBATCH --job-name=slides_landcover_fig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/slides_landcover_fig_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Wide (2:1) per-station ubRMSE by land cover for the results slide; the paper figure is untouched
# (separate out-dir). CPU only, existing per-station CSVs.
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
RUN=baseline_selected_20261005
conda run -n terramind --no-capture-output python plot_eval_ecosystem.py \
    --in-dir eval_output/$RUN --out-dir figures/eval/${RUN}_paper/slides \
    --by igbp_macro --style paper --no-val --fig-size 14 7
echo "=== done $(date) ==="
