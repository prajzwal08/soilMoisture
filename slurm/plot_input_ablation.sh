#!/bin/bash
#SBATCH --job-name=plot_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/plot_input_ablation_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §67: re-draw the input-ablation heatmap from an existing ablation_summary.csv (no re-eval).
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
RUN="${RUN:-baseline_selected_20261005}"     # §73: override with RUN= env
conda run -n terramind --no-capture-output python plot_input_ablation.py \
    --summary eval_output/${RUN}_ablation/ablation_summary.csv \
    --out-dir figures/eval/${RUN}_paper/ablation "$@"
echo "=== done $(date) ==="
