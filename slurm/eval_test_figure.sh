#!/bin/bash
#SBATCH --job-name=eval_test_fig
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_test_figure_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §66: ONE test figure in the paper style (user: "first make one figure, if I am happy we go with all").
# 1) metrics re-computed with predictions clipped to [0, 1]  2) ubRMSE by land cover, --style paper
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1 EVAL_CLIP=0,1
IN=eval_output/baseline_selected_20261005
OUT=figures/eval/baseline_selected_20261005_paper
RUN_PY="conda run -n terramind --no-capture-output python"
$RUN_PY eval_metrics.py --in-dir "${IN}" --out-dir "${IN}"
$RUN_PY - <<'PY'
import matplotlib.font_manager as fm
for name in ["Times New Roman", "Nimbus Roman", "STIXGeneral"]:
    try:
        print(f"font {name!r} -> {fm.findfont(name, fallback_to_default=False)}")
    except Exception as e:
        print(f"font {name!r} -> NOT FOUND ({type(e).__name__})")
PY
$RUN_PY plot_eval_ecosystem.py --in-dir "${IN}" --out-dir "${OUT}" --by igbp_macro --style paper --no-val
echo "=== done $(date) ==="
