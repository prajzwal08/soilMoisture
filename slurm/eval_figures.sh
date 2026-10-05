#!/bin/bash
#SBATCH --job-name=eval_figures
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_figures_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#
# The CPU half of the held-out evaluation (§35.33).  Everything here reads the
# parquets written by slurm/eval_predict.sh; no GPU, no re-inference.
#
#   sbatch slurm/eval_predict.sh <run> best.pt --out-dir eval_output/<run>   # GPU
#   sbatch slurm/eval_figures.sh <run>                                        # this
#
# Outputs are keyed by run name so a second run never overwrites the first:
#   eval_output/<run>/     metrics_summary.csv, per_station_*.csv, gate.json
#   figures/eval/<run>/    scatter, boxplot, ecosystem, timeseries
#
# Extra args after the run name are forwarded to every plot script, so e.g.
#   sbatch slurm/eval_figures.sh <run> --splits oos oost
# narrows the whole sweep at once.

set -eo pipefail
exec 2>&1

cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1

RUN="${1:?usage: sbatch slurm/eval_figures.sh <run-name> [extra plot args]}"
shift
IN="eval_output/${RUN}"
OUT="${OUT:-figures/eval/${RUN}}"      # e.g. OUT=figures/eval/<run>_bw for --style bw
RUN_PY="conda run -n terramind --no-capture-output python"

echo "=== eval_figures  job ${SLURM_JOB_ID}  started $(date) ==="
echo "Node: $(hostname)"
echo "Run:  ${RUN}"
echo "In:   ${IN}"
echo "Out:  ${OUT}"
echo "Extra plot args: $*"

# Fail loudly rather than emit a directory of empty axes.  An absent GPU pass is
# the single most likely reason this job is run by mistake.
if ! compgen -G "${IN}/predictions_*.parquet" > /dev/null; then
    echo "ERROR: no ${IN}/predictions_*.parquet"
    echo "       Run the GPU pass first:"
    echo "       sbatch slurm/eval_predict.sh ${RUN} best.pt --out-dir ${IN}"
    exit 1
fi
ls -la "${IN}"/predictions_*.parquet
mkdir -p "${OUT}"

SPLITS="${SPLITS:-val oos oot oost}"          # §66: SPLITS="oos oot oost" for held-out-only figures
ECO_EXTRA="${NO_VAL:+--no-val}"               # NO_VAL=1 also drops val from the inventory figure

echo ""; echo "───────── metrics ─────────"
$RUN_PY eval_metrics.py --in-dir "${IN}" --out-dir "${IN}"

echo ""; echo "───────── predictions outside the physical range [0, 0.6] ─────────"
# §66: the model output is not clamped; report where it leaves [0, 0.6] before any figure is read.
$RUN_PY - "${IN}" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
d = Path(sys.argv[1])
for f in sorted(d.glob("predictions_*.parquet")):
    df = pd.read_parquet(f)
    bad = df[(df["pred"] < 0) | (df["pred"] > 0.6)]
    print(f"{f.stem}: {len(bad):,} of {len(df):,} rows out of range "
          f"({100 * len(bad) / max(len(df), 1):.3f}%)  <0: {(df['pred'] < 0).sum():,}  >0.6: {(df['pred'] > 0.6).sum():,}")
    if len(bad):
        key = "station_key" if "station_key" in bad else "station"
        top = (bad.groupby([key, "depth"], observed=True)["pred"]
               .agg(n="size", min="min", max="max").sort_values("n", ascending=False).head(8))
        print(top.to_string())
EOF

echo ""; echo "───────── scatter ─────────"
$RUN_PY plot_eval_scatter.py --in-dir "${IN}" --out-dir "${OUT}" "$@"

echo ""; echo "───────── ubRMSE by depth x split ─────────"
# --splits explicitly: the script's default omits val, and val is the split the
# run's own early stopping used, so it is the tie-back to the training log.
$RUN_PY plot_eval_boxplot.py --in-dir "${IN}" --out-dir "${OUT}" \
    --splits ${SPLITS} "$@"
for M in RMSE bias; do                 # §66: level error next to the dynamics error
    $RUN_PY plot_eval_boxplot.py --in-dir "${IN}" --out-dir "${OUT}" \
        --splits ${SPLITS} --metric ${M} "$@"
done

echo ""; echo "───────── ubRMSE by land cover / climate ─────────"
for BY in igbp_macro kg_macro elevation_band network; do
    echo "--- --by ${BY} ---"
    $RUN_PY plot_eval_ecosystem.py --in-dir "${IN}" --out-dir "${OUT}" \
        --by "${BY}" ${ECO_EXTRA} "$@"
done
echo "--- --by IGBP --min-stations 8 (fine classes) ---"
$RUN_PY plot_eval_ecosystem.py --in-dir "${IN}" --out-dir "${OUT}" \
    --by IGBP --min-stations 8 ${ECO_EXTRA} "$@"

echo ""; echo "───────── time series: 5 best / 5 worst ─────────"
$RUN_PY plot_eval_timeseries.py --in-dir "${IN}" \
    --out-dir "${OUT}/timeseries" --splits ${SPLITS} \
    --select extremes --n 5 "$@"

echo ""; echo "───────── time series: the six CR200-18-tile TxSON stations ─────────"
# Since §47 all six are OOS, each predicted at its own station pixel. The within-tile
# readout (six stations from ONE map) is slurm/eval_txson_figures.sh.
$RUN_PY plot_eval_timeseries.py --in-dir "${IN}" \
    --out-dir "${OUT}/timeseries" --splits oos \
    --select named --stations \
        ISMN_TxSON_CR200-18 ISMN_TxSON_CR200-25 ISMN_TxSON_CR1000-2 \
        ISMN_TxSON_CR200-24 ISMN_TxSON_CR200-15 ISMN_TxSON_CR200-6 "$@"

echo ""
echo "=== All done $(date) ==="
echo "Tables : ${IN}/"
echo "Figures: ${OUT}/"
