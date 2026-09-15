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
OUT="figures/eval/${RUN}"
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

SPLITS="val oos oot oost"

echo ""; echo "───────── metrics ─────────"
$RUN_PY eval_metrics.py --in-dir "${IN}" --out-dir "${IN}"

echo ""; echo "───────── scatter ─────────"
$RUN_PY plot_eval_scatter.py --in-dir "${IN}" --out-dir "${OUT}" "$@"

echo ""; echo "───────── ubRMSE by depth x split ─────────"
# --splits explicitly: the script's default omits val, and val is the split the
# run's own early stopping used, so it is the tie-back to the training log.
$RUN_PY plot_eval_boxplot.py --in-dir "${IN}" --out-dir "${OUT}" \
    --splits ${SPLITS} "$@"

echo ""; echo "───────── ubRMSE by land cover / climate ─────────"
for BY in igbp_macro kg_macro; do
    echo "--- --by ${BY} ---"
    $RUN_PY plot_eval_ecosystem.py --in-dir "${IN}" --out-dir "${OUT}" \
        --by "${BY}" "$@"
done
echo "--- --by IGBP --min-stations 8 (fine classes) ---"
$RUN_PY plot_eval_ecosystem.py --in-dir "${IN}" --out-dir "${OUT}" \
    --by IGBP --min-stations 8 "$@"

echo ""; echo "───────── time series: 5 best / 5 worst ─────────"
$RUN_PY plot_eval_timeseries.py --in-dir "${IN}" \
    --out-dir "${OUT}/timeseries" --splits ${SPLITS} \
    --select extremes --n 5 "$@"

echo ""; echo "───────── time series: the six CR200-18-tile TxSON stations ─────────"
# All six sit in the val split with their own tiles, so each is predicted at its
# own token.  This is NOT the §26 within-tile six-token readout -- that needs the
# §28.9 token gather, which eval_predict.py rejects for patchwise checkpoints.
$RUN_PY plot_eval_timeseries.py --in-dir "${IN}" \
    --out-dir "${OUT}/timeseries" --splits val \
    --select named --stations \
        ISMN_TxSON_CR200-18 ISMN_TxSON_CR200-25 ISMN_TxSON_CR1000-2 \
        ISMN_TxSON_CR200-24 ISMN_TxSON_CR200-15 ISMN_TxSON_CR200-6

echo ""
echo "=== All done $(date) ==="
echo "Tables : ${IN}/"
echo "Figures: ${OUT}/"
