#!/bin/bash
#SBATCH --job-name=txson_figs
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/txson_figures_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#
# The CPU half of the §26 TxSON evaluation. Reads the multi-pixel readout written by
# slurm/eval_txson.sh; no GPU.
#
#   sbatch slurm/eval_txson.sh <run> best.pt --out-dir eval_output/<run>   # GPU
#   sbatch slurm/eval_txson_figures.sh <run>                                # this
#
#   eval_output/<run>/    txson_{timeseries.parquet, per_readout.csv, consistency.csv}
#   figures/eval/<run>/   network_timeseries/, tile_context/, txson_map_pred_vs_obs

set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1

RUN="${1:?usage: sbatch slurm/eval_txson_figures.sh <run-name>}"
IN="${IN:-eval_output/${RUN}}"
OUT="${OUT:-figures/eval/${RUN}}"
TILES="${TILES:-ISMN_TxSON_CR200-18 ISMN_TxSON_CR1000-2 ISMN_TxSON_CR200-3 ISMN_TxSON_CR200-26}"
RUN_PY="conda run -n terramind --no-capture-output python"
PRED="${IN}/predictions_network_txson.parquet"
TS="${IN}/txson_timeseries.parquet"

echo "=== txson_figures  job ${SLURM_JOB_ID}  started $(date) ==="
echo "Run: ${RUN}  In: ${IN}  Out: ${OUT}"
if [[ ! -f "${PRED}" ]]; then
    echo "ERROR: no ${PRED} -- run slurm/eval_txson.sh first"; exit 1
fi
mkdir -p "${OUT}"

echo ""; echo "───────── combine: join to observations, score every readout ─────────"
$RUN_PY combine_network.py --pred "${PRED}" --out-dir "${IN}" --tag txson

echo ""; echo "───────── per-tile time series ─────────"
$RUN_PY plot_network_timeseries.py --ts "${TS}" --tile ${TILES} \
    --out-dir "${OUT}/network_timeseries"

echo ""; echo "───────── tile context ─────────"
for T in ${TILES}; do
    $RUN_PY plot_tile_context.py --tile "${T}" --ts "${TS}" --out-dir "${OUT}/tile_context"
done

echo ""; echo "───────── network map (all 40, train as squares) ─────────"
$RUN_PY plot_txson_map.py --network-ts "${TS}" --out-dir "${OUT}"

echo ""
echo "=== All done $(date) ==="
