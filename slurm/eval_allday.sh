#!/bin/bash
#SBATCH --job-name=eval_allday
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_allday_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Every-day model output (incl. days without an observation) for ALL OOS / OOT / OOST stations,
# for TIME-SERIES PLOTS ONLY (user 2026-10-05: the prediction line must continue through gaps).
# Written to a SEPARATE folder (…/allday/); metrics and the ablation use ONLY the observed-day
# files in eval_output/<run>/ (eval_metrics.py globs that top folder only).
# Then every paper time-series figure is redrawn with the line from allday/.
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_CLIP=0,1
ulimit -n 65536
unset S48_CACHE_ROOT
RUN=baseline_selected_20261005
CKPT=/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/nolst_L3_wu200_20261005/best.pt
IN=eval_output/${RUN}
LINES=${IN}/allday
FIG=figures/eval/${RUN}_paper
RUN_PY="conda run -n terramind --no-capture-output python"
echo "=== eval_allday job ${SLURM_JOB_ID} $(date) ==="

$RUN_PY eval_predict.py --run-name "${RUN}" --ckpt "${CKPT}" --batch-size 128 --num-workers 8 \
    --splits oos oot oost --keep-unobserved --out-dir "${LINES}"

echo ""; echo "───────── redraw paper time series (line from allday, dots/metrics from observed days) ─────────"
rm -rf "${FIG}/timeseries" "${FIG}/timeseries_twente"
$RUN_PY plot_eval_timeseries.py --in-dir "${IN}" --out-dir "${FIG}/timeseries" --splits oos oot oost \
    --select extremes --n 5 --style paper --line-dir "${LINES}"
$RUN_PY plot_eval_timeseries.py --in-dir "${IN}" --out-dir "${FIG}/timeseries" --splits oos \
    --select named --stations ISMN_TxSON_CR200-18 ISMN_TxSON_CR200-25 ISMN_TxSON_CR1000-2 \
        ISMN_TxSON_CR200-24 ISMN_TxSON_CR200-15 ISMN_TxSON_CR200-6 --style paper --line-dir "${LINES}"
OUT=${FIG}/timeseries_twente bash slurm/eval_twente_ts.sh "${RUN}" --style paper --line-dir "${LINES}"
echo "=== done $(date) ==="
