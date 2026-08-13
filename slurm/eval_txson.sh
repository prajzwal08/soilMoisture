#!/bin/bash
#SBATCH --job-name=eval_txson
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
# 40 TxSON stations, not the 774 of OOT: the dataset preloads L12 into RAM and
# 40 stations need ~8 GB.  120G is generous headroom, not a measured requirement.
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_txson_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -eo pipefail
exec 2>&1
echo "=== eval_txson  job ${SLURM_JOB_ID}  started $(date) ==="
echo "Node: $(hostname)"

cd /gpfs/work3/0/prjs1968/soilMoisture

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536          # memmap FD pressure

RUN="${1:-cls_depth_star_reg}"
CKPT="${2:-best.pt}"
shift 2 || true
RUN_PY="conda run -n terramind --no-capture-output python"
echo "Run: ${RUN}  Checkpoint: ${CKPT}  Extra args: $*"

# §26 -- multi-pixel readout.  One forward pass per (tile, day) yields a
# prediction at EVERY station that falls inside that tile's 224x224 map, not
# just the supervised centre pixel (112, 112).
#
# Smoke test first:
#   sbatch slurm/eval_txson.sh cls_depth_star_reg best.pt --pixel-tiles ISMN_TxSON_CR200-18
$RUN_PY eval_predict.py \
    --run-name    "${RUN}" \
    --ckpt        "${CKPT}" \
    --batch-size  128 \
    --num-workers 8 \
    --pixel-csv   csvs/txson_readouts.csv \
    --tag         txson \
    "$@"

echo ""
echo "=== All done $(date) ==="
echo "Output: /gpfs/work3/0/prjs1968/soilMoisture/eval_output/predictions_network_txson*.parquet"
