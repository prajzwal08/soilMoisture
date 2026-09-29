#!/bin/bash
#SBATCH --job-name=eval_txson
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
# 40 TxSON tiles. s48 memory-maps the GPFS s48cache; 120G is headroom, not measured.
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
# Eval must read the ALL-YEARS GPFS s48cache, never a /dev/shm training copy.
unset S48_CACHE_ROOT

RUN="${1:?usage: sbatch slurm/eval_txson.sh <run-name> <ckpt> --out-dir eval_output/<run> [...]}"
CKPT="${2:?usage: sbatch slurm/eval_txson.sh <run-name> <ckpt> --out-dir eval_output/<run> [...]}"
shift 2
RUN_PY="conda run -n terramind --no-capture-output python"
echo "Run: ${RUN}  Checkpoint: ${CKPT}  Extra args: $*"

# §26 -- multi-pixel readout.  One forward pass per (tile, day) yields a prediction at
# EVERY station inside that tile's 112x112 (20 m) map, not just the supervised station
# pixel (56, 56).  csvs/txson_readouts.csv is on the 224 x 10 m grid; eval_predict.py
# gathers at row // 2, col // 2 (MAP_STRIDE).
#
# Smoke:
#   sbatch slurm/eval_txson.sh <run> best.pt --out-dir eval_output/<run> \
#          --pixel-tiles ISMN_TxSON_CR200-18
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
