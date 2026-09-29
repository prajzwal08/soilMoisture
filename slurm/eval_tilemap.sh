#!/bin/bash
#SBATCH --job-name=eval_tilemap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_tilemap_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#
# §48 full 20 m SM maps for the densest TxSON tiles (plot_tile_sm_map.py): one forward pass
# per date, cached as {tile}_map20m.npz, then the four figures.
#
#   sbatch slurm/eval_tilemap.sh <run> [--check-preds eval_output/<run>/predictions_oos.parquet]
#
# Tiles: CR200-18 (6 probes), CR1000-2 (6), CR200-3 (5), CR200-26 (4) -- all OOS under §47.

set -eo pipefail
exec 2>&1
echo "=== eval_tilemap  job ${SLURM_JOB_ID}  started $(date) ==="
echo "Node: $(hostname)"

cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
# Eval must read the ALL-YEARS GPFS s48cache, never a /dev/shm training copy.
unset S48_CACHE_ROOT

RUN="${1:?usage: sbatch slurm/eval_tilemap.sh <run-name> [extra plot_tile_sm_map args]}"
shift
TILES="${TILES:-ISMN_TxSON_CR200-18 ISMN_TxSON_CR1000-2 ISMN_TxSON_CR200-3 ISMN_TxSON_CR200-26}"
YEARS="${YEARS:-2019 2020}"
OUT="${OUT:-figures/eval/${RUN}/tile}"
RUN_PY="conda run -n terramind --no-capture-output python"
echo "Run: ${RUN}  Tiles: ${TILES}  Years: ${YEARS}  Out: ${OUT}  Extra: $*"

for T in ${TILES}; do
    echo ""; echo "───────── ${T} ─────────"
    $RUN_PY plot_tile_sm_map.py --run-name "${RUN}" --tile "${T}" \
        --years ${YEARS} --out-dir "${OUT}" "$@"
done

echo ""
echo "=== All done $(date) ==="
