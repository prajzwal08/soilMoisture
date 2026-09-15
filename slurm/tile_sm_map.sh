#!/bin/bash
#SBATCH --job-name=tile_sm_map
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/tile_sm_map_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#
# 160 m predicted-SM field over one tile, via token_sel='all' (K=196 instead of 1).
# One station's store only, so 120G is generous. No shm preload: that cache is narrowed
# to K=1 and cannot serve a 196-patch request.
#
#   sbatch slurm/tile_sm_map.sh ISMN_TxSON_CR200-18 pw_stage2a_L3

set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536

TILE="${1:-ISMN_TxSON_CR200-18}"
RUN="${2:-pw_stage2a_L3}"
shift 2 || true

echo "=== tile_sm_map  job ${SLURM_JOB_ID}  $(date) ==="
echo "Node: $(hostname)   Tile: ${TILE}   Run: ${RUN}"

conda run -n terramind --no-capture-output python plot_tile_sm_map.py \
    --tile "${TILE}" --run-name "${RUN}" "$@"

echo ""
echo "=== All done $(date) ==="
