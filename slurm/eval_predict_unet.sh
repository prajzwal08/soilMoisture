#!/bin/bash
#SBATCH --job-name=eval_predict_unet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_predict_unet_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §24.13 — modality ablation on the FROZEN U-NET arm (dataset_unet / model_unet /
# ckpt_utils_unet), which eval_predict.py stopped serving at fe0dc2c.
#
# Requires stage_ablation_tokens.py to have staged the stations to scratch first: the frozen
# arm reads dataset_unet.py:40 ZARR_ROOT = /gpfs/scratch1/shared/pkhanal/zarr, and an unstaged
# station is SILENT — _open_zarr returns None and the dataset builds 0 samples without raising.
#
#   sbatch slurm/eval_predict_unet.sh --ablate era5 --ablate-mode cross_station --seed 0
#   sbatch slurm/eval_predict_unet.sh --ablate dem  --ablate-mode cross_station --seed 0

set -eo pipefail
exec 2>&1
echo "=== eval_predict_unet  job ${SLURM_JOB_ID}  started $(date) ==="
echo "Node: $(hostname)"

cd /gpfs/work3/0/prjs1968/soilMoisture

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536

RUN="${RUN:-cls_depth_star_reg}"
CKPT="${CKPT:-best.pt}"
RUN_PY="conda run -n terramind --no-capture-output python"
echo "Run: ${RUN}  Checkpoint: ${CKPT}  Extra args: $*"
echo ""

# --splits oos only: the ablation is a PAIRED comparison against eval_output/predictions_oos.parquet
# on the ablation_oos subset. --no-shm is forced inside the script but passed for clarity.
$RUN_PY eval_predict_unet.py \
    --run-name     "${RUN}" \
    --ckpt         "${CKPT}" \
    --batch-size   128 \
    --num-workers  8 \
    --splits       oos \
    --station-flag ablation_oos \
    --no-shm \
    "$@"

echo ""
echo "=== done $(date) ==="
