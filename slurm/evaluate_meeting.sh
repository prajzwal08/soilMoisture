#!/bin/bash
#SBATCH --job-name=eval_meeting
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/eval_meeting_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail
echo "=== evaluate_meeting  job ${SLURM_JOB_ID}  started $(date) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"

cd /gpfs/work3/0/prjs1968/soilMoisture
source ~/.bashrc
conda activate terramind

RUN_NAME="${1:-baseline_huber}"
CKPT="${2:-best.pt}"
echo "Run: ${RUN_NAME}  Checkpoint: ${CKPT}"

# ── Step 1: Evaluate OOS / OOT / OOST ──────────────────────────────────────
echo ""
echo "=== Step 1: evaluate_splits.py ==="
python evaluate_splits.py \
    --run-name    "${RUN_NAME}" \
    --ckpt        "${CKPT}" \
    --batch-size  128 \
    --num-workers 8

# ── Step 2: Time-series plots (5 best + 5 worst OOS) ───────────────────────
echo ""
echo "=== Step 2: plot_timeseries_meeting.py ==="
python plot_timeseries_meeting.py \
    --run-name "${RUN_NAME}" \
    --ckpt     "${CKPT}" \
    --split    oos \
    --n        5

# ── Step 3: SM spatial maps — depth rows × date columns ─────────────────────
echo ""
echo "=== Step 3: plot_spatial_sm_meeting.py ==="
python plot_spatial_sm_meeting.py \
    --run-name "${RUN_NAME}" \
    --ckpt     "${CKPT}" \
    --n        5

echo ""
echo "=== GPU jobs done $(date) ==="
echo ""
echo "Run on login/CPU node after this completes:"
echo "  conda activate terramind && python plot_breakdown_meeting.py"
echo ""
echo "Outputs in: /gpfs/work3/0/prjs1968/soilMoisture/meeting_output/"
