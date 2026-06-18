#!/bin/bash
#SBATCH --job-name=smoke_meeting
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/smoke_meeting_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail
echo "=== smoke_meeting  job ${SLURM_JOB_ID}  started $(date) ==="
echo "Node: $(hostname)"

cd /gpfs/work3/0/prjs1968/soilMoisture
source ~/.bashrc
conda activate terramind

RUN_NAME="baseline_huber"
CKPT="best.pt"

# ── Step 1: evaluate_splits (10 stations per split) ─────────────────────────
echo ""
echo "=== Step 1: evaluate_splits (smoke: 10 stations) ==="
python evaluate_splits.py \
    --run-name      "${RUN_NAME}" \
    --ckpt          "${CKPT}" \
    --batch-size    32 \
    --num-workers   4 \
    --max-stations  10

# ── Step 2: time series (2 best + 2 worst) ──────────────────────────────────
echo ""
echo "=== Step 2: plot_timeseries_meeting (n=2) ==="
python plot_timeseries_meeting.py \
    --run-name "${RUN_NAME}" \
    --ckpt     "${CKPT}" \
    --split    oos \
    --n        2

# ── Step 3: SM spatial maps — depth rows × date cols (2 stations) ───────────
echo ""
echo "=== Step 3: plot_spatial_sm_meeting (n=2) ==="
python plot_spatial_sm_meeting.py \
    --run-name "${RUN_NAME}" \
    --ckpt     "${CKPT}" \
    --n        2

# ── Step 4: CPU breakdown plots ─────────────────────────────────────────────
echo ""
echo "=== Step 4: plot_breakdown_meeting ==="
python plot_breakdown_meeting.py

echo ""
echo "=== Smoke test PASSED $(date) ==="
echo "Outputs:"
ls -lh meeting_output/*.csv 2>/dev/null || true
ls -lh meeting_output/timeseries/*.png 2>/dev/null || true
ls -lh meeting_output/breakdown/*.png 2>/dev/null || true
ls -lh meeting_output/satellite/*.png 2>/dev/null || true
