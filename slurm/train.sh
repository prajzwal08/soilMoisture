#!/bin/bash
#SBATCH --job-name=sm_train
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=120:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
#SBATCH --requeue

set -euo pipefail
ulimit -n 65536   # memmap FDs: 612 stations × 3 keys × 8 workers ≈ 14k handles

cd /gpfs/work3/0/prjs1968/soilMoisture

# /dev/shm L12 preload: ~145 GB measured (shared across 4 ranks as one physical copy).
# Budget at val→train boundary (post CPU-pooling fix): ~324 GB → ~396 GB headroom vs 720G.
# Workers: 12 train + 4 val per rank (48+16=64 total), prefetch_factor=4 — set in train.py CONFIG.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=7200   # 2 hours; covers cold-GPFS val on first epoch

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

# Remove stale /dev/shm L12 caches from previous killed/preempted runs.
# Without this, old sm_l12_* dirs accumulate and double the ~145 GB SHM footprint → OOM.
echo "Cleaning stale SHM caches..."
rm -rf /dev/shm/sm_l12_* 2>/dev/null || true
echo "SHM clean."

# ---------------------------------------------------------------------------
# STORE-INTEGRITY PRE-FLIGHT (§35.6, added 2026-08-26)
#
# The 2026-08-26 scratch purge deleted .zarray headers and, for the small
# arrays, their chunks -- while .zmetadata survived. zarr.open_consolidated
# then returns fill_value with NO exception, so `soil` read as all zeros and
# nothing complained. Soil is §20.14's strongest tabular block, so a run in
# that state produces a plausible-looking number from zeroed input.
#
# Scratch is purged BY AGE, not quota (usage is 8.8% of an 8 TiB allowance),
# so this recurs. ~1 min against a 17-32 min preload and a multi-hour run.
echo "=== store-integrity pre-flight ==="
if ! conda run -n terramind --no-capture-output \
       python verify_zarr_store.py --root /gpfs/scratch1/shared/pkhanal/zarr \
              --out "csvs/verify_preflight_${SLURM_JOB_ID}.csv" --workers 64; then
  echo "STORE VERIFICATION FAILED -- refusing to train against a damaged store."
  echo "Restore with: sbatch slurm/restore_zarr.sh"
  exit 1
fi
echo "=== pre-flight passed ==="
echo

# --use-memmap is always correct on GPFS: the zarr l3/l6/l9 arrays are chunked [32,196,768]
# with blosc, so reading ONE anchor index pulls a 7.8 MB compressed chunk and decompresses
# 9.6 MB to use 294 KB — 32x read amplification, x3 layers. The flat .npy memmaps on scratch
# read exactly 294 KB with no decompression. Omitting this flag is what made 25141399 IO-bound
# (all dataloader workers in D-state, GPU util 28%).
conda run -n terramind --no-capture-output torchrun --nproc_per_node=4 train.py --use-memmap "$@"
