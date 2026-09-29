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
ulimit -n 65536   # kept as headroom; the L3/L6/L9 memmap FDs it was sized for are gone (§35.22)

cd /gpfs/work3/0/prjs1968/soilMoisture

# §48: no /dev/shm preload. History pyramids live in RAM (~3.6 GB per rank, CoW across
# workers) and the anchor L12 / pixel cloud masks are page-cached memmaps from
# /gpfs/scratch1/shared/pkhanal/s48cache (prepare_s48_cache.py).
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
# Review B5: the §48 read cache and raw imagery are not covered by the zarr check above; a
# missing one would only surface after every rank built its datasets.
if ! conda run -n terramind --no-capture-output python preflight_s48.py; then
  echo "§48 CACHE / RAW IMAGERY PRE-FLIGHT FAILED -- run prepare_s48_cache.py / restage first."
  exit 1
fi
echo "=== pre-flight passed ==="
echo

# §48 flags (all optional): --lambda-lst auto|FLOAT (0 = control), --fine-skips cnn|pool,
# --modality-dropout P. The dataset refuses to build without the §48 cache, and with the
# thermal term on it refuses a training set that has no Landsat target at all.
conda run -n terramind --no-capture-output torchrun --nproc_per_node=4 train.py "$@"
