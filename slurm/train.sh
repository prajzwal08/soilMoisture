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

# --use-memmap is GONE (§35.22). The .npy memmaps existed solely to serve the ANCHOR L3/L6/L9
# reads for the U-Net skip connections; the patchwise model has no decoder, no anchor, and
# touches only L12. The flag no longer exists in train.py, so passing it now fails at argparse.
#
# The read amplification that justified it is also gone by a different route: the loader reads
# tokens_z[i, sel, :] -- 1.5 KB, one memmap page -- instead of the full (196,768) slab, so the
# epoch is compute-bound rather than IO-bound (job 26071036: gpu_util 27% -> 93-95%, §35.23).
conda run -n terramind --no-capture-output torchrun --nproc_per_node=4 train.py "$@"
