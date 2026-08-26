#!/bin/bash
#SBATCH --job-name=restore_zarr
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/restore_zarr_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Restore the consolidated per-station zarr stores (tokens + input features + labels)
# from the /projects backup back to scratch ZARR_ROOT after a scratch purge.
# 64-way PARALLEL: one rsync per station dir, 64 in flight (xargs -P 64). Staging
# nodes cap at 32 cores, but rsync is I/O-bound so 64 streams oversubscribe fine.
# Single-stream rsync is metadata-bound on ~993 small-file dirs;
# parallelising per-station is far faster. rsync -a is idempotent, so any stores
# already restored by a prior run are skipped instantly.

set -uo pipefail

export SRC=/projects/prjs1968/zarr_tokens
export DST=/gpfs/scratch1/shared/pkhanal/zarr

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Started: $(date)"
echo "SRC: $SRC  ->  DST: $DST   (64-way parallel)"

for c in sm_only sm_and_flux flux_only; do mkdir -p "$DST/$c"; done

# NOTE (§35.1, 2026-08-26): NO `--delete`, deliberately. This is a MERGE, so it
# can only fill gaps, never remove anything -- which is why it is preferred over
# wipe-and-restore. The backup is also chmod a-w, so a reversed direction fails
# closed rather than destroying it.
export FAILDIR="${FAILDIR:-/tmp/restore_zarr_fail_$SLURM_JOB_ID}"
mkdir -p "$FAILDIR"

# The rsync flags are INLINE, deliberately. They were briefly held in a bash
# array (RSYNC_OPTS) -- but bash cannot export arrays, and copy_one runs inside
# `bash -c` under xargs, where the array is unset. rsync then ran with NO flags:
# no recursion (`skipping directory .` for every station) and no --chmod. It
# looked like it was working. Job 26058649 was cancelled for this.
#
# `--chmod=u+rwX` is REQUIRED, not cosmetic. The backup is chmod a-w (§35.1
# step 1), and `rsync -a` preserves source permissions -- which would propagate
# r--r----- onto the live training store and leave it unwritable, breaking
# memmap regeneration and every later write. Verified with
# --dry-run --itemize-changes: without it ~120 files per station show a `p`
# (permission) change; with it that class disappears and 0 files are overwritten.

copy_one() {
  local rel="$1"                       # e.g. sm_only/ISMN_ARM_Omega
  mkdir -p "$DST/$rel"
  if ! rsync -a --chmod=u+rwX "$SRC/$rel/" "$DST/$rel/"; then
    # Previously this exit code was discarded inside xargs, so a failed station
    # was invisible and the job still reported success.
    echo "$rel rc=$?" >> "$FAILDIR/failures.txt"
    return 1
  fi
}
export -f copy_one

# List every station dir (category/station) and fan out 64 rsyncs at a time.
find "$SRC" -mindepth 2 -maxdepth 2 -type d -printf '%P\n' \
  | xargs -P 64 -I{} bash -c 'copy_one "$@"' _ {}

echo "All rsyncs dispatched/completed at $(date)"

n_fail=0
if [[ -s "$FAILDIR/failures.txt" ]]; then
  n_fail=$(wc -l < "$FAILDIR/failures.txt")
  echo "!!! $n_fail stations FAILED to rsync:"
  head -50 "$FAILDIR/failures.txt"
else
  echo "rsync: no station-level failures reported"
fi

# Verify by ARRAY CONTENT, not by sentinel files.
#
# The previous check counted `.complete` markers. Those are 0-byte files that
# rsync restores first AND that come from the backup, so it reported
# "sm_only: 842" even on a store whose every chunk was missing -- which is
# exactly the state the 2026-08-26 purge left behind. See §35.1.
echo
echo "=== verifying restored store by array content ==="
conda run -n terramind --no-capture-output \
  python /gpfs/work3/0/prjs1968/soilMoisture/verify_zarr_store.py \
    --root "$DST" --out "csvs/verify_live_${SLURM_JOB_ID}.csv" --workers 64
rc_verify=$?

echo "Finished: $(date)"
echo "rsync failures: $n_fail   verify exit: $rc_verify"
if (( n_fail > 0 )) || (( rc_verify != 0 )); then
  echo "RESTORE INCOMPLETE -- do NOT train against this store."
  exit 1
fi
echo "RESTORE VERIFIED."
