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

copy_one() {
  local rel="$1"                       # e.g. sm_only/ISMN_ARM_Omega
  mkdir -p "$DST/$rel"
  rsync -a "$SRC/$rel/" "$DST/$rel/"
}
export -f copy_one

# List every station dir (category/station) and fan out 64 rsyncs at a time.
find "$SRC" -mindepth 2 -maxdepth 2 -type d -printf '%P\n' \
  | xargs -P 64 -I{} bash -c 'copy_one "$@"' _ {}

echo "All rsyncs dispatched/completed at $(date)"

# Verify: count restored .complete markers per category
for c in sm_only sm_and_flux flux_only; do
  n=$(find "$DST/$c" -maxdepth 2 -name .complete 2>/dev/null | wc -l)
  echo "  $c: $n stores with .complete"
done
echo "Finished: $(date)"
