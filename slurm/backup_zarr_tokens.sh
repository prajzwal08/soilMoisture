#!/bin/bash
#SBATCH --job-name=backup_zarr_tokens
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/backup_zarr_tokens_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Permanent backup of the active token zarr stores (993/993 stations,
# ~900GB on scratch) to /projects/prjs1968/zarr_tokens/. Training continues
# to read from /gpfs/scratch1/shared/pkhanal/zarr (scratch, fast I/O);
# this copy guards against scratch purges. Parallelized one rsync per
# station across 64 workers.

set -euo pipefail

SRC=/gpfs/scratch1/shared/pkhanal/zarr
DST=/projects/prjs1968/zarr_tokens

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "Source : $SRC"
echo "Dest   : $DST"

mkdir -p "$DST"

sync_station() {
    rel="$1"
    mkdir -p "$DST/$rel"
    rsync -a "$SRC/$rel/" "$DST/$rel/"
}
export -f sync_station
export SRC DST

find "$SRC" -mindepth 2 -maxdepth 2 -type d -printf '%P\n' \
    | xargs -P 64 -I{} bash -c 'sync_station "$@"' _ {}

echo "--- Verification ---"
src_complete=$(find "$SRC"/*/* -maxdepth 1 -name ".complete" 2>/dev/null | wc -l)
dst_complete=$(find "$DST"/*/* -maxdepth 1 -name ".complete" 2>/dev/null | wc -l)
echo "Source .complete markers: $src_complete"
echo "Dest   .complete markers: $dst_complete"

src_files=$(find "$SRC" -type f | wc -l)
dst_files=$(find "$DST" -type f | wc -l)
echo "Source file count: $src_files"
echo "Dest   file count: $dst_files"

du -sh "$SRC" "$DST"

if [ "$src_complete" -eq "$dst_complete" ] && [ "$src_files" -eq "$dst_files" ]; then
    echo "VERIFICATION PASSED"
else
    echo "VERIFICATION MISMATCH -- do not delete source data"
    exit 1
fi

echo "Done."
