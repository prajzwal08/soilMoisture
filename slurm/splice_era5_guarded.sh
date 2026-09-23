#!/bin/bash
#SBATCH --job-name=era5_splice_guarded
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §43.12 -- unlock NARROWLY, splice, verify, RE-LOCK.
#
# zarr_tokens is chmod'd dr-xr-x--- as the data-safety lock on the only copy of the
# drivers.  This grants u+w to exactly two directories per station -- {station}/ (for
# .zmetadata, rewritten by zarr.consolidate_metadata) and {station}/era5/ (for the new
# values18 / vars18 arrays) -- and nothing else.  era5/values keeps its read-only bit
# and is never written.
#
# The re-lock runs from a TRAP, so it fires on success, on failure, and on Ctrl-C.
# It does NOT fire on SIGKILL (scancel -9, OOM); if that happens, re-lock by hand:
#     bash slurm/splice_era5_guarded.sh --relock-only

set -uo pipefail

REPO=/gpfs/work3/0/prjs1968/soilMoisture
ZARR=/projects/prjs1968/zarr_tokens
REPORT=$REPO/csvs/era5_splice_report.csv
DIRLIST=$REPO/csvs/.era5_unlocked_dirs.txt

build_dirlist() {
  : > "$DIRLIST"
  # folder names come from the dry-run report (col 1); no commas in folder names
  tail -n +2 "$REPORT" | cut -d, -f1 | while read -r f; do
    [ -z "$f" ] && continue
    for cat in sm_only sm_and_flux flux_only; do
      if [ -d "$ZARR/$cat/$f/era5" ]; then
        echo "$ZARR/$cat/$f"      >> "$DIRLIST"
        echo "$ZARR/$cat/$f/era5" >> "$DIRLIST"
        break
      fi
    done
  done
  echo "  dirs to unlock: $(wc -l < "$DIRLIST")"
}

relock() {
  echo ""
  echo "=== RE-LOCKING ==="
  if [ -s "$DIRLIST" ]; then
    xargs -a "$DIRLIST" -d '\n' -r chmod u-w 2>/dev/null
  fi
  local n_writable
  n_writable=$(find "$ZARR" -mindepth 2 -maxdepth 3 -type d -writable 2>/dev/null | wc -l)
  echo "  writable dirs remaining under $ZARR (depth 2-3): $n_writable"
  if [ "$n_writable" -eq 0 ]; then
    echo "  RE-LOCK VERIFIED -- store is read-only again."
  else
    echo "  *** WARNING: $n_writable dir(s) still writable -- inspect before leaving ***"
    find "$ZARR" -mindepth 2 -maxdepth 3 -type d -writable 2>/dev/null | head -20
  fi
}

if [ "${1:-}" = "--relock-only" ]; then
  relock
  exit 0
fi

trap relock EXIT INT TERM

echo "=== BEFORE ==="
stat -c '%A %n' "$ZARR/sm_only/ISMN_ARM_Anthony" "$ZARR/sm_only/ISMN_ARM_Anthony/era5"
echo "  writable dirs (depth 2-3): $(find "$ZARR" -mindepth 2 -maxdepth 3 -type d -writable 2>/dev/null | wc -l)"

echo ""
echo "=== NARROW UNLOCK ==="
build_dirlist
xargs -a "$DIRLIST" -d '\n' -r chmod u+w
stat -c '%A %n' "$ZARR/sm_only/ISMN_ARM_Anthony" "$ZARR/sm_only/ISMN_ARM_Anthony/era5" \
                "$ZARR/sm_only/ISMN_ARM_Anthony/era5/values"
echo "  (note era5/values itself stays read-only and is never written)"

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd "$REPO"

echo ""
echo "=== SPLICE --execute ==="
python splice_era5_radiation.py --workers 64 --execute
rc=$?
echo "splice exit: $rc"

if [ $rc -eq 0 ]; then
  echo ""
  echo "=== VERIFY ==="
  python verify_era5_18.py
  echo "verify exit: $?"
else
  echo "splice failed -- skipping verify"
fi

exit $rc
