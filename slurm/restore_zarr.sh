#!/bin/bash
#SBATCH --job-name=restore_zarr
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/restore_zarr_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Restore the consolidated per-station zarr stores (tokens + input features + labels)
# from the permanent backup on /projects back to scratch ZARR_ROOT, which was purged.
# All three categories: sm_only (842), sm_and_flux (48), flux_only (103).
# Trailing slash on SRC copies its CONTENTS into DEST (merges the three category dirs).

set -euo pipefail

SRC=/projects/prjs1968/zarr_tokens/
DST=/gpfs/scratch1/shared/pkhanal/zarr/

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Started: $(date)"
echo "SRC: $SRC"
echo "DST: $DST"

mkdir -p "$DST"

# -a archive (perms/times/symlinks), --info=progress2 overall progress,
# --human-readable. No --delete: scratch only has empty skeletons, so a plain
# overlay fills in chunk data + .complete markers without touching anything else.
rsync -a --info=progress2 --human-readable "$SRC" "$DST"

echo "rsync exit: $?"
echo "Finished: $(date)"

# Verify: count restored .complete markers per category
for c in sm_only sm_and_flux flux_only; do
  n=$(find "$DST$c" -maxdepth 2 -name .complete 2>/dev/null | wc -l)
  echo "  $c: $n stores with .complete"
done
