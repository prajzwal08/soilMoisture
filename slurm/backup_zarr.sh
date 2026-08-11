#!/bin/bash
#SBATCH --job-name=backup_zarr
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/backup_zarr_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Back up the scratch token zarr (what dataset.py reads) to permanent /projects space,
# INCLUDING the *_l3/l6/l9.npy memmaps. Additive: NO --delete, so a partially-purged
# scratch can never propagate deletions to the backup. -W = whole-file (faster local copy).

set -euo pipefail
SRC=/gpfs/scratch1/shared/pkhanal/zarr/
DST=/projects/prjs1968/zarr_tokens/

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Backup: $SRC  ->  $DST   (additive, no --delete, whole-file)"
date
mkdir -p "$DST"

rsync -aW --info=stats2 "$SRC" "$DST"

echo "=== rsync done — verifying station counts ==="
for c in sm_only sm_and_flux flux_only; do
  s=$(ls "$SRC$c" 2>/dev/null | wc -l)
  b=$(ls "$DST$c" 2>/dev/null | wc -l)
  echo "  $c: scratch=$s  backup=$b"
done
echo "=== verifying memmaps copied (sample station) ==="
ls "$DST"sm_only/ISMN_Berlin_PSA7Ruebezahl/*.npy 2>/dev/null | wc -l
date
echo "Backup complete."
