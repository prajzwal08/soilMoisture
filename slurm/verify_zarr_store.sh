#!/bin/bash
#SBATCH --job-name=verify_zarr
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §35.1 step 2 / step 6 -- verify a per-station zarr store by ARRAY CONTENT.
#
# The 2026-08-26 purge deleted `.zarray` headers and, for the small auxiliary
# arrays, the chunks as well, while `.zmetadata` survived at each station root.
# zarr.open_consolidated therefore still reports every array as present and
# returns fill_value. restore_zarr.sh's `.complete`-counting check cannot see
# any of that. This script reads the arrays.
#
# Usage:
#   sbatch slurm/verify_zarr_store.sh backup    # BEFORE restoring  (step 2)
#   sbatch slurm/verify_zarr_store.sh live      # AFTER  restoring  (step 6)
#
# Exit code is non-zero if any station fails, so the job state itself is the
# verdict -- do not rely on reading the log.

set -uo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

WHICH="${1:-backup}"
case "$WHICH" in
  backup) ROOT=/projects/prjs1968/zarr_tokens ;;
  live)   ROOT=/gpfs/scratch1/shared/pkhanal/zarr ;;
  *)      ROOT="$WHICH" ;;                        # allow an explicit path
esac
OUT="csvs/verify_${WHICH}_${SLURM_JOB_ID}.csv"

echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "ROOT:    $ROOT"
echo "OUT:     $OUT"
echo

conda run -n terramind --no-capture-output \
  python verify_zarr_store.py --root "$ROOT" --out "$OUT" --workers 64
rc=$?

echo
echo "Finished: $(date)   exit=$rc"
exit $rc
