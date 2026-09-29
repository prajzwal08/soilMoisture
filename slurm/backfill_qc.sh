#!/bin/bash
#SBATCH --job-name=bf_qc
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=10:00:00
#SBATCH --output=logs/bf_qc_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50 post-backfill QC, read-only on the stores. Four checks (see backfill_qc.py):
#   A offset (fresh re-download vs stored)  B repair diff  C coverage re-audit  D token drift
#   sbatch slurm/backfill_qc.sh                                   # all stations
#   sbatch slurm/backfill_qc.sh ISMN_LABFLUX_Nivolet ISMN_SNOTEL_Brighton ...   # smoke
# Every step runs even if an earlier one fails; the job exits non-zero if any check failed.
set -uo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
ST=(); [ $# -gt 0 ] && ST=(--stations "$@")
RC=0

conda activate terramind
python -m py_compile backfill_qc.py audit_s2_coverage.py || exit 1
echo "compile OK"

echo "=== A fetch (fresh re-download of the offset sample) ==="
conda activate soilmoisture
python backfill_qc.py --fetch --workers 16 "${ST[@]}" || RC=1

echo "=== A, B, D ==="
conda activate terramind
python backfill_qc.py --check --workers 64 "${ST[@]}" || RC=1

echo "=== C coverage re-audit ==="
if [ $# -eq 0 ]; then
  python audit_s2_coverage.py --dump-store --store-json csvs/_s2_store_dates_post_backfill.json || RC=1
  conda activate soilmoisture
  python audit_s2_coverage.py --store-json csvs/_s2_store_dates_post_backfill.json \
         --out csvs/s2_coverage_audit_post_backfill.csv --workers 16 || RC=1
  python backfill_qc.py --coverage || RC=1
else
  echo "skipped in smoke mode (needs the full station set)"
fi

echo "=== bf_qc exit $RC ==="
exit $RC
