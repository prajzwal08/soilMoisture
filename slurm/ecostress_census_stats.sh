#!/bin/bash
#SBATCH --job-name=eco_stats
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §36.23 -- NO-PIXEL census across ALL in-range stations.  Counts only.
#
#   sbatch slurm/ecostress_census_stats.sh            # all 918
#   sbatch slurm/ecostress_census_stats.sh 40         # first 40, to sanity-check
#
# What this does and does not do:
#   DOES  CMR query, both dedupes, solar geometry, phase, and the full candidate-pair
#         inventory -- pairing needs no pixels, only UTC + solar date + phase.
#   DOES NOT open a single COG.  No EDL, no /vsicurl, no masks.  So `quality`,
#         `day_clear`, `night_clear`, `clear_frac` and `passed_qc` come back EMPTY,
#         meaning unknown -- never 0.
#
# --cpus-per-task=4, NOT 16.  Measured on census array 26742514 task 8: 71 core-hours
# billed against 0.1 s of CPU actually used.  This job is pure network latency; cores are
# the wrong currency and billing is on what you REQUEST.  Concurrency comes from
# --workers, which costs nothing.
#
# --mem=7G is load-bearing for that, not a guess.  Snellius bills max(cpus, mem/1792MiB)
# cores, so asking 12G would silently bill 7+ cores and a larger request 16 -- undoing
# the saving entirely.  4 cores x 1792 MiB = 7G is the ceiling that keeps the bill at 4.
#
# Output goes to *.stats.csv via --out-tag, so the headline CSVs from array job 26742514
# are untouched.  Rows are logged with status='dryrun', which load_done() deliberately
# excludes, so this cannot satisfy or block a later full census.

set -eo pipefail
exec 2>&1

N="${1:-0}"
WORKERS="${2:-32}"

export PYTHONUNBUFFERED=1
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "=== eco_stats  job=$SLURM_JOB_ID  node=$(hostname)  $(date) ==="
echo "workers=$WORKERS  sample=${N:-all}"

# CMR is a different server from the data host and IS rate-sensitive, so keep workers
# moderate here.  The measured no-throttle result applies to
# data.lpdaac.earthdatacloud.nasa.gov, which this run never touches.
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 https://cmr.earthdata.nasa.gov/ || echo 000)
echo "cmr.earthdata.nasa.gov -> HTTP $CODE"
[ "$CODE" = "000" ] && { echo "FATAL: no route to CMR from this node."; exit 1; }

ARGS="--dry-run --out-tag stats --workers $WORKERS"
[ "$N" -gt 0 ] && ARGS="$ARGS --sample $N"

echo "--- census_ecostress.py $ARGS ---"
conda run -n soilmoisture --no-capture-output python census_ecostress.py $ARGS

echo
echo "--- output ---"
ls -la csvs/ecostress_census_*.stats.csv 2>/dev/null || echo "(no stats CSVs written)"
echo "=== done $(date) ==="
