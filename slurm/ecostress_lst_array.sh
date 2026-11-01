#!/bin/bash
#SBATCH --job-name=eco_lst_arr
#SBATCH --partition=rome
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §37 TIER 2 -- the LST pull.  ~78.9k granule-halves x 4 opens ~ 315k opens.
#
# 8 tasks x 64 workers -- MEASURED 2026-09-21 on a random cross-station sample with
# disjoint granules per config (§37.6): 16 -> 2.45, 32 -> 4.31, 64 -> 5.60 reads/s, cold
# and monotonic. That is a different curve from §36.23.5 (which flattened at 4.0) because
# this is a 4-open read. LP DAAC does
# NOT throttle -- 1 error in 51k opens -- it QUEUES, so latency inflates to absorb
# concurrency.  16 workers gave 2.8 reads/s, 64 gave 4.0.  The ceiling is per client, so
# the way past it is more clients, not more threads.  This contradicts §36.15c's
# "keep --workers 8 -- LP DAAC throttles above that", which was never measured.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
ulimit -n 65536
cd /gpfs/work3/0/prjs1968/soilMoisture

# NSHARDS derived, never hardcoded.  ecostress_wp_qc_array.sh:31 hardcodes NSHARDS=8 and
# it must agree with --array=0-7; a mismatch silently drops a fraction of the work with
# no error at all.
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
WORKERS="${1:-64}"
EXTRA="${2:-}"

echo "=== eco_lst_arr job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID/$NSHARDS host=$(hostname) $(date) ==="

[ -r "$HOME/.netrc" ] || { echo "FATAL: no ~/.netrc"; exit 1; }
grep -q urs.earthdata "$HOME/.netrc" || { echo "FATAL: ~/.netrc has no urs.earthdata"; exit 1; }
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
       https://data.lpdaac.earthdatacloud.nasa.gov/ || echo 000)
[ "$CODE" = "000" ] && { echo "FATAL: LP DAAC unreachable from this node"; exit 1; }

conda run -n soilmoisture --no-capture-output python read_ecostress_lst.py \
    --out-tag dtr --workers "$WORKERS" \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS" $EXTRA
echo "=== done $(date) ==="
