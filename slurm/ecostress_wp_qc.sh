#!/bin/bash
#SBATCH --job-name=eco_wpqc
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=10:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §36.24 -- IMAGE QC over the WELL-PHASED pairs only.
#
#   sbatch slurm/ecostress_wp_qc.sh              # all ~56k reads
#   sbatch slurm/ecostress_wp_qc.sh 400          # smoke test, first 400 reads
#   sbatch slurm/ecostress_wp_qc.sh 0 96         # all reads, 96 workers
#
# Reads the pair inventory the --dry-run census already wrote and opens COGs for exactly
# the granules named in it.  ~56k reads instead of the 988k a full census would do.
#
# --cpus-per-task=16 / --mem=28G is the SAME bill as 4 / 7G: a rome node is shared by at
# most 8 jobs, so 128/8 = 16 CPUs is the minimum billable slice and asking for 4 bought
# nothing while starving the TLS/decode work.  Measured (job 26798022): each COG open
# costs ~2.1 s of LP DAAC redirect latency whether or not the connection is reused, so
# open COUNT is the lever -- READ_VZA=False drops the 4th layer for a straight 25% cut.
# Workers stay moderate: at 96 on 4 cores throughput FELL to 1.9/s.

# Resume-safe: reads append to csvs/ecostress_wp_reads.wp.csv and are skipped on restart.

set -eo pipefail
exec 2>&1

LIMIT="${1:-0}"
WORKERS="${2:-48}"

export PYTHONUNBUFFERED=1
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "=== eco_wpqc  job=$SLURM_JOB_ID  node=$(hostname)  $(date) ==="
echo "limit=$LIMIT  workers=$WORKERS"

# EDL auth is netrc + cookie jar; without it every /vsicurl open 401s and the run
# finishes 'cleanly' with read_ok=0 everywhere, which looks exactly like total cloud.
[ -r "$HOME/.netrc" ] || { echo "FATAL: no ~/.netrc -- EDL auth would fail silently."; exit 1; }
grep -q urs.earthdata "$HOME/.netrc" || { echo "FATAL: ~/.netrc has no urs.earthdata entry."; exit 1; }

CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
       https://data.lpdaac.earthdatacloud.nasa.gov/ || echo 000)
echo "data.lpdaac.earthdatacloud.nasa.gov -> HTTP $CODE"
[ "$CODE" = "000" ] && { echo "FATAL: no route to LP DAAC from this node."; exit 1; }

ARGS="--out-tag wp --workers $WORKERS"
[ "$LIMIT" -gt 0 ] && ARGS="$ARGS --limit $LIMIT"

echo "--- qc_wellphased_pairs.py $ARGS ---"
conda run -n soilmoisture --no-capture-output python qc_wellphased_pairs.py $ARGS

echo
echo "--- output ---"
ls -la csvs/ecostress_wp_*.csv 2>/dev/null || echo "(no output CSVs)"
echo "=== done $(date) ==="
