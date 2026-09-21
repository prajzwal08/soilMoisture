#!/bin/bash
#SBATCH --job-name=eco_wpqc_arr
#SBATCH --partition=rome
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §36.24 -- the same well-phased QC pass, spread over 8 NODES.
#
# Why an array and not more --workers.  Measured on job 26798662: 16 workers gave 2.8
# reads/s and 64 workers gave 4.0 -- 4x the concurrency for 1.4x the throughput, with
# ZERO errors.  LP DAAC does not reject concurrent requests from one client, it queues
# them, so per-granule latency inflates to absorb whatever is thrown at it.  That ceiling
# is per client, so the way past it is more clients, not more threads inside one.
#
# 8 tasks x ~4/s should finish the remaining ~50k reads in roughly 30 min.
#
# Each task keeps its OWN reads CSV (…s{N}.csv); concurrent appends to one file would
# interleave partial rows.  The report step merges them.  Work is dealt round-robin so no
# task inherits a whole run of tile-edge stations.

set -eo pipefail
exec 2>&1

WORKERS="${1:-32}"
NSHARDS=8
EXTRA="${2:-}"

export PYTHONUNBUFFERED=1
ulimit -n 65536
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "=== eco_wpqc_arr job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$(hostname) $(date) ==="
[ -r "$HOME/.netrc" ] || { echo "FATAL: no ~/.netrc -- EDL auth would fail silently."; exit 1; }

conda run -n soilmoisture --no-capture-output python qc_wellphased_pairs.py \
    --out-tag wp --workers "$WORKERS" \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS" $EXTRA

echo "=== done $(date) ==="
