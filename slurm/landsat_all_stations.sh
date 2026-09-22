#!/bin/bash
#SBATCH --job-name=ls_st30
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-49%12
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# The 993-station Landsat C2 L2 ST pull -- QC'd at 30 m, pooled to 22x22 @ 100 m (§41.7 step 1).
#
# CONCURRENCY.  12 concurrent tasks x 6 workers = ~72 concurrent MPC readers, down from the 160
# of job 27015709.  That job's 25x401 / 28x403 / 25x429 were signing-endpoint pressure, not
# corrupt COGs (§41.6 misdiagnosed it) -- the failing URLs carried SAS tokens that had already
# expired.  Fewer readers plus a retryable 401/403/429 is the fix; a PC subscription key would
# be the other half but is not required at this rate.
#
# SHARDING IS BY md5(station_id), NOT BY POSITION.  Every task must compute the same partition
# no matter when it starts or what has already finished; positional slicing of a
# checkpoint-filtered list silently dropped 7,785 of 119,566 reads on array 26800268.
# NSHARDS is derived from SLURM_ARRAY_TASK_COUNT so it can never disagree with --array
# (ecostress_wp_qc_array.sh hardcoded NSHARDS=8 against --array=0-7 and lost work to the gap).
#
# DOWNLOAD ONLY, soilmoisture env. Never combined with analysis in one job -- terramind has no
# MPC stack and the two envs are not interchangeable.

set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
ulimit -n 65536

cd /gpfs/work3/0/prjs1968/soilMoisture

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "=== landsat st30  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  $(date) ==="
echo "shard ${SHARD} of ${NSHARDS}"

conda run -n soilmoisture --no-capture-output python download_landsat_st30.py \
    --shard "$SHARD" --nshards "$NSHARDS" --workers 6 "$@"

echo "=== done $(date) ==="
