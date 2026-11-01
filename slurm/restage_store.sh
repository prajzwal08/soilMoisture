#!/bin/bash
#SBATCH --job-name=restage
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/restage_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Re-stage a purged zarr store onto scratch. The SOURCE IS THE ONLY COPY and is never
# written: every rsync is src -> dst, no --delete. Merges into the surviving skeletons.
#
#   sbatch slurm/restage_store.sh tokens
#   sbatch --array=0-7 slurm/restage_store.sh imagery
#
# Shard i of the array takes stations [i::8], so any shard can be re-run alone and a
# re-run of a finished shard is a no-op (rsync skips same-size, same-mtime files).

set -euo pipefail
STORE="${1:-tokens}"
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python restage_store.py \
    --store "$STORE" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"
