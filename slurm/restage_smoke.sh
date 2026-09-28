#!/bin/bash
#SBATCH --job-name=restage_smoke
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/restage_smoke_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Smoke: measure both sides for a handful of stations (--dry-run, copies nothing), then
# copy those same stations for real so the throughput number is measured, not guessed.

set -euo pipefail
STORE="${1:-tokens}"
N="${2:-8}"
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "===== DRY RUN — measuring $N units, copying nothing ====="
conda run -n terramind --no-capture-output python restage_store.py \
    --store "$STORE" --limit "$N" --workers 8 --dry-run
echo
echo "===== REAL COPY of the same $N units ====="
conda run -n terramind --no-capture-output python restage_store.py \
    --store "$STORE" --limit "$N" --workers 8
