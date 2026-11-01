#!/bin/bash
#SBATCH --job-name=s48cache
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=06:00:00
#SBATCH --output=logs/s48cache_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §48 — per-station read cache (pyramids, flat L12, aligned pixel cloud masks) from the
# token store. Read-only over the store; writes /gpfs/scratch1/shared/pkhanal/s48cache.
#
#   sbatch slurm/prepare_s48_cache.sh --limit 8     # smoke
#   sbatch slurm/prepare_s48_cache.sh               # all stations (resume-safe)

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python prepare_s48_cache.py "$@"
