#!/bin/bash
#SBATCH --job-name=lst_stats
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lst_stats_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §46 — sigma_ST and sigma_level over train stations and pre-OOT years only.
# Read-only over the Landsat archive; writes csvs/lst_stats.json + csvs/lst_tile_means.csv.
#
#   sbatch slurm/compute_lst_stats.sh --dry-run --limit 8    # smoke
#   sbatch slurm/compute_lst_stats.sh

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python compute_lst_stats.py "$@"
