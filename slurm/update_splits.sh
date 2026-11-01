#!/bin/bash
#SBATCH --job-name=update_splits
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/update_splits_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §35.29 — hold tile-sharing stations out of training, and sweep for cross-network
# duplicate records. See update_splits_tile_pairs.py for the argument.
#
# Pass --apply to write; without it the script is a dry run that only reports.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python update_splits_tile_pairs.py "$@"
