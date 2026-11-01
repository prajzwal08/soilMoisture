#!/bin/bash
#SBATCH --job-name=update_splits_v2
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/update_splits_v2_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §47 — bring sm_and_flux into training, demote tile-sharing / Dutch / thin-record
# stations to oos, top val back up from oos, and regenerate OOT/OOST eligibility from
# measured coverage. See update_splits_v2.py for the argument.
#
# Dry run by default. Pass --apply to write (backup lands at csvs/station_splits.csv.pre_s47).

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python update_splits_v2.py "$@"
