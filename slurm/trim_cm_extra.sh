#!/bin/bash
#SBATCH --job-name=trim_cm_extra
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/trim_cm_extra_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Trim cm/masks + cm/dates to match s2/dates for the 920 stations where
# cm has extra (unused) date entries; archive the removed entries to
# /projects/prjs1968/data/excluded_stations/_cm_extra/{station}_cm_extra.npz
# AmeriFlux_CA-Cbo (cm subset of s2) is excluded by the script itself.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"

conda run --no-capture-output -n sensei python trim_cm_extra.py \
    --execute \
    --workers 32

echo "Done."
