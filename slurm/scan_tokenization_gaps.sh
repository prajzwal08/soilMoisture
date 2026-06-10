#!/bin/bash
#SBATCH --job-name=scan_token_gaps
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/scan_token_gaps_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Scan all 993 stations: compare satellite_zarr/{station}.zarr (raw imagery)
# against the production token zarr to find S2/S1 dates that exist in
# satellite_zarr but were never tokenized (same gap as AmeriFlux_US-xUN).

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"

conda run --no-capture-output -n sensei python scan_tokenization_gaps.py --workers 32

echo "Done."
