#!/bin/bash
#SBATCH --job-name=audit_zarr_complete
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/audit_zarr_complete_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Comprehensive per-station zarr audit: modality existence, cloud-mask <-> S2
# date alignment, and per-year coverage within [start_date, end_date].

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"

conda run --no-capture-output -n sensei python audit_zarr_complete.py --workers 64

echo "Done."
