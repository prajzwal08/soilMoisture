#!/bin/bash
#SBATCH --job-name=sm_token_zarr
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/scratch1/shared/pkhanal/logs/create_token_zarr_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail

cd /gpfs/work3/0/prjs1968/soilMoisture

ZARR_OUT="/gpfs/scratch1/shared/pkhanal/zarr"

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "Output : ${ZARR_OUT}"

conda run --no-capture-output -n terramind python /home/pkhanal/rechunk_zarr.py \
    --workers 64 \
    --execute

echo "Done. Zarr stores created:"
find "${ZARR_OUT}" -name ".complete" | wc -l
