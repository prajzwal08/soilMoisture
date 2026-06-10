#!/bin/bash
#SBATCH --job-name=group_a_zarr
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/group_a_zarr_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Group A: 24 stations with .complete in zarr but missing S2 tokens.
# They have full .pt bundles in /projects/prjs1968/data/ so no GPU needed.
# --force overwrites the broken .complete; --data-root points to .pt bundles.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"

STATIONS="ISMN_SCAN_Moccasin,ISMN_SNOTEL_Aniak,ICOS_CH-Cha,ICOS_CH-Fru,ICOS_CH-Lae,ICOS_CZ-KrP,ICOS_CZ-wet,ICOS_FR-Aur,ICOS_RU-Fy2,AmeriFlux_CA-Cbo,AmeriFlux_CA-Mer,AmeriFlux_US-BZB,AmeriFlux_US-Bi2,AmeriFlux_US-DFC,AmeriFlux_US-Mpj,AmeriFlux_US-MtB,AmeriFlux_US-ONA,AmeriFlux_US-Rls,AmeriFlux_US-Rms,AmeriFlux_US-Rwf,AmeriFlux_US-Seg,AmeriFlux_US-Ses,AmeriFlux_US-xDJ,AmeriFlux_US-xSE"

conda run --no-capture-output -n terramind python create_token_zarr.py \
    --data-root /projects/prjs1968/data \
    --stations  "${STATIONS}" \
    --workers   64 \
    --force \
    --execute

echo "Done."
find /gpfs/scratch1/shared/pkhanal/zarr -name ".complete" | wc -l
