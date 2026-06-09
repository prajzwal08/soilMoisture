#!/bin/bash
#SBATCH --job-name=retok_b
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/retokenize_b_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Group B: 83 stations missing S2/S1/DEM/LULC tokens and cloud masks.
# Reads from /projects/prjs1968/satellite_zarr/, writes to ZARR_ROOT.
# TerraMind + SEnSeIv2 inference on GPU. Tabular fill from scratch/data.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

conda run -n terramind python retokenize_satellite_zarr.py \
    --mode       all \
    --batch-size 8 \
    --device     cuda \
    --execute

echo "Done. Total .complete:"
find /gpfs/scratch1/shared/pkhanal/zarr -name ".complete" | wc -l
