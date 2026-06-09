#!/bin/bash
#SBATCH --job-name=retok_c
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/retokenize_c_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Group C: 62 stations with S2 tokens in zarr but no .complete, no cloud masks.
# Only runs CloudSEN12 + fills ERA5/SIF/labels from scratch/data and level1_organised.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

conda run -n terramind python retokenize_satellite_zarr.py \
    --mode       cm-only \
    --batch-size 16 \
    --device     cuda \
    --execute

echo "Done. Total .complete:"
find /gpfs/scratch1/shared/pkhanal/zarr -name ".complete" | wc -l
