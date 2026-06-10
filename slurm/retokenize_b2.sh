#!/bin/bash
#SBATCH --job-name=retok_b2
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/retokenize_b2_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Group B Phase 2: CloudSEN12 cloud masks + tabular fill + .complete sentinel.
# Requires sensei conda env (senseiv2). Run after retokenize_b.sh (Phase 1).
# Also picks up any Group C stations not yet .complete.
# Submit with: sbatch --dependency=afterok:<PHASE1_JOBID> slurm/retokenize_b2.sh

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

conda run --no-capture-output -n sensei python retokenize_satellite_zarr.py \
    --mode       cm-fill \
    --batch-size 16 \
    --device     cuda \
    --execute

echo "Done Phase 2. Total .complete:"
find /gpfs/scratch1/shared/pkhanal/zarr -name ".complete" | wc -l
