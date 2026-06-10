#!/bin/bash
#SBATCH --job-name=fix_token_gaps_tm
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/fix_token_gaps_tm_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Phase 1: re-encode S2/S1 tokens for the 36 stations where satellite_zarr
# has dates that were never tokenized (csvs/tokenization_gap_scan.csv).

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

conda run --no-capture-output -n terramind python fix_tokenization_gaps.py \
    --mode terramind-force \
    --batch-size 8 \
    --device cuda \
    --execute

echo "Done."
