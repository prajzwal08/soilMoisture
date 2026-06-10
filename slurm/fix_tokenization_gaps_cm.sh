#!/bin/bash
#SBATCH --job-name=fix_token_gaps_cm
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/fix_token_gaps_cm_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Phase 2: re-run cloud masks for stations whose s2/dates changed in
# fix_tokenization_gaps_terramind.sh (s2_n_extra > 0, 17 stations).

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

conda run --no-capture-output -n sensei python fix_tokenization_gaps.py \
    --mode cm-force \
    --batch-size 16 \
    --device cuda \
    --execute

echo "Done."
