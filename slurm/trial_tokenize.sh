#!/bin/bash
#SBATCH --job-name=tm_trial
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/trial_tokenize_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/trial_tokenize_%j.err

# ── env ───────────────────────────────────────────────────────────────────────
PYTHON=/gpfs/home5/pkhanal/miniforge3/envs/terramind/bin/python
# Use terramind env's PROJ to suppress version-mismatch warnings from rasterio
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job       : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started   : $(date)"
echo ""

# ── trial run ─────────────────────────────────────────────────────────────────
# --trial 3  : process 3 stations then print timing + GPU memory breakdown
# --batch-size 4 : conservative for 20 GB MIG slice
$PYTHON precompute_terramind.py \
    --trial 3 \
    --batch-size 4

echo ""
echo "Finished  : $(date)"
