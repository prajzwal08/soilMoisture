#!/bin/bash
#SBATCH --job-name=tokenize_s1only
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/tokenize_s1only_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/tokenize_s1only_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# ── env ───────────────────────────────────────────────────────────────────────
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
PYTHON=$(which python)
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job     : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started : $(date)"
echo ""

# ── S1-only tokenisation ──────────────────────────────────────────────────────
# Targets the 26 stations that have S1RTC scratch TIFs but no consolidated
# tokens — skipped in previous runs because the CSV filter only checked S2L2A.
# --s1-only skips S2L2A / DEM / LULC entirely.
# --csv-start-idx 0 with no --csv-end-idx processes all matching stations.
$PYTHON precompute_terramind.py \
    --batch-size      8  \
    --num-workers     3  \
    --compile             \
    --s1-only             \
    --csv-start-idx   0

echo ""
echo "Finished : $(date)"
