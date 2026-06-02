#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --array=0-26
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/tokenize_%A_%a.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/tokenize_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# ── env ───────────────────────────────────────────────────────────────────────
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
PYTHON=$(which python)
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job       : $SLURM_JOB_ID  array task: $SLURM_ARRAY_TASK_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started   : $(date)"
echo ""

# ── tokenization (10 stations per array task, driven from station_splits.csv) ─
# Previous runs sorted scratch dirs (1858 total, 860 duplicates) causing 353
# active stations to fall past index 999 and never be processed.
# Now we drive explicitly from station_splits.csv, filtering to untokenized only.
START=$(( SLURM_ARRAY_TASK_ID * 10 ))
END=$(( START + 10 ))

$PYTHON precompute_terramind.py \
    --batch-size       8    \
    --num-workers      3    \
    --compile               \
    --csv-start-idx  $START \
    --csv-end-idx    $END

echo ""
echo "Finished  : $(date)"
