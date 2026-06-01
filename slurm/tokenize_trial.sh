#!/bin/bash
#SBATCH --job-name=tokenize_trial
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/tokenize_trial_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/tokenize_trial_%j.err

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
PYTHON=$(which python)
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job    : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"
echo ""

for BS in 8 16 32 64; do
    echo "========================================"
    echo "TRIAL — batch-size $BS  (5 stations, --compile)"
    echo "========================================"
    $PYTHON precompute_terramind.py \
        --trial      5   \
        --batch-size $BS \
        --num-workers 3  \
        --compile        \
        --start-idx  10
    echo ""
done

echo "Finished: $(date)"
