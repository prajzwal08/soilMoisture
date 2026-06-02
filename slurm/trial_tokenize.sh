#!/bin/bash
#SBATCH --job-name=tm_trial
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/trial_tokenize_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/trial_tokenize_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
PYTHON=$(which python)
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job        : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Started    : $(date)"
echo ""

MEMLOG=$(mktemp)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null >> "$MEMLOG"; sleep 10; done ) &
MONITOR_PID=$!

$PYTHON precompute_terramind.py \
    --trial      5    \
    --batch-size 8    \
    --num-workers 3   \
    --compile

kill $MONITOR_PID 2>/dev/null
PEAK=$(sort -n "$MEMLOG" 2>/dev/null | tail -1)
echo "Peak GPU memory : ${PEAK:-n/a}"
rm -f "$MEMLOG"
echo ""
echo "Finished : $(date)"
