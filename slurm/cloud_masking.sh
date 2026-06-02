#!/bin/bash
#SBATCH --job-name=cloud_mask
#SBATCH --array=0-1027
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/cloud_mask_%A_%a.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/cloud_mask_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate sensei
PYTHON=$(which python)

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job       : $SLURM_JOB_ID  array task: $SLURM_ARRAY_TASK_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Started   : $(date)"
echo ""

MEMLOG=$(mktemp)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null >> "$MEMLOG"; sleep 10; done ) &
MONITOR_PID=$!

$PYTHON cloud_masking_inference.py \
    --start-idx  $SLURM_ARRAY_TASK_ID \
    --end-idx    $((SLURM_ARRAY_TASK_ID + 1)) \
    --batch-size 16 \
    --io-workers 3

kill $MONITOR_PID 2>/dev/null
PEAK=$(sort -n "$MEMLOG" 2>/dev/null | tail -1)
echo "Peak GPU memory : ${PEAK:-n/a}"
rm -f "$MEMLOG"
echo ""
echo "Finished  : $(date)"
