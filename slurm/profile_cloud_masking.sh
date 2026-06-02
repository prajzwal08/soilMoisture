#!/bin/bash
#SBATCH --job-name=profile_cm
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/profile_cloud_masking_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/profile_cloud_masking_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate sensei

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job    : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"
echo ""

MEMLOG=$(mktemp)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null >> "$MEMLOG"; sleep 10; done ) &
MONITOR_PID=$!

python profile_cloud_masking.py \
    --station ISMN_SNOTEL_Truckee#2 \
    --n-tiles 128 \
    --batch-sizes 1 2 4 8 16 32 64

kill $MONITOR_PID 2>/dev/null
PEAK=$(sort -n "$MEMLOG" 2>/dev/null | tail -1)
echo "Peak GPU memory : ${PEAK:-n/a}"
rm -f "$MEMLOG"
echo ""
echo "Finished: $(date)"
