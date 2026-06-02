#!/bin/bash
#SBATCH --job-name=sat_to_zarr
#SBATCH --array=0-99
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/sat_zarr_%A_%a.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/sat_zarr_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
PYTHON=$(which python)
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job     : $SLURM_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo ""

# 998 stations / 100 tasks = ~10 stations per task
# --workers 4 = 4 parallel stations within each task (5 CPUs: 4 workers + 1 main)
STATIONS_PER_TASK=10
START=$(( SLURM_ARRAY_TASK_ID * STATIONS_PER_TASK ))
END=$(( START + STATIONS_PER_TASK ))

$PYTHON convert_satellite_to_zarr.py \
    --start    $START  \
    --end      $END    \
    --workers  4       \
    --s1-dtype float16 \
    --execute

echo ""
echo "Finished : $(date)"
