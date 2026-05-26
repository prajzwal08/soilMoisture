#!/bin/bash
#SBATCH --job-name=terramind
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-52
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/terramind_%A_%a.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/terramind_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=p.khanal@utwente.nl

# 1048 stations split into 53 chunks of 20 (array indices 0-52).
# Each task processes stations [TASK_ID*20 : TASK_ID*20 + 20].
# Resume-safe: skips any acquisition whose _L12.pt already exists.

module load 2023
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind

CHUNK=20
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))

cd /gpfs/work3/0/prjs1968/soilMoisture
python precompute_terramind.py \
    --start-idx  $START \
    --end-idx    $END   \
    --batch-size 16
