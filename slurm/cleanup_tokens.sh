#!/bin/bash
#SBATCH --job-name=cleanup_tokens
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/cleanup_tokens_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/cleanup_tokens_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
PYTHON=$(which python)

cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job     : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "CPUs    : $SLURM_CPUS_PER_TASK"
echo "Started : $(date)"
echo ""

$PYTHON cleanup_tokens.py --execute --workers 8

echo ""
echo "Finished : $(date)"
