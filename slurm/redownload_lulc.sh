#!/bin/bash
#SBATCH --job-name=redownload_lulc
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/redownload_lulc_%j.out

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture

echo "Job     : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo ""

cd /gpfs/work3/0/prjs1968/soilMoisture
python download_s1_lulc_mpc.py

echo ""
echo "Finished : $(date)"
