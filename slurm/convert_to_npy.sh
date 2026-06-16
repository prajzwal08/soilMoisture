#!/bin/bash
#SBATCH --job-name=convert_npy
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/convert_npy_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python convert_l369_to_npy.py --execute --workers 64
