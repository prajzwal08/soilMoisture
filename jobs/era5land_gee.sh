#!/bin/bash
#SBATCH --job-name=era5land_gee
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/%x_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/%x_%j.err
#SBATCH --mail-user=p.khanal@utwente.nl
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture

cd /gpfs/work3/0/prjs1968/soilMoisture
python download_era5land_gee.py
