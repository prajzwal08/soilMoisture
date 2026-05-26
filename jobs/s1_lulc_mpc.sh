#!/bin/bash
#SBATCH --job-name=s1_lulc_mpc
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/%x_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/%x_%j.err
#SBATCH --mail-user=p.khanal@utwente.nl
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# Phase 2 — submit only after S2 tokenization is complete and scratch is cleared.

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture

cd /gpfs/work3/0/prjs1968/soilMoisture
python download_s1_lulc_mpc.py
