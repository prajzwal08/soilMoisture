#!/bin/bash
#SBATCH --job-name=verify_lst_lvl
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/s48_lst_ablation/verify_lst_level_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §52 dT checks, CPU only (never on a GPU node: check 4 opens every station store).
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
conda run -n terramind --no-capture-output python verify_lst_level.py
