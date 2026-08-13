#!/bin/bash
#SBATCH --job-name=lst_redo
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python extract_lst_timeseries.py --extent tile --workers 16
conda run -n terramind --no-capture-output python analyze_lst_heterogeneity.py
conda run -n terramind --no-capture-output python plot_lst.py
