#!/bin/bash
#SBATCH --job-name=lst_deseason_train
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=logs/lst_tmean_diff/probe_lst_deseason_train_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# 1) Re-run the §52 LST level/pattern probe on ALL train+val+TxSON stations with Landsat lst22
#    (the 2026-09-30 run 27386018 was --max-stations 20 + TxSON = 60). Writes to the _full
#    folders so the 60-station outputs (B3 etc.) are kept.
# 2) Deseasonalised level/pattern vs SM plot, train split only.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
conda run -n terramind --no-capture-output python probe_lst_level_pattern.py --workers 64 --out-suffix _full
conda run -n terramind --no-capture-output python plot_lst_level_deseason_all.py --out-suffix _full --split train
