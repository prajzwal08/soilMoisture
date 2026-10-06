#!/bin/bash
#SBATCH --job-name=mead_pw_plot
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/mead_pw_plot_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/mead_pw_plot_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Runbook §69.7: patchwise vs 20 m model at Ne1 / Ne2 / south plots, figures. Args pass through.
set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "Job $SLURM_JOB_ID  commit $(git rev-parse --short HEAD)  $(date)"
python -m py_compile plot_mead_patchwise.py
$(which python) plot_mead_patchwise.py "$@"
