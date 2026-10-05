#!/bin/bash
#SBATCH --job-name=plot_depth_sweep
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=logs/lst_tmean_diff/plot_depth_sweep_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Depth sweep 1/2/3/6 layers: figure + ep10 table from the training logs (CPU only, ~1 min).
# Submit with --dependency=afterany:<1L job id> so the 1-layer run is included.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture


conda run -n terramind --no-capture-output python plot_depth_sweep.py "$@"
