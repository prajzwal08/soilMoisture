#!/bin/bash
#SBATCH --job-name=drydown
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/drydown_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §32.10 follow-up: does HAND modulate soil moisture DYNAMICS rather than level?
# The gate tested the mean and found nothing. This tests drydown time, memory,
# recession floor and storm response over the same colocated pairs, on their common
# observed dates so weather is held fixed.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"; date
conda run -n terramind --no-capture-output python -u probe_drydown_dynamics.py
date
