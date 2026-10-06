#!/bin/bash
#SBATCH --job-name=plot_split_flowchart
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/plot_split_flowchart_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Paper split flowchart (Train / Val / OOS, OOT / OOST). CPU, seconds.
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python plot_split_flowchart.py "$@"
echo "=== done $(date) ==="
