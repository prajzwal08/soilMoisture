#!/bin/bash
#SBATCH --job-name=plot_data_overview
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/plot_data_overview_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Paper data-overview figure (map, split definition, record length, data points). CPU, Pool(64).
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python plot_data_overview.py "$@"
echo "=== done $(date) ==="
