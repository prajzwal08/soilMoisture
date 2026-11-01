#!/bin/bash
#SBATCH --job-name=plot_tile_paper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/plot_tile_paper_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Publication tile figure (RGB + per-station time series), CPU only, reads existing all-day
# predictions of the frozen baseline. Default = FMI Sodankylä 6-station cluster; pass other
# plot_tile_paper.py arguments to override.
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
if [[ $# -eq 0 ]]; then
    set -- --tile-station ISMN_FMI_SOD101 \
        --stations ISMN_FMI_SOD101 ISMN_FMI_SOD102 ISMN_FMI_SOD103 \
                   ISMN_FMI_SOD071 ISMN_FMI_SOD072 ISMN_FMI_SOD073 \
        --name fmi_sodankyla
fi
conda run -n terramind --no-capture-output python plot_tile_paper.py "$@"
echo "=== done $(date) ==="
