#!/bin/bash
#SBATCH --job-name=arch_fig
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/arch_fig_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Architecture flowchart of the baseline (nolst_L3_wu200_20261005), CR200-18 imagery.
#   sbatch slurm/plot_architecture_s48.sh            prep (zarr -> assets.npz) + render
#   sbatch slurm/plot_architecture_s48.sh render     render only (layout iterations)
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
MODE="${1:-all}"
echo "=== arch_fig ${MODE}  job ${SLURM_JOB_ID}  $(date)  git $(git rev-parse --short HEAD) ==="
RUN_PY="conda run -n terramind --no-capture-output python"
if [[ "${MODE}" == render ]]; then
    $RUN_PY plot_architecture_s48.py --render
else
    $RUN_PY plot_architecture_s48.py --prep --render
fi
# vertical-input variant, same assets.npz, separate output file
$RUN_PY plot_architecture_s48_vertical.py
echo "=== done $(date) ==="
