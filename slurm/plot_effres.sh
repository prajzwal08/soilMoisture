#!/bin/bash
#SBATCH --job-name=plot_effres
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/plot_effres_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §70 effective-resolution figure (spectrum, variance retained, map vs observed variogram).
#   sbatch slurm/plot_effres.sh [in_dir] [out_stem]
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
IN="${1:-eval_output/effres_txson_smoke}"
OUT="${2:-figures/effres/effres_smoke}"
echo "=== plot_effres  job ${SLURM_JOB_ID}  $(date)  git $(git rev-parse --short HEAD)  in ${IN} ==="
conda run -n terramind --no-capture-output python plot_effres.py --in-dir "${IN}" --out "${OUT}"
echo "=== done $(date) ==="
