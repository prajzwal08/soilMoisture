#!/bin/bash
#SBATCH --job-name=effres_analysis
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/effres_analysis_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §70 effective resolution, CPU half: Tests 0/A/B/C + control on eval_output/effres_txson.
#   sbatch slurm/effres_analysis.sh [in_dir]
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
IN="${1:-eval_output/effres_txson}"
echo "=== effres_analysis  job ${SLURM_JOB_ID}  $(date)  git $(git rev-parse --short HEAD)  in ${IN} ==="
conda run -n terramind --no-capture-output python effres_analysis.py --in-dir "${IN}" --workers 64
echo "=== done $(date) ==="
