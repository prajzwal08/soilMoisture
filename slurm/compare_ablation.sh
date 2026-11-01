#!/bin/bash
#SBATCH --job-name=compare_ablation
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/compare_ablation_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §24.13 — pair ablation parquets against the baseline and emit per-depth deltas.
# CPU/pandas only; no model, no checkpoint. Arm-agnostic.
#
#   sbatch slurm/compare_ablation.sh eval_output_unet/predictions_oos_era5_cross_station_s0.parquet \
#          --base eval_output/predictions_oos.parquet --csv eval_output_unet/ablation_summary.csv

set -eo pipefail
exec 2>&1
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job     : $SLURM_JOB_ID"
echo "Args    : $*"
echo ""
python "${SCRIPT:-compare_ablation.py}" "$@"
echo ""
echo "=== done $(date) ==="
