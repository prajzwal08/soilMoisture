#!/bin/bash
#SBATCH --job-name=audit_comprehensive
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/audit_comprehensive_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Single comprehensive per-station audit: modality existence, cm<->s2
# alignment, per-year coverage, token-vs-raw image alignment, tabular/static
# NaN sanity, and label continuity (>= 3 yr) for training readiness.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"

conda run --no-capture-output -n sensei python audit_comprehensive.py --workers 64

echo "Done."
