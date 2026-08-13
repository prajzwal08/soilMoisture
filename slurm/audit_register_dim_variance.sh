#!/bin/bash
#SBATCH --job-name=reg_dim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/reg_dim_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §27a.5 -- is the register direction a constant offset (dilution) or real between-site signal?
# Measures compression on raw tokens and on the 4 pooled vectors, all 993 stations.
#
#   sbatch slurm/audit_static_token_outliers.sh
#   sbatch slurm/audit_static_token_outliers.sh 

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate terramind

cd /gpfs/work3/0/prjs1968/soilMoisture

python audit_register_dim_variance.py --workers 64 "$@"
