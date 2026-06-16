#!/bin/bash
#SBATCH --job-name=audit_sm_range
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/audit_sm_range_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

conda run -n terramind --no-capture-output python audit_sm_range.py
