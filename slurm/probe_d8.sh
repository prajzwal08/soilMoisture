#!/bin/bash
#SBATCH --job-name=probe_d8
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe_d8_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §32.9.4 — settle the Tier-3 D8 disagreement before it blocks the sufficiency gate.
# Six regions spanning the range of conditioning burden and the range of Tier-3
# correlation, so the answer is not read off the easy cases.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
date

conda run -n terramind --no-capture-output python -u probe_d8_disagreement.py \
    --region-id 122 --region-id 187 --region-id 206 \
    --region-id 141 --region-id 198 --region-id 114

date
