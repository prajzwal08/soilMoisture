#!/bin/bash
#SBATCH --job-name=ens_fab
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/probe_ensemble_fabdem_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Can a 30 m spatially-resolved TWI be made reproducible by averaging over DEM-error
# realisations rather than over space? Region 10 is the one that failed Tier 3b
# (r(dTWI) = -0.163), 31 and 29 passed — so the probe spans the range rather than
# reading the answer off the easy cases.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"; date

conda run -n terramind --no-capture-output python -u probe_twi_ensemble.py \
    --region-id 10 --region-id 31 --region-id 29 \
    --n-members 8 --sigma 0.15 --source fabdem

date
