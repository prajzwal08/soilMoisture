#!/bin/bash
#SBATCH --job-name=mead_irr
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/mead_irr_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/mead_irr_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Runbook §69 steps 1+2: Mead (US-Ne1/Ne2/Ne3) daily SM from AmeriFlux BASE,
# then the observed irrigated-vs-rainfed contrast + one sample figure.
# Writes only under /gpfs/work3/0/prjs1968/data_mead/ and figures/mead/.

set -euo pipefail
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job     : $SLURM_JOB_ID"
echo "Started : $(date)"
echo "Commit  : $(git rev-parse --short HEAD)"
echo ""

$(which python) preprocessing_mead_base.py
echo ""
$(which python) analyse_mead_irrigation.py --year ${@:-2022}

echo ""
echo "Finished : $(date)"
