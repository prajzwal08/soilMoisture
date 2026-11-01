#!/bin/bash
#SBATCH --job-name=mead_swc
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/mead_swc_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/mead_swc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Download AmeriFlux BASE-BADM for the three Mead sites and report whether
# they carry SWC_* columns.  ameriflux_summary.csv says has_swc=False for
# Ne1/Ne2, but that was built from the FLUXNET product, not BASE.

source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job     : $SLURM_JOB_ID"
echo "Started : $(date)"
echo ""

$(which python) check_mead_swc.py --sites US-Ne1 US-Ne2 US-Ne3 --policy "${1:-CCBY4.0}"

echo ""
echo "Finished : $(date)"
