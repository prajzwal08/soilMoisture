#!/bin/bash
#SBATCH --job-name=twi_bulk
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/twi_hand_bulk_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §32.4 — TWI/HAND for the 345 regions under 20e6 cells: 0.505e9 cells, 711 stations.
#
# Split from the 8 large regions (twi_hand_big.sh) so one slow region cannot hold up
# the other 345. Conditioning cost is what varies, not cell count: an inland 742x742
# region conditioned in 2 s while a coastal one of identical size took 30 s, because
# GLO-30's flat sea surface presents enormous numbers of pits.
#
# --tier3 runs the WhiteboxTools-vs-pyflwdir D8 cross-check on every region. It roughly
# doubles the accumulation cost but this is the pass where the sample is large enough
# to tell whether the 0.69 hillslope correlation seen on two regions is general.
#
# Resume-safe: a region with terrain_30m.tif already present is skipped.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
date

conda run -n terramind --no-capture-output python -u build_twi_hand.py \
    --max-cells 20000000 --workers 14 --tier3

date
echo "regions with terrain: $(ls -d /gpfs/work3/0/prjs1968/data/terrain/region_*/terrain_30m.tif 2>/dev/null | wc -l) / 353"
