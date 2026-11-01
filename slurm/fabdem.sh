#!/bin/bash
#SBATCH --job-name=fabdem
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fabdem_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §32.3 — wide DEM per processing region: GLO-30 public COGs on AWS -> per-region
# Lambert Azimuthal Equal Area at exactly 30 m. 353 regions, ~0.98e9 cells, ~3.9 GB out.
#
# Not purely I/O-bound like the other download scripts: the warp of ~1e9 cells is real
# CPU, hence 16 cores rather than 4. Peak RAM is set by the largest region (region 0,
# 420 x 562 km, 262e6 cells) which holds a source mosaic plus the destination grid,
# about 3 GB; 64 G leaves ample headroom for 8 concurrent regions.
#
# Resume-safe: a region with dem_fabdem_30m.tif already present is skipped, so a
# requeue after a timeout costs nothing.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
date

conda run -n soilmoisture --no-capture-output python download_wide_dem.py --source fabdem --workers 8

date
echo "regions written: $(ls -d /gpfs/work3/0/prjs1968/data/terrain/region_*/dem_fabdem_30m.tif 2>/dev/null | wc -l) / 353"
du -sh /gpfs/work3/0/prjs1968/data/terrain 2>/dev/null || true
