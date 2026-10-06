#!/bin/bash
#SBATCH --job-name=nl_s2fix
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nl_s2fix_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §68 re-fetch truncated S2 TIFs (cm task 0 of 27681280 hit one: T103_084 20160714, 5,590 B).
#   sbatch slurm/nl_s2_refetch.sh TILE DATE [DATE ...]
# The bad TIF is MOVED to nl/quarantine (not deleted); download_s2_mpc.py runs on a one-row CSV
# in its own data root (nl/data_fix_TILE) so the main per-task checkpoints are untouched, then the
# new TIF must open and read every band, or the job exits 1.
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
NL=/gpfs/scratch1/shared/pkhanal/nl
TILE=${1:?tile e.g. T103_084}; shift
DATES=("$@"); [[ ${#DATES[@]} -gt 0 ]] || { echo "need dates"; exit 1; }
D=NLGRID_${TILE}
FIX=${NL}/data_fix_${TILE}
mkdir -p ${FIX} ${NL}/quarantine/${D}/S2L2A
PY="conda run -n soilmoisture --no-capture-output python"

$PY - <<EOF
import pandas as pd
d = pd.read_csv("${NL}/data/station_splits.csv")
r = d[d.dir_name == "${D}"]
assert len(r) == 1, len(r)
r.to_csv("${FIX}/station_splits.csv", index=False)
EOF

for dt in "${DATES[@]}"; do
    f=${NL}/satellite/${D}/S2L2A/${dt}.tif
    [[ -f $f ]] && { ls -l $f; mv $f ${NL}/quarantine/${D}/S2L2A/; }
done

SOIL_DATA_ROOT=${FIX} SOIL_SCRATCH_DIR=${NL}/satellite SOIL_S2_HARMONISE=1 $PY download_s2_mpc.py

$PY - <<EOF
import rasterio, numpy as np, sys
bad = 0
for dt in "${DATES[*]}".split():
    f = "${NL}/satellite/${D}/S2L2A/%s.tif" % dt
    try:
        with rasterio.open(f) as s:
            a = s.read()
        print(f, a.shape, a.dtype, "min", int(a.min()), "max", int(a.max()))
    except Exception as e:
        print("FAIL", f, e); bad += 1
sys.exit(1 if bad else 0)
EOF
echo "refetch OK  $(date)"
