#!/bin/bash
#SBATCH --job-name=dtr_bands
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=00:40:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# Rebuild the DTR/SM join WITH the dt_hours covariate, then draw the pooled surface plot
# for the whole band and for the two extremes the ISS geometry actually separates.
set -eo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1
cd /gpfs/work3/0/prjs1968/soilMoisture
F=/gpfs/work3/0/prjs1968/soilMoisture/fig/dtr_txson
echo "=== dtr_bands job=$SLURM_JOB_ID $(date) ==="

conda run -n terramind --no-capture-output python -c "
import py_compile
for f in ('plot_dtr_vs_sm.py','plot_dtr_sm_pooled.py'): py_compile.compile(f, doraise=True)
print('syntax OK')"

echo '--- rebuild the join with dt_hours ---'
conda run -n terramind --no-capture-output \
    python plot_dtr_vs_sm.py --workers 64 --out-tag dtr_vs_sm_all

echo '--- dt_hours distribution ---'
conda run -n terramind --no-capture-output python -c "
import pandas as pd
d = pd.read_csv('csvs/ecostress_dtr_vs_sm_all.csv')
print(d['dt_hours'].describe())
for lo,hi,l in ((6,9,'6-9 h'),(9,12,'9-12'),(12,15,'12-15'),(15,19,'>=15 h')):
    s=d[(d.dt_hours>=lo)&(d.dt_hours<hi)]
    print(f'{l:8s} n={len(s):6d}  stations={s.station_id.nunique():4d}')"

for spec in "ALL::" "6-9h:6:9" "ge15h:15:19"; do
  tag=${spec%%:*}; rest=${spec#*:}; lo=${rest%%:*}; hi=${rest##*:}
  args=""; lbl=""
  [ -n "$lo" ] && args="$args --dt-lo $lo --dt-hi $hi" && lbl="   —   dt_hours in [$lo, $hi)"
  echo "--- $tag ---"
  conda run -n terramind --no-capture-output python plot_dtr_sm_pooled.py \
      $args --label "$lbl" --out "$F/dtr_sm_pooled_surface_${tag}.png"
done
echo "=== done $(date) ==="
