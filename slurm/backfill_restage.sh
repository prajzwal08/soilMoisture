#!/bin/bash
#SBATCH --job-name=bf_restage
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bf_restage_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50 phase 6: re-stage backfilled stations into CLEAN scratch dirs, rebuild their §48 cache.
# restage_store.py never deletes, so a stale s2 chunk set / npy would survive a plain rsync;
# the scratch copy (a copy — the /projects original is checked first) is removed and re-copied.
set -uo pipefail
TOK=/projects/prjs1968/zarr_tokens;   TOK_S=/gpfs/scratch1/shared/pkhanal/zarr
RAW=/projects/prjs1968/satellite_zarr; RAW_S=/gpfs/scratch1/shared/pkhanal/satellite_zarr
CACHE=/gpfs/scratch1/shared/pkhanal/s48cache
one() {
  st=$1
  cat=""; for c in sm_only sm_and_flux flux_only; do [ -d "$TOK/$c/$st" ] && cat=$c && break; done
  if [ -z "$cat" ] || [ ! -e "$TOK/$cat/$st/.complete" ] || [ ! -d "$RAW/$st.zarr/s2" ]; then
    echo "!! $st: source incomplete — scratch copy NOT touched"; return; fi
  chmod -R u+w "$TOK_S/$cat/$st" 2>/dev/null; rm -rf "$TOK_S/$cat/$st"
  rsync -a --no-compress "$TOK/$cat/$st/" "$TOK_S/$cat/$st/"
  chmod -R u+w "$RAW_S/$st.zarr" 2>/dev/null; rm -rf "$RAW_S/$st.zarr"
  rsync -a --no-compress "$RAW/$st.zarr/" "$RAW_S/$st.zarr/"
  rm -f "$CACHE/$cat/$st/pyr.npz"
  echo "restaged $st ($cat): tok $(find $TOK_S/$cat/$st -type f | wc -l) files, raw $(find $RAW_S/$st.zarr -type f | wc -l) files"
}
export -f one; export TOK TOK_S RAW RAW_S CACHE
printf "%s\n" "$@" | xargs -P 8 -I{} bash -c 'one {}'
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd /gpfs/work3/0/prjs1968/soilMoisture
python prepare_s48_cache.py --stations "$@" --force --workers 16
