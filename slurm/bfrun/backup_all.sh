#!/bin/bash
#SBATCH --job-name=bf_backupall
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bf_backupall_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50 0b for the full run: ONE job (concurrent jobs would fight over the a-w root). Per station
# copy + count/bytes verify in parallel, then lock only the new station copies.
set -uo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
LIST=$1; OK=$2
RAW=/projects/prjs1968/satellite_zarr; TOK=/projects/prjs1968/zarr_tokens
DST=/gpfs/work3/0/prjs1968/backfill_backup/20260928
chmod u+w "$DST" "$DST"/raw "$DST"/tokens "$DST"/tokens/* 2>/dev/null
mkdir -p "$DST/raw" "$DST/tokens"
one() {
  st=$1; cat=""
  for c in sm_only sm_and_flux flux_only; do [ -d "$TOK/$c/$st" ] && cat=$c && break; done
  [ -z "$cat" ] && { echo "!! $st no token dir"; return; }
  if [ -d "$DST/raw/$st.zarr/s2" ] && [ -d "$DST/tokens/$cat/$st/s2" ]; then echo "OK $st (already backed up)"; return; fi
  mkdir -p "$DST/raw/$st.zarr" "$DST/tokens/$cat/$st"
  rsync -a "$RAW/$st.zarr/s2" "$DST/raw/$st.zarr/"
  for f in .zgroup .zattrs; do [ -e "$RAW/$st.zarr/$f" ] && cp -a "$RAW/$st.zarr/$f" "$DST/raw/$st.zarr/"; done
  rsync -a "$TOK/$cat/$st/s2" "$TOK/$cat/$st/cm" "$DST/tokens/$cat/$st/"
  for f in .zmetadata .zgroup .zattrs s2_l3.npy s2_l6.npy s2_l9.npy s2_l3.json s2_l6.json s2_l9.json; do
    [ -e "$TOK/$cat/$st/$f" ] && cp -a "$TOK/$cat/$st/$f" "$DST/tokens/$cat/$st/"; done
  for pair in "$RAW/$st.zarr/s2:$DST/raw/$st.zarr/s2" "$TOK/$cat/$st/s2:$DST/tokens/$cat/$st/s2" "$TOK/$cat/$st/cm:$DST/tokens/$cat/$st/cm"; do
    s=${pair%%:*}; d=${pair##*:}
    if [ "$(find $s -type f | wc -l)" != "$(find $d -type f | wc -l)" ] ||        [ "$(du -sb --apparent-size $s | cut -f1)" != "$(du -sb --apparent-size $d | cut -f1)" ]; then
      echo "!! $st MISMATCH $s"; return; fi
  done
  chmod -R a-w "$DST/raw/$st.zarr" "$DST/tokens/$cat/$st"
  echo "OK $st"
}
export -f one; export RAW TOK DST
xargs -a "$LIST" -P 16 -I{} bash -c 'one {}' | tee /dev/stderr | awk '/^OK /{print $2}' > "$OK"
chmod a-w "$DST" "$DST"/raw "$DST"/tokens "$DST"/tokens/* 2>/dev/null
echo "backed up + verified: $(wc -l < $OK) of $(wc -l < $LIST)"
[ -s "$OK" ]
