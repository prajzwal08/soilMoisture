#!/bin/bash
#SBATCH --job-name=bf_backup
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bf_backup_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §50 phase 0b — copy exactly what the backfill will modify, per station, verify, lock.
#   raw   : satellite_zarr/{st}.zarr/{.zgroup,.zattrs,s2/}
#   tokens: zarr_tokens/{cat}/{st}/{.zmetadata,.zgroup,.zattrs,s2/,cm/,s2_l*.npy,s2_l*.json}
# Verified by file count AND bytes (never a sentinel). Usage: sbatch ... ST1 ST2 ...
set -uo pipefail
RAW=/projects/prjs1968/satellite_zarr
TOK=/projects/prjs1968/zarr_tokens
DST=${BACKFILL_BACKUP:-/gpfs/work3/0/prjs1968/backfill_backup/20260928}
# A second backup into the same (a-w) root: open ONLY the container dirs, never an existing
# station backup, then re-lock everything at the end.
[ -d "$DST" ] && chmod u+w "$DST" "$DST"/raw "$DST"/tokens "$DST"/tokens/* 2>/dev/null
mkdir -p "$DST/raw" "$DST/tokens"
fail=0
cnt() { find "$1" -type f | wc -l; }
byt() { du -sb --apparent-size "$1" | cut -f1; }
for st in "$@"; do
  cat=""
  for c in sm_only sm_and_flux flux_only; do [ -d "$TOK/$c/$st" ] && cat=$c && break; done
  [ -z "$cat" ] && { echo "!! $st: no token dir"; fail=1; continue; }
  mkdir -p "$DST/raw/$st.zarr" "$DST/tokens/$cat/$st"
  rsync -a "$RAW/$st.zarr/s2" "$DST/raw/$st.zarr/"
  for f in .zgroup .zattrs; do [ -e "$RAW/$st.zarr/$f" ] && cp -a "$RAW/$st.zarr/$f" "$DST/raw/$st.zarr/"; done
  rsync -a "$TOK/$cat/$st/s2" "$TOK/$cat/$st/cm" "$DST/tokens/$cat/$st/"
  for f in .zmetadata .zgroup .zattrs s2_l3.npy s2_l6.npy s2_l9.npy s2_l3.json s2_l6.json s2_l9.json; do
    [ -e "$TOK/$cat/$st/$f" ] && cp -a "$TOK/$cat/$st/$f" "$DST/tokens/$cat/$st/"
  done
  ok=1
  for pair in "$RAW/$st.zarr/s2:$DST/raw/$st.zarr/s2" "$TOK/$cat/$st/s2:$DST/tokens/$cat/$st/s2" \
              "$TOK/$cat/$st/cm:$DST/tokens/$cat/$st/cm"; do
    s=${pair%%:*}; d=${pair##*:}
    if [ "$(cnt $s)" != "$(cnt $d)" ] || [ "$(byt $s)" != "$(byt $d)" ]; then
      echo "!! $st: MISMATCH $s ($(cnt $s) files, $(byt $s) B) vs $d ($(cnt $d), $(byt $d))"; ok=0; fail=1
    fi
  done
  [ $ok = 1 ] && echo "ok  $st  ($cat)  raw_s2=$(cnt $DST/raw/$st.zarr/s2) files  tok_s2=$(cnt $DST/tokens/$cat/$st/s2)"
done
chmod -R a-w "$DST"
echo "backup at $DST  (chmod a-w)   status: $([ $fail = 0 ] && echo ALL VERIFIED || echo FAILURES — do not proceed)"
exit $fail
