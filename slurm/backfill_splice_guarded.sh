#!/bin/bash
#SBATCH --job-name=bf_splice
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bf_splice_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §50 phase 5b — unlock NARROWLY (only the listed stations), splice, RE-LOCK RECURSIVELY.
# The splice creates new subdirs (s2_merged, cm_merged) which a flat chmod would leave
# writable, hence chmod -R. Relock runs from a TRAP; after a SIGKILL run:
#     bash slurm/backfill_splice_guarded.sh --relock-only ST1 ST2 ...
set -uo pipefail
REPO=/gpfs/work3/0/prjs1968/soilMoisture
TOK=/projects/prjs1968/zarr_tokens
DIRS=()
for st in "${@/--relock-only/}"; do
  [ "$st" = "--repair" ] && continue
  [ "$st" = "--cleanup" ] && continue
  [ -z "$st" ] && continue
  for c in sm_only sm_and_flux flux_only; do [ -d "$TOK/$c/$st" ] && DIRS+=("$TOK/$c/$st") && break; done
done
relock() {
  echo "=== RE-LOCKING ${#DIRS[@]} station dirs (recursive) ==="
  for d in "${DIRS[@]}"; do chmod -R u-w "$d"; done
  n=$(for d in "${DIRS[@]}"; do find "$d" -writable; done | wc -l)
  echo "  writable paths remaining in those stations: $n"
  [ "$n" -eq 0 ] && echo "  RE-LOCK VERIFIED" || echo "  *** WARNING: still writable ***"
}
if [ "${1:-}" = "--relock-only" ]; then relock; exit 0; fi
trap relock EXIT INT TERM
echo "=== NARROW UNLOCK: ${#DIRS[@]} stations ==="
for d in "${DIRS[@]}"; do
  # Recursive, but ONLY the listed station dirs: cleanup deletes *_prebackfill subtrees, which
  # needs write on every directory inside them. The trap re-locks recursively.
  chmod -R u+w "$d"
done
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate terramind
cd "$REPO"
if [ "${1:-}" = "--repair" ]; then
  shift
  python backfill_repair.py --splice --stations "$@"
elif [ "${1:-}" = "--cleanup" ]; then
  shift
  python backfill_cleanup.py --stations-file <(printf "%s\n" "$@")
else
  python backfill_merge.py --splice --stations "$@"
fi
rc=$?
echo "splice exit: $rc"
exit $rc
