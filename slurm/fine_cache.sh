#!/bin/bash
#SBATCH --job-name=fine_cache
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fine_cache_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# 2026-09-29 data-loading fix: precompute the fine CNN inputs per scene, then prove the fast
# path is bit-identical (verify_fine_cache.py). Resume-safe.
#   sbatch slurm/fine_cache.sh --stations A B C ...     # smoke
#   sbatch slurm/fine_cache.sh                          # all stations
set -uo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
unset S48_CACHE_ROOT
RUN="conda run -n terramind --no-capture-output python -u"
$RUN -m py_compile dataset.py train.py prepare_fine_cache.py stage_shm.py verify_fine_cache.py || exit 1
echo "compile OK"
$RUN prepare_fine_cache.py "$@" || exit 1
if [ $# -gt 0 ]; then
  $RUN verify_fine_cache.py --stations 6 --dates 40 --stage-stations 6
else
  $RUN verify_fine_cache.py --stations 60 --dates 30 --stage-stations 12
fi
