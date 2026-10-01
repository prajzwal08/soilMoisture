#!/bin/bash
#SBATCH --job-name=verify_s57
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_s57_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §57 verification, CPU only: byte-compile, verify_s48.py (the default "bands" mode must be
# unchanged), then verify_s57.py (the "indices" conversion against the raw store).
set -uo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
rc=0
echo "== py_compile"
conda run -n terramind --no-capture-output python -m py_compile \
    model.py train.py ckpt_utils.py verify_s57.py && echo "compile OK" || rc=1
echo "== verify_s48.py (bands mode)"
conda run -n terramind --no-capture-output python verify_s48.py || rc=1
echo "== verify_s57.py (indices mode)"
conda run -n terramind --no-capture-output python verify_s57.py || rc=1
echo "== overall rc=$rc"
exit $rc
