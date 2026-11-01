#!/bin/bash
#SBATCH --job-name=verify_s48
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_s48_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §48 verification, CPU only: byte-compile the changed files, then verify_s48.py
# (synthetic model checks + the real dataset on whatever stations are cached).

set -uo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
echo "== py_compile"
conda run -n terramind --no-capture-output python -m py_compile \
    model.py dataset.py train.py ckpt_utils.py prepare_s48_cache.py \
    consolidate_landsat_st.py verify_s48.py && echo "compile OK"
echo "== verify_s48.py"
conda run -n terramind --no-capture-output python verify_s48.py
