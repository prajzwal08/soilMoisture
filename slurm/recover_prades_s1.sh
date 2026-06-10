#!/bin/bash
#SBATCH --job-name=recover_prades_s1
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/recover_prades_s1_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# CRITICAL recovery: ISMN_SMOSMANIA_Prades-le-Lez has MISSING_s1 (both asc &
# desc) in the token zarr, but satellite_zarr has full s1_asc/s1_desc data
# (151 dates each, 2016-2018) — Phase 1 (terramind) tokenization for S1 was
# never written. Run Phase 1 for this station only (terramind env).

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

conda run --no-capture-output -n terramind python retokenize_satellite_zarr.py \
    --mode       terramind \
    --station    ISMN_SMOSMANIA_Prades-le-Lez \
    --batch-size 8 \
    --device     cuda \
    --execute

echo "Done."
