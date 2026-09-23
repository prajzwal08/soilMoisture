#!/bin/bash
#SBATCH --job-name=audit_ismn_flags
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/audit_ismn_flags_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/audit_ismn_flags_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Census of raw ISMN quality flags: which codes occur, how much data each
# costs, and whether C-codes ever co-occur with D-codes (which would make
# the startswith("D") whitelist in preprocessing_ISMN_soilMoisture.py:100
# leak physically-implausible values into the training targets).
#
# Requires the raw .stm archive.  Pass its path as the first argument:
#   sbatch slurm/audit_ismn_flags.sh /path/to/Data_separate_files_header_...

# ── env ───────────────────────────────────────────────────────────────────────
source /gpfs/home5/pkhanal/miniforge3/etc/profile.d/conda.sh
conda activate soilmoisture
PYTHON=$(which python)

cd /gpfs/work3/0/prjs1968/soilMoisture

ISMN_DIR=${1:-/home/khanalp/data/ISMNsoilMoisture/Data_separate_files_header_20140101_20251231_13107_18mx_20260208}

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "Archive  : $ISMN_DIR"
echo "Started  : $(date)"
echo ""

$PYTHON audit_ismn_flags.py \
    --ismn-dir "$ISMN_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --out csvs/ismn_flag_census.csv

echo ""
echo "Finished : $(date)"
