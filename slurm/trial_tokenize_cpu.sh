#!/bin/bash
#SBATCH --job-name=tm_trial_cpu
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/gpfs/work3/0/prjs1968/data/logs/trial_tokenize_cpu_%j.out
#SBATCH --error=/gpfs/work3/0/prjs1968/data/logs/trial_tokenize_cpu_%j.err

# ── env ───────────────────────────────────────────────────────────────────────
PYTHON=/gpfs/home5/pkhanal/miniforge3/envs/terramind/bin/python
export PROJ_LIB=/gpfs/home5/pkhanal/miniforge3/envs/terramind/share/proj

cd /gpfs/work3/0/prjs1968/soilMoisture

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job       : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo ""

# ── trial run (CPU, 1 station) ────────────────────────────────────────────────
# --trial 1  : just 1 station to validate normalization fix (NaN count should be 0)
# --batch-size 1 : conservative for CPU
$PYTHON precompute_terramind.py \
    --trial 1 \
    --batch-size 1

echo ""
echo "Finished  : $(date)"
