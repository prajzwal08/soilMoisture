#!/bin/bash
#SBATCH --job-name=probe_stab
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe_stability_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# Tier 3b (§32.9.4) — replaces §32.5's Tier 3, which cannot pass as specified because
# it asks two D8 implementations to agree on a quantity D8 does not determine stably.
#
# Six multi-station regions so the between-station contrast dTWI — the quantity the
# §32.8 sufficiency gate actually regresses on — has pairs to measure. Region 10 has
# 13 stations, giving 78 pairs on its own.
#
# Each region is derived 4x (baseline + 3 noise levels), so this is ~4x the cost of a
# normal region; they were chosen small for that reason.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
date

conda run -n terramind --no-capture-output python -u probe_terrain_stability.py \
    --region-id 10 --region-id 38 --region-id 37 \
    --region-id 29 --region-id 31 --region-id 39 \
    --sigma 0.05 --sigma 0.2 --sigma 1.0

date
