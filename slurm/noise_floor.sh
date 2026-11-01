#!/bin/bash
#SBATCH --job-name=noise_floor
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/noise_floor_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §35.30 — measure the irreducible 160 m noise floor from §35.29's same-patch pairs.
# Two sensors inside one TerraMind token cannot be told apart by a patchwise model,
# so their disagreement bounds what any model at this resolution can achieve. Run
# BEFORE pre-registering the gate: every threshold in §35.30 is expressed relative
# to what this produces.

set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
conda run -n terramind --no-capture-output python measure_noise_floor.py
