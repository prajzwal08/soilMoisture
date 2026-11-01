#!/bin/bash
#SBATCH --job-name=driver_stats
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/driver_stats_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §35.24 — recompute BOTH normalisation-constant files from the train split and
# the training years only:
#
#   csvs/era5_stats.json     <- compute_era5_stats.py     (19 ERA5-Land variables)
#   csvs/driver_stats.json   <- compute_driver_stats.py   (SIF, TWSA, 21 soil channels,
#                                                          per-depth label mean)
#
# Why they run together: both were re-scoped to 2016-2022 in the same session.
# The old era5_stats.json was fitted on each train station's whole record,
# INCLUDING the held-out 2023 OOT year, so any checkpoint trained against it
# carries a (weak, but real) test-year signal in its input scaling. Any run that
# uses the new driver_stats.json must therefore also use the new era5_stats.json,
# or the two halves of the input normalisation disagree about what "train" means.
#
# CPU-only work: no GPU, no torch kernels — the scripts import torch only because
# they import dataset.py for parity (dataset.py owns fill_soil_nans, _open_zarr
# and the zarr loaders, and these stats must be measured on exactly the arrays it
# feeds the model). Hence the terramind env, which is the one with torch + zarr 2.x
# + scipy; the soilmoisture env has no torch and no zarr and cannot import dataset.py.
#
# Both scripts use Pool(64), matching --cpus-per-task=64 above.
#
# Expected runtime: a few minutes each. The per-station reads are small
# (era5 ~2.5k x 19, sif/twsa vectors, labels, one 21x74x74 soil patch, the s2
# date list) over ~590 sm_only train stations, so this is GPFS-latency bound, not
# compute bound. The 1 h wall clock is deliberate slack for a cold/contended GPFS.
#
# Reads the token zarr store on scratch (/gpfs/scratch1/shared/pkhanal/zarr) —
# if scratch has been purged again, restore it before submitting (slurm/restore_zarr.sh).
#
# Submit:  sbatch slurm/driver_stats.sh
# Neither script writes anything except its own JSON, and both refuse to write at
# all if no station contributed — a failed run leaves the previous constants intact.

set -euo pipefail

cd /gpfs/work3/0/prjs1968/soilMoisture
mkdir -p logs csvs
export PYTHONUNBUFFERED=1

echo "Job ID : ${SLURM_JOB_ID:-interactive}"
echo "Node   : ${SLURM_NODELIST:-$(hostname)}"
echo "CPUs   : ${SLURM_CPUS_PER_TASK:-?}"
echo "Started: $(date)"
echo

echo "==================================================================="
echo " 1/2  compute_era5_stats.py  -> csvs/era5_stats.json"
echo "==================================================================="
conda run -n terramind --no-capture-output python compute_era5_stats.py

echo
echo "==================================================================="
echo " 2/2  compute_driver_stats.py -> csvs/driver_stats.json"
echo "==================================================================="
conda run -n terramind --no-capture-output python compute_driver_stats.py

echo
echo "Finished: $(date)"
ls -l csvs/era5_stats.json csvs/driver_stats.json
