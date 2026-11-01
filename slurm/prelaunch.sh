#!/bin/bash
#SBATCH --job-name=prelaunch
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/prelaunch_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# Pre-launch gate for the §48 full run (2026-09-29 review), CPU only, after the full s48 cache:
#   0  byte-compile the training path
#   1  B5  preflight_s48.py      every train/val/oos station has its s48 cache + raw imagery;
#                                prints stored TWSA stamps (A1 premise)
#   2  B4  refit era5_stats18.json + driver_stats.json (train split, 2016-2022) — the 8 §50
#          stations are admitted now; old versions stay in git
#   3  A3  verify_splits_v2.py on ALL stations: asserts admitted == assigned (573 train)
set -uo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
RUN="conda run -n terramind --no-capture-output python -u"

$RUN -m py_compile train.py dataset.py model.py ckpt_utils.py eval_predict.py verify_splits_v2.py \
     preflight_s48.py compute_era5_stats.py compute_driver_stats.py || exit 1
echo "compile OK"

echo "=== 1  preflight_s48 ==="
$RUN preflight_s48.py || exit 1

echo "=== 2  stats refit ==="
sha() { sha256sum "$1" | cut -c1-16; }
echo "before: era5_stats18 $(sha csvs/era5_stats18.json)  driver_stats $(sha csvs/driver_stats.json)"
$RUN compute_era5_stats.py || exit 1
$RUN compute_driver_stats.py || exit 1
echo "after:  era5_stats18 $(sha csvs/era5_stats18.json)  driver_stats $(sha csvs/driver_stats.json)"

echo "=== 3  verify_splits_v2 (all stations) ==="
$RUN verify_splits_v2.py || exit 1
echo "=== prelaunch PASS ==="
