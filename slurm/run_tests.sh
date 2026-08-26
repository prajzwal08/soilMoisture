#!/bin/bash
#SBATCH --job-name=sm_run_tests
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/run_tests_%j.out
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# The unit suite for the patchwise arm (§35.24).
#
#   test_patchwise_model.py     mask polarity both ways, depth ordering, the per-DAY driver
#                               cache, K=196 end to end, the collapse detectors, the loss
#                               invariants, the DOY encoding
#   test_patchwise_dataset.py   fail-closed loaders (cloud mask, S1 token mask, QC, DEM/LULC,
#                               dead soil channels, driver_stats), ERA5 staleness from real
#                               dates, S1 orbit tagging, the emitted-key contract
#
# NOTHING RUNS ON THE LOGIN NODE. That is the whole reason this file exists: the tests are
# written there and executed only here. Do not `python test_*.py` interactively.
#
# CPU-only by default. Anything needing a GPU is marked `@pytest.mark.gpu` and is deselected
# below; to run those too, resubmit against a GPU partition:
#
#   sbatch --partition=gpu_h100 --gpus=1 --export=ALL,RUN_GPU=1 slurm/run_tests.sh
#
# Fixtures are small synthetic zarr stores under $TMPDIR (scratch, node-local). The real token
# store is touched by exactly one test, which SKIPS cleanly when the store or
# csvs/driver_stats.json is absent — a purged scratch must not turn into a red suite.
#
# Submit:  sbatch slurm/run_tests.sh
# Result:  logs/run_tests_<jobid>.out   (also copied to logs/run_tests_<jobid>.txt)

set -uo pipefail

cd /gpfs/work3/0/prjs1968/soilMoisture
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
# Keep the CPU-only forward passes from oversubscribing the allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

JOB="${SLURM_JOB_ID:-manual}"
LOG="logs/run_tests_${JOB}.txt"
# pytest writes a few hundred MB of synthetic zarr into its basetemp; keep that off GPFS.
BASETEMP="${TMPDIR:-/scratch-local}/pytest_sm_${JOB}"
mkdir -p "$BASETEMP"

# CPU-only by default. RUN_GPU=1 also runs the @pytest.mark.gpu tests (which then
# skip themselves anyway if the node has no CUDA device).
if [ "${RUN_GPU:-0}" = "1" ]; then
    MARK_ARGS=()
else
    MARK_ARGS=(-m "not gpu")
fi

echo "Job ID  : ${JOB}"
echo "Node    : ${SLURM_NODELIST:-$(hostname)}"
echo "CPUs    : ${SLURM_CPUS_PER_TASK:-?}"
echo "Basetemp: ${BASETEMP}"
echo "Started : $(date)"
echo

# ── pytest is not in environment-terramind.yml; install it once if missing ────────
# The tests themselves need nothing but torch, numpy, pandas, scipy and zarr, all of which
# the terramind env already has. Only the runner is missing.
if ! conda run -n terramind python -c "import pytest" >/dev/null 2>&1; then
    echo "pytest not found in the terramind env — installing it"
    conda run -n terramind --no-capture-output python -m pip install --quiet pytest
    if ! conda run -n terramind python -c "import pytest" >/dev/null 2>&1; then
        echo
        echo "FAILED: pytest could not be installed into the terramind env."
        echo "Install it once from a compute node and resubmit:"
        echo "    conda run -n terramind python -m pip install pytest"
        echo "(and add pytest to environment-terramind.yml so it survives an env rebuild)."
        exit 2
    fi
fi
conda run -n terramind --no-capture-output python -c \
    "import pytest, torch, zarr, numpy, pandas, scipy; \
     print(f'pytest {pytest.__version__} | torch {torch.__version__} | zarr {zarr.__version__}')"

echo
echo "==================================================================="
echo " pytest ${MARK_ARGS[*]:-(all marks)}"
echo "==================================================================="

conda run -n terramind --no-capture-output \
    python -m pytest \
        test_patchwise_model.py \
        test_patchwise_dataset.py \
        -v -ra --tb=short --durations=15 \
        --basetemp="$BASETEMP" \
        -p no:cacheprovider \
        -W "ignore::pytest.PytestUnknownMarkWarning" \
        "${MARK_ARGS[@]}" \
    2>&1 | tee "$LOG"
STATUS="${PIPESTATUS[0]}"

echo
echo "Finished: $(date)"
rm -rf "$BASETEMP"

if [ "$STATUS" -eq 0 ]; then
    echo "ALL TESTS PASSED  (log: ${LOG})"
else
    echo "TESTS FAILED with exit ${STATUS}  (log: ${LOG})"
fi
exit "$STATUS"
