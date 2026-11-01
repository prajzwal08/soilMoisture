#!/bin/bash
#SBATCH --job-name=s73_paper_figs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=rome
#SBATCH --output=/gpfs/work3/0/prjs1968/soilMoisture/logs/s73_paper_figs_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com
# §73.10: the downstream paper figures of baseline_selected_20261005 (§66 jobs 27641852-27713769),
# redrawn for RUN. Needs eval_output/$RUN/{allday,per_station_*.csv} (eval_final + eval_figures + eval_allday).
#   RUN=nolst_L3_wu200_sd05_20261012 sbatch --export=ALL slurm/s73_paper_figs.sh
set -eo pipefail
exec 2>&1
cd /gpfs/work3/0/prjs1968/soilMoisture
export EVAL_CLIP=0,1
RUN="${RUN:?set RUN}"
PY="conda run -n terramind --no-capture-output python"
E=eval_output/$RUN; F=figures/eval/${RUN}_paper
DIRS="--pred-dir $E/allday --metrics-dir $E --out-dir $F/tiles"
TXSON="ISMN_TxSON_CR200-18 ISMN_TxSON_CR200-25 ISMN_TxSON_CR1000-2 ISMN_TxSON_CR200-24 ISMN_TxSON_CR200-15 ISMN_TxSON_CR200-6"
echo "=== s73_paper_figs  run $RUN  job $SLURM_JOB_ID  $(date) ==="
# tiles (27641852/59/60/63, 27642010/405)
$PY plot_tile_paper.py --tile-station ISMN_FMI_SOD101 --stations ISMN_FMI_SOD101 ISMN_FMI_SOD102 ISMN_FMI_SOD103 \
    ISMN_FMI_SOD071 ISMN_FMI_SOD072 ISMN_FMI_SOD073 --name fmi_sodankyla $DIRS
$PY plot_tile_paper.py --tile-station ISMN_TxSON_CR200-18 --stations $TXSON --name txson_cr200-18 $DIRS
$PY plot_tile_paper.py --tile-station ISMN_TxSON_CR200-18 --stations $TXSON --name txson_cr200-18_unbiased --unbiased $DIRS
$PY plot_tile_paper.py --tile-station ISMN_FMI_SOD011 --stations ISMN_FMI_SOD011 ISMN_FMI_SOD012 ISMN_FMI_SOD013 --name fmi_sod011 $DIRS
$PY plot_tile_paper.py --tile-station ISMN_SOILSCAPE_Lucky-Hills-1503 --stations ISMN_SOILSCAPE_Lucky-Hills-1503 \
    ISMN_SOILSCAPE_Lucky-Hills-1506 AmeriFlux_US-Whs --name luckyhills $DIRS
$PY plot_tile_paper.py --tile-station ISMN_ARM_Lamont-CF1 --stations ISMN_ARM_Lamont-CF1 ISMN_ARM_Lamont-CF2 AmeriFlux_US-ARM --name arm_lamont $DIRS
# bias groups (27646664)
$PY plot_bias_groups.py --per-station $E/per_station_oos.csv --out-dir $F
# slides (27705238, 27707235, 27713397/27713013/27713769)
RUN=$RUN bash slurm/slides_results_figs.sh
RUN=$RUN bash slurm/slides_landcover_fig.sh
$PY plot_eval_scatter.py --in-dir $E --out-dir $F --style paper --only scatter_pred_obs_density scatter_pred_obs_stations hist_pred_obs
echo "=== done $(date) ==="
