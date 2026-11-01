#!/bin/bash
#SBATCH --job-name=nl_drv
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nl_drv_%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ktm.prajwalkhanal@gmail.com

# §68 drivers for the Overijssel grid (C1-C4). Everything is written under scratch nl/ only.
#   static : py_compile every §68 file + ERA5 pixel list (all NL tiles + controls) + SIF/soil/ERA5 on the 3 smoke
#            tiles (data_smoke) -> fanout to them. Exit 1 on any failure.
#   sif    : SIF from the kept global L2B files for all 688 tiles (no network)
#   era5   : ERA5-Land meteo + TWSA + radiation for every pixel pseudo-station (GEE)
#   fanout : copy pixel ERA5/TWSA to the 688 tiles (refuses any non-point pixel)
#   soil   : OpenLandMap soil patch for the 688 tiles
set -euo pipefail
cd /gpfs/work3/0/prjs1968/soilMoisture
export PYTHONUNBUFFERED=1
NL=/gpfs/scratch1/shared/pkhanal/nl
PIX=${NL}/era5pix
SIFKEEP=/projects/prjs1968/sif_tropomi_l2b
PY="conda run -n soilmoisture --no-capture-output python"
MODE=${1:?static|sif|era5|fanout|soil}
echo "Job ${SLURM_JOB_ID} mode ${MODE} on $(hostname)  $(date)"

sif()  { $PY download_sif_tropomi.py --data-root "$1" --from-local ${SIFKEEP} --end-date 2025-12-31 --workers 12; }
era5() { $PY download_era5land_gee.py --data-root "$1"; $PY download_era5_radiation.py --data-root "$1" --workers 6; }

case ${MODE} in
static)
    conda run -n terramind --no-capture-output python -m py_compile \
        dataset.py eval_predict.py assemble_nl_sm.py check_predict_all_days.py era5_pixel_fanout.py \
        convert_satellite_to_zarr.py retokenize_satellite_zarr.py compute_s1_dem_lulc_token_masks.py \
        create_token_zarr.py splice_era5_radiation.py prepare_s48_cache.py cloud_masking_inference.py \
        filter_cloudy_tiles.py download_sif_tropomi.py download_era5land_gee.py download_era5_radiation.py \
        download_soil_openlandmap.py download_s2_mpc.py download_s1_lulc_mpc.py
    echo "py_compile OK"
    $PY era5_pixel_fanout.py pixels --tiles csvs/nl_tiles.csv --out-root ${PIX} --extra-csv ${NL}/data/station_splits.csv
    # smoke pixel root = only the pixels of the 3 smoke tiles
    mkdir -p ${PIX}_smoke; cp ${PIX}/tile_to_pixel.csv ${PIX}_smoke/
    $PY - <<EOF
import pandas as pd
t2p = pd.read_csv("${PIX}/tile_to_pixel.csv")
smoke = pd.read_csv("${NL}/data_smoke/station_splits.csv")

pcol, tcol = "pixel_dir", "tile_dir"
want = set(t2p[t2p[tcol].isin(smoke.dir_name)][pcol])
px = pd.read_csv("${PIX}/station_splits.csv")
px[px.dir_name.isin(want)].to_csv("${PIX}_smoke/station_splits.csv", index=False)
print("smoke pixels:", sorted(want))
EOF
    sif ${NL}/data_smoke
    era5 ${PIX}_smoke
    $PY era5_pixel_fanout.py fanout --pixel-root ${PIX}_smoke --tile-csv ${NL}/data_smoke/station_splits.csv --data-root ${NL}/data_smoke
    $PY download_soil_openlandmap.py --data-root ${NL}/data_smoke
    find ${NL}/data_smoke/sm_only -maxdepth 2 -type d | sort
    for d in ${NL}/data_smoke/sm_only/*/; do echo "$(basename $d): $(ls -R $d | grep -c '\.nc$') nc, soil=$(ls $d/soil 2>/dev/null | tr '\n' ' ')"; done
    ;;
sif)    sif ${NL}/data ;;
era5)   era5 ${PIX}_ov ;;   # Overijssel + controls only (66 px of 567)
fanout) $PY era5_pixel_fanout.py fanout --pixel-root ${PIX}_ov --tile-csv ${NL}/data/station_splits.csv --data-root ${NL}/data ;;
soil)   $PY download_soil_openlandmap.py --data-root ${NL}/data ;;
esac
date
