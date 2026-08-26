"""
Create one zarr store per station containing ALL modalities.

Output: /gpfs/scratch1/shared/pkhanal/zarr/{category}/{station}/
  s2/l12, l9, l6, l3   (N, 196, 768) fp16  chunks=(32,196,768)  zstd-3  [9.6 MB/chunk]
  s2/dates              (N,) str
  s1_asc/{layers,dates}  s1_desc/{layers,dates}   (if orbits exist)
  cm/masks              (N, 224, 224) uint8  chunks=(128,224,224) zstd-3  [6.4 MB/chunk]
  cm/dates              (N,) str
  dem                   (196, 768) fp16
  lulc                  (196, 768) fp16
  era5/values (N,19), date_ints (N,) int32, doys (N,) int32  [single chunk each]
  sif/values  (N,),   date_ints (N,) int32, doys (N,) int32
  twsa/lwe (N,), lwe_uncertainty (N,), date_ints (N,) int32, doys (N,) int32
  labels/sm (n_depths,n_days), qc (n_depths,n_days), depths (n_depths,), dates (n_days,)
                        qc: 0=observed, 1=gap-filled, 2=missing, 255=no QC source in the NetCDF
  .zmetadata  ← consolidated metadata for fast zarr.open_consolidated()
  .complete   ← sentinel written last; station skipped if already present

Chunk sizes follow the GPFS 10-100 MB rule: T=32 for tokens (9.6 MB), T=128 for masks
(6.4 MB), full-array for tabular (tiny, preloaded into RAM anyway). This reduces inodes
~13x vs chunks=(1,...) and cuts rolling-window reads from 60 to 2 per training sample.

Usage:
    python create_token_zarr.py [--start 0] [--end 993] [--workers 32] [--execute]
                                [--data-root /path/to/data]
                                [--stations ISMN_SNOTEL_Aniak,ICOS_CH-Fru]  # target specific stations
                                [--force]  # re-run even if .complete exists (Group A repair)
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
import numcodecs

DATA_ROOT  = Path("/gpfs/work3/0/prjs1968/data")
ZARR_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/zarr")   # SSD scratch — matches dataset.py ZARR_ROOT_FAST
LEVEL1_DIR = Path("/projects/prjs1968/level1_organised")   # authoritative SM labels
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")

ERA5_VARS = [
    "t2m_mean", "t2m_min", "t2m_max", "d2m_mean", "d2m_min", "d2m_max",
    "skt_mean", "skt_min", "skt_max", "u10_mean", "u10_min", "u10_max",
    "v10_mean", "v10_min", "v10_max", "sp_mean",  "sp_min",  "sp_max",
    "tp_sum",
]

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
LAYERS     = ["l12", "l9", "l6", "l3"]

# labels/qc encoding, matching preprocessing_ISMN_soilMoisture.py:
#   0 = directly observed, 1 = gap-filled from the month-day climatology, 2 = still missing.
# 255 is NEW (§35.24 audit item 4): "the source NetCDF carried no QC variable at all".
# write_labels used to default the whole array to ZEROS in that case, i.e. it asserted every
# day was a direct observation. The preprocessing pipeline gap-fills with a climatological
# mean, so those zeros told dataset.py to train on climatology as if it were ground truth —
# which is a station-mean predictor with a ground-truth badge, and no downstream check could
# tell the difference. The sentinel makes the absence explicit and dataset.py drops the
# station rather than trusting it.
QC_NO_SOURCE = 255

# Temporal chunk sizes tuned for GPFS (10-100 MB uncompressed per chunk):
#   T_TOKENS = 32  →  32 × 196 × 768 × 2 B fp16 = 9.6 MB  (rolling-window of 60 = 2 reads)
#   T_CM     = 128 → 128 × 224 × 224 × 1 B uint8 = 6.4 MB (full 60-acq window in 1-2 reads)
T_TOKENS = 32
T_CM     = 128


# ── helpers ───────────────────────────────────────────────────────────────────

def _first(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern)) if directory.exists() else []
    return files[0] if files else None


def _merge_nc(directory: Path, stem: str) -> "xr.Dataset | None":
    """Open and merge all {stem}_*.nc files in directory (handles both merged
    'stem_YYYYMMDD_YYYYMMDD.nc' and per-year 'stem_YYYY.nc' layouts).
    Returns a single sorted, deduplicated Dataset or None if no files found."""
    files = sorted(directory.glob(f"{stem}_*.nc")) if directory.exists() else []
    if not files:
        return None
    if len(files) == 1:
        return xr.open_dataset(files[0])
    ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal",
                           coords="minimal", compat="override")
    # Deduplicate time in case a merged file overlaps with per-year files
    _, idx = np.unique(ds["time"].values, return_index=True)
    return ds.isel(time=sorted(idx))


def _date_to_int(t) -> int:
    ts = pd.Timestamp(t)
    return ts.year * 10000 + ts.month * 100 + ts.day


# ── per-modality writers ──────────────────────────────────────────────────────

def write_tokens(root: zarr.Group, key: str, pt_dir: Path, orbit: str | None = None):
    """Write one set of token layers (S2 or one S1 orbit) into zarr group.

    Pre-flight checks ALL layers exist before writing anything — prevents partial
    zarr arrays if one layer file is missing or corrupted.
    """
    patterns = [
        f"*_{orbit}_{l.upper()}_*.pt" if orbit else f"*_{l.upper()}_*.pt"
        for l in LAYERS
    ]
    pt_files = [_first(pt_dir, p) for p in patterns]
    if any(p is None for p in pt_files):
        return False   # clean skip — no partial arrays written

    for layer, pt in zip(LAYERS, pt_files):
        d   = torch.load(pt, map_location="cpu", weights_only=False)
        arr = d["tokens"].numpy()                    # (N, 196, 768) fp16
        root.array(f"{key}/{layer}", arr,
                   chunks=(min(T_TOKENS, arr.shape[0]), arr.shape[1], arr.shape[2]),
                   dtype=arr.dtype, compressor=COMPRESSOR, overwrite=True)

    # Dates from L12 (all layers share the same dates)
    d_l12 = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    root.array(f"{key}/dates", np.array(d_l12["dates"], dtype="U8"),
               chunks=(len(d_l12["dates"]),), overwrite=True)
    if orbit:
        root[key].attrs["orbit"] = orbit
    return True


def write_cloud_mask(root: zarr.Group, cm_dir: Path):
    cm_pt = _first(cm_dir, "*_*.pt")
    if cm_pt is None:
        return
    d = torch.load(cm_pt, map_location="cpu", weights_only=False)
    masks = d["masks"].numpy()                       # (N, 224, 224) uint8
    root.array("cm/masks", masks,
               chunks=(min(T_CM, masks.shape[0]), masks.shape[1], masks.shape[2]),
               dtype=masks.dtype, compressor=COMPRESSOR, overwrite=True)
    root.array("cm/dates", np.array(d["dates"], dtype="U8"),
               chunks=(len(d["dates"]),), overwrite=True)


def write_static(root: zarr.Group, station_dir: Path):
    """Write DEM, LULC, and soil patch (all static, one value per station)."""
    dem_pt  = station_dir / "DEM"  / "dem_L12.pt"
    lulc_pt = station_dir / "LULC" / "lulc_L12.pt"
    for name, path in [("dem", dem_pt), ("lulc", lulc_pt)]:
        if path.exists():
            t = torch.load(path, map_location="cpu", weights_only=True)
            root.array(name, t.numpy(), compressor=COMPRESSOR, overwrite=True)

    soil_path = station_dir / "soil" / "soil_patch.tif"
    if soil_path.exists():
        import rasterio
        with rasterio.open(soil_path) as src:
            patch   = src.read().astype(np.float32)   # (21, 74, 74)
            nodata  = src.nodata
        if nodata is not None:
            patch[patch == nodata] = np.nan
        root.array("soil", patch, compressor=COMPRESSOR, overwrite=True)


def write_era5(root: zarr.Group, era5_dir: Path):
    ds = _merge_nc(era5_dir, "meteo")
    if ds is None:
        return
    with ds:
        vals  = np.stack([ds[v].values for v in ERA5_VARS], axis=-1).astype(np.float32)
        times = pd.DatetimeIndex(ds["time"].values)
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.day_of_year  for t in times], dtype=np.int32)
    root.array("era5/values",    vals,       compressor=COMPRESSOR, overwrite=True)
    root.array("era5/date_ints", date_ints,  overwrite=True)
    root.array("era5/doys",      doys,       overwrite=True)


def write_sif(root: zarr.Group, sif_dir: Path):
    ds = _merge_nc(sif_dir, "sif")
    if ds is None:
        return
    with ds:
        vals  = ds["sif"].values.astype(np.float32)
        times = pd.to_datetime(ds["time"].values)
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.timetuple().tm_yday for t in times], dtype=np.int32)
    root.array("sif/values",    vals,      compressor=COMPRESSOR, overwrite=True)
    root.array("sif/date_ints", date_ints, overwrite=True)
    root.array("sif/doys",      doys,      overwrite=True)


def write_twsa(root: zarr.Group, twsa_dir: Path):
    ds = _merge_nc(twsa_dir, "twsa")
    if ds is None:
        return
    with ds:
        lwe   = ds["lwe"].values.astype(np.float32)
        unc   = ds["lwe_uncertainty"].values.astype(np.float32) if "lwe_uncertainty" in ds else np.zeros_like(lwe)
        times = pd.to_datetime(ds["time"].values)
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.timetuple().tm_yday for t in times], dtype=np.int32)
    root.array("twsa/lwe",             lwe,       compressor=COMPRESSOR, overwrite=True)
    root.array("twsa/lwe_uncertainty", unc,       compressor=COMPRESSOR, overwrite=True)
    root.array("twsa/date_ints",       date_ints, overwrite=True)
    root.array("twsa/doys",            doys,      overwrite=True)


def write_flux_labels(root: zarr.Group, station_dir: Path,
                      category: str = None, dir_name: str = None):
    """Write LE_F_MDS flux labels.

    Primary source: station_dir / labels.nc (old data_root layout).
    Fallback: LEVEL1_DIR / category / dir_name.nc (new level1_organised layout).
    """
    nc = station_dir / "labels.nc"
    if not nc.exists() and category and dir_name:
        nc = LEVEL1_DIR / category / f"{dir_name}.nc"
    if not nc.exists():
        return
    with xr.open_dataset(nc) as ds:
        if "LE_F_MDS" not in ds:
            return
        le    = ds["LE_F_MDS"].values.astype(np.float32)
        qc    = ds["LE_F_MDS_QC"].values.astype(np.float32) if "LE_F_MDS_QC" in ds else np.zeros_like(le, dtype=np.float32)
        tc    = "date_time" if "date_time" in ds else "time"
        times = pd.DatetimeIndex(ds[tc].values)
    dates = np.array([t.strftime("%Y%m%d") for t in times], dtype="U8")
    root.array("labels/le",         le,    compressor=COMPRESSOR, overwrite=True)
    root.array("labels/le_qc",      qc,    compressor=COMPRESSOR, overwrite=True)
    root.array("labels/dates_flux", dates,                        overwrite=True)


def write_labels(root: zarr.Group, dir_name: str, category: str,
                 start_date: str, end_date: str):
    """Write SM labels from level1_organised/{category}/{dir_name}.nc.

    start_date / end_date : clip window from station_splits.csv (YYYY-MM-DD).
    """
    nc = LEVEL1_DIR / category / f"{dir_name}.nc"
    if not nc.exists():
        return

    ds         = xr.open_dataset(nc)
    time_coord = "date_time" if "date_time" in ds else "time"
    times_all  = pd.DatetimeIndex(ds[time_coord].values)

    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)
    mask    = (times_all >= t_start) & (times_all <= t_end)

    if mask.sum() == 0:
        ds.close()
        return

    if "soil_moisture" not in ds:
        ds.close()
        return   # flux_only station — SM labels not present, handled by write_flux_labels

    # Normalise to (depth, date_time) — sm_only: (depth, date_time),
    # sm_and_flux: (date_time, depth)
    sm_da = ds["soil_moisture"]
    if sm_da.dims[0] == time_coord:
        sm_da = sm_da.transpose("depth", time_coord)
    sm     = sm_da.values.astype(np.float32)[:, mask]
    depths = np.array([str(d) for d in ds["depth"].values], dtype="U20")
    # Fail closed: the sentinel, NOT zeros. See the QC_NO_SOURCE note at the top of the file —
    # defaulting to 0 ("observed") is the difference between training on measurements and
    # training on a month-day climatology while believing it is measurements.
    qc     = np.full(sm.shape, QC_NO_SOURCE, dtype=np.uint8)
    for qc_var in ("soil_moisture_qc", "quality_flag"):
        if qc_var in ds:
            qc_da = ds[qc_var]
            if qc_da.dims[0] == time_coord:
                qc_da = qc_da.transpose("depth", time_coord)
            qc = qc_da.values.astype(np.uint8)[:, mask]
            break
    ds.close()

    times = times_all[mask]
    dates = np.array([t.strftime("%Y%m%d") for t in times], dtype="U8")
    root.array("labels/sm",     sm,     compressor=COMPRESSOR, overwrite=True)
    root.array("labels/qc",     qc,     compressor=COMPRESSOR, overwrite=True)
    root.array("labels/depths", depths, overwrite=True)
    root.array("labels/dates",  dates,  overwrite=True)


# ── per-station conversion ────────────────────────────────────────────────────

def convert_station(args: tuple) -> str:
    station_dir, category, start_date, end_date, execute, force = args
    station_dir = Path(station_dir)
    out_dir     = ZARR_ROOT / category / station_dir.name
    sentinel    = out_dir / ".complete"

    if sentinel.exists() and not force:
        return f"SKIP {station_dir.name}"

    # Skip stations with no S2L2A tokens — avoids writing empty .complete stores
    # that would block rechunk_zarr.py from processing the same station
    s2_dir = station_dir / "S2L2A"
    if not s2_dir.exists() or not any(s2_dir.glob("*_L12_*.pt")):
        return f"MISS {station_dir.name}"

    if not execute:
        return f"DRY  {station_dir.name}"

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        store = zarr.DirectoryStore(str(out_dir))
        root  = zarr.open_group(store=store, mode="w")

        # Tokens
        write_tokens(root, "s2",       station_dir / "S2L2A")
        write_tokens(root, "s1_asc",   station_dir / "S1RTC", orbit="ASC")
        write_tokens(root, "s1_desc",  station_dir / "S1RTC", orbit="DESC")
        write_cloud_mask(root, station_dir / "CloudMask")
        write_static(root, station_dir)

        # Tabular / label modalities
        write_era5(root,   station_dir / "ERA5Land")
        write_sif(root,    station_dir / "SIF")
        write_twsa(root,   station_dir / "TWSA")
        write_labels(root, station_dir.name, category, start_date, end_date)
        if category in ("flux_only", "sm_and_flux"):
            write_flux_labels(root, station_dir, category, station_dir.name)

        zarr.consolidate_metadata(store)
        sentinel.touch()
        size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
        return f"OK   {station_dir.name} ({size_mb:.0f} MB)"

    except Exception as e:
        return f"ERR  {station_dir.name}: {e}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     type=int, default=0)
    parser.add_argument("--end",       type=int, default=None)
    parser.add_argument("--workers",   type=int, default=64)
    parser.add_argument("--execute",   action="store_true")
    parser.add_argument("--force",     action="store_true",
                        help="Re-run stations that already have .complete (Group A re-runs)")
    parser.add_argument("--stations",  type=str, default=None,
                        help="Comma-separated station dir names to process (e.g. ISMN_SNOTEL_Aniak,ICOS_CH-Fru)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="Root of token/feature data (default: %(default)s)")
    args = parser.parse_args()

    data_root = args.data_root
    station_filter = set(args.stations.split(",")) if args.stations else None

    df = pd.read_csv(SPLITS_CSV)

    def _category(r):
        if r["has_soil_moisture"] and r["has_flux"]: return "sm_and_flux"
        if r["has_soil_moisture"]: return "sm_only"
        return "flux_only"

    def _station_dir(r):
        cat = _category(r)
        if str(r["source_network"]) == "ISMN":
            sid = f"ISMN_{r['network']}_{r['station_name']}"
        else:
            sid = f"{r['source_network']}_{r['station_id']}"
        return str(data_root / cat / sid)

    df = df.dropna(subset=["source_network", "network", "station_id"])
    df["station_dir"] = df.apply(_station_dir, axis=1)
    df["category"]    = df.apply(_category, axis=1)

    end = args.end if args.end is not None else len(df)
    rows = df.iloc[args.start:end]
    if station_filter:
        rows = rows[rows["station_dir"].apply(lambda p: Path(p).name in station_filter)]

    batch = [
        (row["station_dir"], row["category"],
         str(row["start_date"]), str(row["end_date"]),
         args.execute, args.force)
        for _, row in rows.iterrows()
    ]

    print(f"Stations : {len(batch)}")
    print(f"Execute  : {args.execute}")
    print(f"Force    : {args.force}")
    print(f"Workers  : {args.workers}")
    print(f"Data root: {data_root}")
    print(f"Output   : {ZARR_ROOT}")

    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(convert_station, batch)):
            print(f"[{i+1:4d}/{len(batch)}] {result}")

    n_ok  = sum(1 for _ in ZARR_ROOT.rglob(".complete"))
    n_tot = len(batch)
    print(f"\nCompleted zarr stores: {n_ok} / {n_tot}")


if __name__ == "__main__":
    main()
