"""
Create one zarr store per station containing ALL modalities.

Output: /gpfs/work3/0/prjs1968/data/zarr/{category}/{station}/
  s2/l12, l9, l6, l3   (N, 196, 768) fp16  chunks=(1,196,768)  zstd-3
  s2/dates              (N,) str
  s1_asc/{layers,dates}  s1_desc/{layers,dates}   (if orbits exist)
  cm/masks              (N, 224, 224) uint8  chunks=(1,224,224)  zstd-3
  cm/dates              (N,) str
  dem                   (196, 768) fp16
  lulc                  (196, 768) fp16
  era5/values (N,19), date_ints (N,) int32, doys (N,) int32
  sif/values  (N,),   date_ints (N,) int32, doys (N,) int32
  twsa/lwe (N,), lwe_uncertainty (N,), date_ints (N,) int32, doys (N,) int32
  labels/sm (n_depths,n_days), qc (n_depths,n_days), depths (n_depths,), dates (n_days,)
  .complete   ← sentinel written last; station skipped if already present

zarr becomes the single format for training AND publication. Original .pt and .nc
files are deleted after zarr is verified clean.

Usage:
    python create_token_zarr.py [--start 0] [--end 993] [--workers 32] [--execute]
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
ZARR_ROOT  = Path("/gpfs/work3/0/prjs1968/data/zarr")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")

ERA5_VARS = [
    "t2m_mean", "t2m_min", "t2m_max", "d2m_mean", "d2m_min", "d2m_max",
    "skt_mean", "skt_min", "skt_max", "u10_mean", "u10_min", "u10_max",
    "v10_mean", "v10_min", "v10_max", "sp_mean",  "sp_min",  "sp_max",
    "tp_sum",
]

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
LAYERS     = ["l12", "l9", "l6", "l3"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _first(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern)) if directory.exists() else []
    return files[0] if files else None


def _date_to_int(t) -> int:
    ts = pd.Timestamp(t)
    return ts.year * 10000 + ts.month * 100 + ts.day


# ── per-modality writers ──────────────────────────────────────────────────────

def write_tokens(root: zarr.Group, key: str, pt_dir: Path, orbit: str | None = None):
    """Write one set of token layers (S2 or one S1 orbit) into zarr group."""
    dates = None
    for layer in LAYERS:
        if orbit:
            pt = _first(pt_dir, f"*_{orbit}_{layer.upper()}_*.pt")
        else:
            pt = _first(pt_dir, f"*_{layer.upper()}_*.pt")
        if pt is None:
            return False   # missing — skip this modality entirely
        d = torch.load(pt, map_location="cpu", weights_only=False)
        arr = d["tokens"].numpy()                    # (N, 196, 768) fp16
        root.array(f"{key}/{layer}", arr,
                   chunks=(1, arr.shape[1], arr.shape[2]),
                   dtype=arr.dtype, compressor=COMPRESSOR, overwrite=True)
        if dates is None:
            dates = d["dates"]
    if dates is not None:
        root.array(f"{key}/dates", np.array(dates, dtype="U8"),
                   chunks=(len(dates),), overwrite=True)
    return True


def write_cloud_mask(root: zarr.Group, cm_dir: Path):
    cm_pt = _first(cm_dir, "*_*.pt")
    if cm_pt is None:
        return
    d = torch.load(cm_pt, map_location="cpu", weights_only=False)
    masks = d["masks"].numpy()                       # (N, 224, 224) uint8
    root.array("cm/masks", masks,
               chunks=(1, masks.shape[1], masks.shape[2]),
               dtype=masks.dtype, compressor=COMPRESSOR, overwrite=True)
    root.array("cm/dates", np.array(d["dates"], dtype="U8"),
               chunks=(len(d["dates"]),), overwrite=True)


def write_static(root: zarr.Group, dem_pt: Path, lulc_pt: Path):
    for name, path in [("dem", dem_pt), ("lulc", lulc_pt)]:
        if path.exists():
            t = torch.load(path, map_location="cpu", weights_only=True)
            root.array(name, t.numpy(), compressor=COMPRESSOR, overwrite=True)


def write_era5(root: zarr.Group, era5_dir: Path):
    nc = _first(era5_dir, "meteo_*_*.nc")
    if nc is None:
        return
    ds    = xr.open_dataset(nc)
    vals  = np.stack([ds[v].values for v in ERA5_VARS], axis=-1).astype(np.float32)
    times = pd.DatetimeIndex(ds["time"].values)
    ds.close()
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.day_of_year  for t in times], dtype=np.int32)
    root.array("era5/values",    vals,       compressor=COMPRESSOR, overwrite=True)
    root.array("era5/date_ints", date_ints,  overwrite=True)
    root.array("era5/doys",      doys,       overwrite=True)


def write_sif(root: zarr.Group, sif_dir: Path):
    nc = _first(sif_dir, "sif_*_*.nc")
    if nc is None:
        return
    ds    = xr.open_dataset(nc)
    vals  = ds["sif"].values.astype(np.float32)
    times = pd.to_datetime(ds["time"].values)
    ds.close()
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.timetuple().tm_yday for t in times], dtype=np.int32)
    root.array("sif/values",    vals,      compressor=COMPRESSOR, overwrite=True)
    root.array("sif/date_ints", date_ints, overwrite=True)
    root.array("sif/doys",      doys,      overwrite=True)


def write_twsa(root: zarr.Group, twsa_dir: Path):
    nc = _first(twsa_dir, "twsa_*_*.nc")
    if nc is None:
        return
    ds    = xr.open_dataset(nc)
    lwe   = ds["lwe"].values.astype(np.float32)
    unc   = ds["lwe_uncertainty"].values.astype(np.float32) if "lwe_uncertainty" in ds else np.zeros_like(lwe)
    times = pd.to_datetime(ds["time"].values)
    ds.close()
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.timetuple().tm_yday for t in times], dtype=np.int32)
    root.array("twsa/lwe",             lwe,       compressor=COMPRESSOR, overwrite=True)
    root.array("twsa/lwe_uncertainty", unc,       compressor=COMPRESSOR, overwrite=True)
    root.array("twsa/date_ints",       date_ints, overwrite=True)
    root.array("twsa/doys",            doys,      overwrite=True)


def write_labels(root: zarr.Group, labels_nc: Path):
    if not labels_nc.exists():
        return
    ds         = xr.open_dataset(labels_nc)
    sm         = ds["soil_moisture"].values.astype(np.float32)   # (n_depths, n_days)
    depths     = np.array([str(d) for d in ds["depth"].values], dtype="U20")
    time_coord = "date_time" if "date_time" in ds else "time"
    times      = pd.DatetimeIndex(ds[time_coord].values)
    # QC flag: 0=observed, 1=gap-filled, 2=missing
    qc = np.zeros_like(sm, dtype=np.uint8)
    if "quality_flag" in ds:
        qc = ds["quality_flag"].values.astype(np.uint8)
    elif "sm_flag" in ds:
        qc = ds["sm_flag"].values.astype(np.uint8)
    ds.close()
    dates = np.array([t.strftime("%Y%m%d") for t in times], dtype="U8")
    root.array("labels/sm",     sm,     compressor=COMPRESSOR, overwrite=True)
    root.array("labels/qc",     qc,     compressor=COMPRESSOR, overwrite=True)
    root.array("labels/depths", depths, overwrite=True)
    root.array("labels/dates",  dates,  overwrite=True)


# ── per-station conversion ────────────────────────────────────────────────────

def convert_station(args: tuple) -> str:
    station_dir, category, execute = args
    station_dir = Path(station_dir)
    out_dir     = ZARR_ROOT / category / station_dir.name
    sentinel    = out_dir / ".complete"

    if sentinel.exists():
        return f"SKIP {station_dir.name}"

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
        write_static(root, station_dir / "DEM" / "dem_L12.pt",
                          station_dir / "LULC" / "lulc_L12.pt")

        # Tabular / label modalities
        write_era5(root,   station_dir / "ERA5Land")
        write_sif(root,    station_dir / "SIF")
        write_twsa(root,   station_dir / "TWSA")
        write_labels(root, station_dir / "labels.nc")

        sentinel.touch()
        size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
        return f"OK   {station_dir.name} ({size_mb:.0f} MB)"

    except Exception as e:
        return f"ERR  {station_dir.name}: {e}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   type=int, default=0)
    parser.add_argument("--end",     type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(SPLITS_CSV)

    def _category(r):
        if r["has_soil_moisture"] and r["has_flux"]: return "sm_and_flux"
        if r["has_soil_moisture"]: return "sm_only"
        return "flux_only"

    def _station_dir(r):
        cat = _category(r)
        if str(r["source_network"]) != str(r["network"]):
            sid = f"{r['source_network']}_{r['network']}_{r['station_id']}"
        else:
            sid = f"{r['source_network']}_{r['station_id']}"
        return str(DATA_ROOT / cat / sid)

    df = df.dropna(subset=["source_network", "network", "station_id"])
    df["station_dir"] = df.apply(_station_dir, axis=1)
    df["category"]    = df.apply(_category, axis=1)

    end   = args.end if args.end is not None else len(df)
    batch = [(row["station_dir"], row["category"], args.execute)
             for _, row in df.iloc[args.start:end].iterrows()]

    print(f"Stations : {len(batch)}")
    print(f"Execute  : {args.execute}")
    print(f"Workers  : {args.workers}")
    print(f"Output   : {ZARR_ROOT}")

    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(convert_station, batch)):
            print(f"[{i+1:4d}/{len(batch)}] {result}")

    # Summary
    n_ok   = sum(1 for _ in ZARR_ROOT.rglob(".complete"))
    n_tot  = len(batch)
    print(f"\nCompleted zarr stores: {n_ok} / {n_tot}")


if __name__ == "__main__":
    main()
