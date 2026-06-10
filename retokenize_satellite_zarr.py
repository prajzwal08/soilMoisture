"""
retokenize_satellite_zarr.py

Two-phase pipeline for Group B and Group C stations.

PHASE 1 — TerraMind tokens (run in terramind conda env):
    --mode terramind   → Group B only: S2/S1/DEM/LULC tokens from satellite_zarr

PHASE 2 — CloudSEN12 + tabular fill (run in sensei conda env):
    --mode cm-fill     → Groups B and C: cloud masks + ERA5/SIF/labels + .complete

Run order:
    # Group B (83 stations)
    sbatch slurm/retokenize_b.sh      # Phase 1: terramind env
    sbatch --dependency=afterok:$JOBID slurm/retokenize_b2.sh  # Phase 2: sensei env

    # Group C (62 stations) — tokens already in zarr, only Phase 2 needed
    sbatch slurm/retokenize_c.sh      # Phase 2: sensei env

All lazy imports: top-level only imports numpy/pandas/zarr/pathlib so the script
can be loaded in either conda env without missing-module errors.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import rasterio
import zarr

# ── paths ─────────────────────────────────────────────────────────────────────

ZARR_ROOT     = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SAT_ZARR_ROOT = Path("/projects/prjs1968/satellite_zarr")
SCRATCH_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/data")
DATA_ROOT     = Path("/projects/prjs1968/data")
LEVEL1_DIR    = Path("/projects/prjs1968/level1_organised")
SPLITS_CSV    = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
T_TOKENS   = 32
T_CM       = 128
LAYERS     = ["l12", "l9", "l6", "l3"]


# ── lazy model loaders ────────────────────────────────────────────────────────

def _load_encoder(device):
    """Load TerraMindEncoder — requires terramind conda env (terratorch)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from precompute_terramind import TerraMindEncoder
    return TerraMindEncoder(frozen=True).to(device).eval()


def _load_sensei(device_str: str):
    """Load SEnSeIv2 CloudMask — requires sensei conda env (senseiv2)."""
    from senseiv2.inference import CloudMask
    from senseiv2.utils import get_model_files
    cfg_path, wts_path = get_model_files("SEnSeIv2-SegFormerB2-alldata-ambiguous")
    return CloudMask(cfg_path, wts_path, device=device_str,
                     output_style=None, categorise=True)


def _s2_descriptors():
    """S2L2A band descriptors for SEnSeIv2."""
    from senseiv2.constants import SENTINEL2_DESCRIPTORS
    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    return [SENTINEL2_DESCRIPTORS[i] for i in idx]


# ── nodata fill helpers (copied from precompute_terramind — no import needed) ─

_FILLABLE_THRESH = 0.01
_TOKEN_SIZE      = 16
_N_SIDE          = 14


def _nn_fill_and_sanitize(arr: np.ndarray, modality: str) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    if modality == "S2L2A":
        nodata_2d = (arr == 0).any(axis=0)
    elif modality == "LULC":
        nodata_2d = (arr == 0).all(axis=0)
    else:
        nodata_2d = np.isnan(arr).any(axis=0)

    if not nodata_2d.any():
        return arr

    out      = arr.copy()
    fill_2d  = np.zeros(nodata_2d.shape, dtype=bool)
    for i in range(_N_SIDE):
        for j in range(_N_SIDE):
            rs, re = i * _TOKEN_SIZE, (i + 1) * _TOKEN_SIZE
            cs, ce = j * _TOKEN_SIZE, (j + 1) * _TOKEN_SIZE
            patch  = nodata_2d[rs:re, cs:ce]
            frac   = patch.sum() / (_TOKEN_SIZE * _TOKEN_SIZE)
            if 0 < frac < _FILLABLE_THRESH:
                fill_2d[rs:re, cs:ce] = patch

    if fill_2d.any():
        _, nn_idx = distance_transform_edt(nodata_2d, return_indices=True)
        for c in range(out.shape[0]):
            out[c][fill_2d] = out[c][nn_idx[0][fill_2d], nn_idx[1][fill_2d]]

    if modality != "S2L2A":
        np.nan_to_num(out, nan=0.0, copy=False)
    return out


_LULC_REMAP = np.array([0, 1, 2, 9, 3, 4, 9, 5, 6, 7, 8, 9], dtype=np.uint8)


def _remap_lulc(arr: np.ndarray) -> np.ndarray:
    raw = np.asarray(arr, dtype=np.int16)
    out = np.where((raw >= 0) & (raw < len(_LULC_REMAP)),
                   _LULC_REMAP[np.clip(raw, 0, len(_LULC_REMAP) - 1)], np.uint8(0))
    return out.astype(np.uint8)


# ── zarr state checks ─────────────────────────────────────────────────────────

def _has_all_layers(token_dir: Path, key: str) -> bool:
    return (
        all((token_dir / key / lay / ".zarray").exists() for lay in LAYERS)
        and (token_dir / key / "dates" / ".zarray").exists()
    )


def _has_cm(token_dir: Path) -> bool:
    return (
        (token_dir / "cm" / "masks" / ".zarray").exists()
        and (token_dir / "cm" / "dates" / ".zarray").exists()
    )


# ── TerraMind inference ───────────────────────────────────────────────────────

def _encode_temporal(encoder, sat_zarr, src_key, modality,
                     token_root, zarr_key, batch_size, device) -> int:
    import torch
    if f"{src_key}/data" not in sat_zarr or f"{src_key}/dates" not in sat_zarr:
        return 0
    data_z  = sat_zarr[f"{src_key}/data"]
    dates_z = sat_zarr[f"{src_key}/dates"]
    N = data_z.shape[0]
    if N == 0:
        return 0

    all_layers: dict[str, list] = {lay: [] for lay in LAYERS}
    for start in range(0, N, batch_size):
        end     = min(start + batch_size, N)
        chunk   = data_z[start:end].astype(np.float32)
        patches = [
            torch.from_numpy(_nn_fill_and_sanitize(chunk[i], modality))
            for i in range(end - start)
        ]
        batch_t = torch.stack(patches).to(device)
        with torch.no_grad():
            feats = encoder(batch_t, modality)
        for lay in LAYERS:
            all_layers[lay].append(feats[lay.upper()].half().cpu().numpy())

    dates_raw = dates_z[:]
    dates_np  = np.array([
        d.decode() if isinstance(d, (bytes, np.bytes_)) else str(d)
        for d in dates_raw
    ], dtype="U8")
    for lay in LAYERS:
        arr = np.concatenate(all_layers[lay], axis=0)
        token_root.array(
            f"{zarr_key}/{lay}", arr,
            chunks=(min(T_TOKENS, N), 196, 768),
            dtype=arr.dtype, compressor=COMPRESSOR, overwrite=True,
        )
    token_root.array(f"{zarr_key}/dates", dates_np, chunks=(N,), overwrite=True)
    if "asc"  in zarr_key: token_root[zarr_key].attrs["orbit"] = "ASC"
    if "desc" in zarr_key: token_root[zarr_key].attrs["orbit"] = "DESC"
    return N


def _encode_dem(encoder, sat_zarr, token_root, device) -> bool:
    import torch
    if "dem/data" not in sat_zarr: return False
    arr = sat_zarr["dem/data"][:].astype(np.float32)
    arr = _nn_fill_and_sanitize(arr, "DEM")
    with torch.no_grad():
        feats = encoder(torch.from_numpy(arr).unsqueeze(0).to(device), "DEM")
    tok = feats["L12"][0].half().cpu().numpy()
    token_root.array("dem", tok, compressor=COMPRESSOR, overwrite=True)
    return True


def _encode_lulc(encoder, sat_zarr, token_root, device) -> bool:
    import torch
    if "lulc/data" not in sat_zarr: return False
    raw = sat_zarr["lulc/data"][-1]
    arr = _remap_lulc(raw).astype(np.float32)[np.newaxis]
    arr = _nn_fill_and_sanitize(arr, "LULC")
    with torch.no_grad():
        feats = encoder(torch.from_numpy(arr).unsqueeze(0).to(device), "LULC")
    tok = feats["L12"][0].half().cpu().numpy()
    token_root.array("lulc", tok, compressor=COMPRESSOR, overwrite=True)
    return True


# ── CloudSEN12 inference ──────────────────────────────────────────────────────

def _run_cloud_masks(sensei, sat_zarr, token_root, batch_size, device) -> int:
    import torch
    descriptors = _s2_descriptors()
    if "s2/data" not in sat_zarr or "s2/dates" not in sat_zarr: return 0
    data_z  = sat_zarr["s2/data"]
    dates_z = sat_zarr["s2/dates"]
    N = data_z.shape[0]
    if N == 0: return 0

    all_masks = []
    for start in range(0, N, batch_size):
        end      = min(start + batch_size, N)
        raw      = data_z[start:end]               # (B, 12, H, W) int16
        nodata   = (raw == 0).all(axis=1)          # (B, H, W)
        arr_f    = np.clip(raw.astype(np.int32) - 1000, 0, None).astype(np.float32) / 10_000.0
        batch_t  = torch.from_numpy(arr_f).to(device)
        desc_b   = [descriptors for _ in range(batch_t.shape[0])]
        with torch.no_grad():
            preds = sensei.model(batch_t, desc_b)
        for i, pred in enumerate(preds.cpu().numpy()):
            mask = sensei.postprocess(pred).astype(np.uint8)
            mask[nodata[i]] = 255
            all_masks.append(mask)

    masks_arr = np.stack(all_masks, axis=0)
    dates_raw = dates_z[:]
    dates_np  = np.array([
        d.decode() if isinstance(d, (bytes, np.bytes_)) else str(d)
        for d in dates_raw
    ], dtype="U8")
    H, W = masks_arr.shape[1], masks_arr.shape[2]
    token_root.array("cm/masks", masks_arr,
                     chunks=(min(T_CM, N), H, W),
                     dtype=np.uint8, compressor=COMPRESSOR, overwrite=True)
    token_root.array("cm/dates", dates_np, chunks=(N,), overwrite=True)
    return N


# ── tabular fill ──────────────────────────────────────────────────────────────

def _date_to_int(t) -> int:
    ts = pd.Timestamp(t)
    return ts.year * 10000 + ts.month * 100 + ts.day


def _merge_nc(directory: Path, stem: str):
    import xarray as xr
    files = sorted(directory.glob(f"{stem}_*.nc")) if directory.exists() else []
    if not files: return None
    if len(files) == 1: return xr.open_dataset(files[0])
    ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal",
                           coords="minimal", compat="override")
    _, idx = np.unique(ds["time"].values, return_index=True)
    return ds.isel(time=idx)


def _write_era5(root, era5_dir):
    era5_vars = [
        "t2m_mean","t2m_min","t2m_max","d2m_mean","d2m_min","d2m_max",
        "skt_mean","skt_min","skt_max","u10_mean","u10_min","u10_max",
        "v10_mean","v10_min","v10_max","sp_mean","sp_min","sp_max","tp_sum",
    ]
    ds = _merge_nc(era5_dir, "meteo")
    if ds is None: return
    with ds:
        vals  = np.stack([ds[v].values for v in era5_vars], axis=-1).astype(np.float32)
        times = pd.DatetimeIndex(ds["time"].values)
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.day_of_year for t in times], dtype=np.int32)
    root.array("era5/values",    vals,      compressor=COMPRESSOR, overwrite=True)
    root.array("era5/date_ints", date_ints, overwrite=True)
    root.array("era5/doys",      doys,      overwrite=True)


def _write_sif(root, sif_dir):
    ds = _merge_nc(sif_dir, "sif")
    if ds is None: return
    with ds:
        vals  = ds["sif"].values.astype(np.float32)
        times = pd.to_datetime(ds["time"].values)
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.timetuple().tm_yday for t in times], dtype=np.int32)
    root.array("sif/values",    vals,      compressor=COMPRESSOR, overwrite=True)
    root.array("sif/date_ints", date_ints, overwrite=True)
    root.array("sif/doys",      doys,      overwrite=True)


def _write_labels(root, dir_name, category, start_date, end_date):
    import xarray as xr
    nc = LEVEL1_DIR / category / f"{dir_name}.nc"
    if not nc.exists(): return
    ds = xr.open_dataset(nc)
    tc = "date_time" if "date_time" in ds else "time"
    t_all = pd.DatetimeIndex(ds[tc].values)
    mask  = (t_all >= pd.Timestamp(start_date)) & (t_all <= pd.Timestamp(end_date))
    if mask.sum() == 0:
        ds.close()
        return
    sm_da = ds["soil_moisture"]
    if sm_da.dims[0] == tc:
        sm_da = sm_da.transpose("depth", tc)
    sm     = sm_da.values.astype(np.float32)[:, mask]
    depths = np.array([str(d) for d in ds["depth"].values], dtype="U20")
    qc     = np.zeros_like(sm, dtype=np.uint8)
    for qv in ("soil_moisture_qc", "quality_flag"):
        if qv in ds:
            qc_da = ds[qv]
            if qc_da.dims[0] == tc:
                qc_da = qc_da.transpose("depth", tc)
            qc = qc_da.values.astype(np.uint8)[:, mask]
            break
    # Flux labels (sm_and_flux files include LE_F_MDS)
    if "LE_F_MDS" in ds:
        le   = ds["LE_F_MDS"].values.astype(np.float32)[mask]
        le_q = (ds["LE_F_MDS_QC"].values.astype(np.float32)[mask]
                if "LE_F_MDS_QC" in ds else np.zeros_like(le, dtype=np.float32))
        d_fl = np.array([t.strftime("%Y%m%d") for t in t_all[mask]], dtype="U8")
    else:
        le = le_q = d_fl = None
    ds.close()
    dates = np.array([t.strftime("%Y%m%d") for t in t_all[mask]], dtype="U8")
    root.array("labels/sm",     sm,     compressor=COMPRESSOR, overwrite=True)
    root.array("labels/qc",     qc,     compressor=COMPRESSOR, overwrite=True)
    root.array("labels/depths", depths, overwrite=True)
    root.array("labels/dates",  dates,  overwrite=True)
    if le is not None:
        root.array("labels/le",         le,   compressor=COMPRESSOR, overwrite=True)
        root.array("labels/le_qc",      le_q, compressor=COMPRESSOR, overwrite=True)
        root.array("labels/dates_flux", d_fl, overwrite=True)


def _write_flux_labels(root, station_dir, category=None, dir_name=None):
    """Write LE_F_MDS flux labels.

    Primary: station_dir/labels.nc; fallback: LEVEL1_DIR/category/dir_name.nc.
    LE_F_MDS_QC is a daily gap fraction in [0,1] → stored as float32.
    """
    import xarray as xr
    nc = station_dir / "labels.nc"
    if not nc.exists() and category and dir_name:
        nc = LEVEL1_DIR / category / f"{dir_name}.nc"
    if not nc.exists(): return
    with xr.open_dataset(nc) as ds:
        if "LE_F_MDS" not in ds: return
        le   = ds["LE_F_MDS"].values.astype(np.float32)
        qc   = (ds["LE_F_MDS_QC"].values.astype(np.float32)
                if "LE_F_MDS_QC" in ds else np.zeros_like(le, dtype=np.float32))
        tc   = "date_time" if "date_time" in ds else "time"
        t    = pd.DatetimeIndex(ds[tc].values)
    dates = np.array([x.strftime("%Y%m%d") for x in t], dtype="U8")
    root.array("labels/le",         le,    compressor=COMPRESSOR, overwrite=True)
    root.array("labels/le_qc",      qc,    compressor=COMPRESSOR, overwrite=True)
    root.array("labels/dates_flux", dates, overwrite=True)


def _fill_tabular(token_root, cat, dir_name, start_date, end_date):
    era5_dir  = SCRATCH_ROOT / cat / dir_name / "ERA5Land"
    sif_dir   = SCRATCH_ROOT / cat / dir_name / "SIF"
    soil_path = SCRATCH_ROOT / cat / dir_name / "soil" / "soil_patch.tif"

    if "era5/values" not in token_root and era5_dir.exists():
        _write_era5(token_root, era5_dir)
    if "sif/values" not in token_root and sif_dir.exists():
        _write_sif(token_root, sif_dir)
    if "soil" not in token_root and soil_path.exists():
        with rasterio.open(soil_path) as src:
            patch  = src.read().astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            patch[patch == nodata] = np.nan
        token_root.array("soil", patch, compressor=COMPRESSOR, overwrite=True)
    if "labels/sm" not in token_root and cat in ("sm_only", "sm_and_flux"):
        _write_labels(token_root, dir_name, cat, start_date, end_date)
    if cat == "flux_only" and "labels/le" not in token_root:
        station_dir = DATA_ROOT / cat / dir_name
        _write_flux_labels(token_root, station_dir, category=cat, dir_name=dir_name)


# ── per-station drivers ───────────────────────────────────────────────────────

def process_terramind(dir_name, cat, encoder, batch_size, device) -> str:
    """Phase 1: write S2/S1/DEM/LULC tokens from satellite_zarr."""
    sat_path  = SAT_ZARR_ROOT / f"{dir_name}.zarr"
    token_dir = ZARR_ROOT / cat / dir_name
    if not sat_path.exists():
        return f"MISS   {dir_name}"
    try:
        sat_zarr = zarr.open_group(str(sat_path), mode="r")
        token_dir.mkdir(parents=True, exist_ok=True)
        store      = zarr.DirectoryStore(str(token_dir))
        token_root = zarr.open_group(store=store, mode="a")
        n_s2 = n_s1 = 0
        if not _has_all_layers(token_dir, "s2"):
            n_s2 = _encode_temporal(encoder, sat_zarr, "s2", "S2L2A",
                                     token_root, "s2", batch_size, device)
        if not _has_all_layers(token_dir, "s1_asc"):
            n = _encode_temporal(encoder, sat_zarr, "s1_asc", "S1RTC",
                                  token_root, "s1_asc", batch_size, device)
            n_s1 += n
        if not _has_all_layers(token_dir, "s1_desc"):
            n = _encode_temporal(encoder, sat_zarr, "s1_desc", "S1RTC",
                                  token_root, "s1_desc", batch_size, device)
            n_s1 += n
        if "dem"  not in token_root: _encode_dem(encoder,  sat_zarr, token_root, device)
        if "lulc" not in token_root: _encode_lulc(encoder, sat_zarr, token_root, device)
        zarr.consolidate_metadata(store)
        return f"OK-TM  {dir_name}  s2={n_s2} s1={n_s1}"
    except Exception as exc:
        import traceback
        return f"ERR    {dir_name}: {exc}\n{traceback.format_exc()}"


def process_cm_fill(dir_name, cat, start_date, end_date,
                    sensei, batch_size, device) -> str:
    """Phase 2: cloud masks + tabular fill + .complete."""
    sat_path  = SAT_ZARR_ROOT / f"{dir_name}.zarr"
    token_dir = ZARR_ROOT / cat / dir_name
    sentinel  = token_dir / ".complete"
    if not sat_path.exists():
        return f"MISS   {dir_name}"
    try:
        t0 = time.perf_counter()
        sat_zarr = zarr.open_group(str(sat_path), mode="r")
        token_dir.mkdir(parents=True, exist_ok=True)
        store      = zarr.DirectoryStore(str(token_dir))
        token_root = zarr.open_group(store=store, mode="a")
        n_cm = 0
        if not _has_cm(token_dir):
            n_cm = _run_cloud_masks(sensei, sat_zarr, token_root, batch_size, device)
        _fill_tabular(token_root, cat, dir_name, start_date, end_date)
        zarr.consolidate_metadata(store)
        sentinel.touch()
        elapsed = time.perf_counter() - t0
        return f"OK-CM  {dir_name}  cm={n_cm}  ({elapsed:.0f}s)"
    except Exception as exc:
        import traceback
        return f"ERR    {dir_name}: {exc}\n{traceback.format_exc()}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["terramind", "cm-fill"],
                        help="terramind=Phase1 (terramind env), cm-fill=Phase2 (sensei env)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start",      type=int, default=0)
    parser.add_argument("--end",        type=int, default=None)
    parser.add_argument("--station",    type=str, default=None)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--execute",    action="store_true")
    args = parser.parse_args()

    import torch
    df = pd.read_csv(SPLITS_CSV)
    df = df.dropna(subset=["source_network", "network", "station_id"])

    def _cat(r):
        if r["has_soil_moisture"] and r["has_flux"]: return "sm_and_flux"
        if r["has_soil_moisture"]: return "sm_only"
        return "flux_only"

    def _sid(r):
        if str(r["source_network"]) == "ISMN":
            return f"ISMN_{r['network']}_{r['station_name']}"
        return f"{r['source_network']}_{r['station_id']}"

    df["cat"]      = df.apply(_cat, axis=1)
    df["dir_name"] = df.apply(_sid, axis=1)

    if args.station:
        df = df[df["dir_name"] == args.station]
    else:
        end = args.end if args.end is not None else len(df)
        df  = df.iloc[args.start:end]
        if args.mode == "terramind":
            # Group B: .complete exists but s2 missing
            df = df[df.apply(
                lambda r: (ZARR_ROOT / r["cat"] / r["dir_name"] / ".complete").exists()
                          and not _has_all_layers(ZARR_ROOT / r["cat"] / r["dir_name"], "s2"),
                axis=1,
            )]
        else:  # cm-fill
            # Group B (after terramind phase) + Group C (s2 tokens present, no .complete)
            df = df[df.apply(
                lambda r: _has_all_layers(ZARR_ROOT / r["cat"] / r["dir_name"], "s2")
                          and not (ZARR_ROOT / r["cat"] / r["dir_name"] / ".complete").exists(),
                axis=1,
            )]

    stations = list(df.itertuples(index=False))
    print(f"Stations : {len(stations)}")
    print(f"Mode     : {args.mode}")
    print(f"Execute  : {args.execute}")
    if not args.execute:
        for r in stations[:5]:
            print(f"  DRY {r.cat}/{r.dir_name}")
        if len(stations) > 5:
            print(f"  ... and {len(stations)-5} more")
        return

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    print(f"Device   : {device}")

    if args.mode == "terramind":
        print("Loading TerraMindEncoder...")
        encoder = _load_encoder(device)
        for i, r in enumerate(stations):
            result = process_terramind(r.dir_name, r.cat,
                                       encoder, args.batch_size, device)
            print(f"[{i+1:4d}/{len(stations)}] {result}", flush=True)
    else:
        print("Loading SEnSeIv2 (CloudSEN12)...")
        sensei = _load_sensei(str(device))
        for i, r in enumerate(stations):
            result = process_cm_fill(
                r.dir_name, r.cat, str(r.start_date), str(r.end_date),
                sensei, args.batch_size, device,
            )
            print(f"[{i+1:4d}/{len(stations)}] {result}", flush=True)

    n_complete = sum(1 for _ in ZARR_ROOT.rglob(".complete"))
    print(f"\nTotal .complete: {n_complete}")


if __name__ == "__main__":
    main()
