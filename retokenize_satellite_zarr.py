"""
retokenize_satellite_zarr.py

Fills missing tokens and cloud masks for Group B and Group C stations by reading
raw pixels from /projects/prjs1968/satellite_zarr/{station}.zarr/.

  Group B (--mode all):      TerraMind S2/S1/DEM/LULC + CloudSEN12 + tabular fill
  Group C (--mode cm-only):  CloudSEN12 only (S2 tokens already in zarr) + tabular fill

Also fills ERA5/SIF from /gpfs/scratch1/shared/pkhanal/data/{cat}/{station}/ and
writes SM labels from /projects/prjs1968/level1_organised/.

GPU models are loaded once in main() and reused across all stations.

Usage:
    python retokenize_satellite_zarr.py --mode all     --batch-size 8 --execute
    python retokenize_satellite_zarr.py --mode cm-only --batch-size 16 --execute
    python retokenize_satellite_zarr.py --station ISMN_SCAN_HubbardBrook --mode all --execute
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
import torch
import zarr

sys.path.insert(0, str(Path(__file__).parent))
from precompute_terramind import (
    TerraMindEncoder, _nn_fill_and_sanitize, _remap_lulc,
)
from cloud_masking_inference import load_sensei, S2_L2A_DESCRIPTORS
from create_token_zarr import (
    write_era5, write_sif, write_labels, write_flux_labels,
    ZARR_ROOT, LEVEL1_DIR, SPLITS_CSV, COMPRESSOR, T_TOKENS, T_CM,
)

SAT_ZARR_ROOT = Path("/projects/prjs1968/satellite_zarr")
SCRATCH_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/data")
DATA_ROOT     = Path("/projects/prjs1968/data")   # for flux labels.nc
LAYERS        = ["l12", "l9", "l6", "l3"]         # zarr key names (lowercase)


# ── zarr state helpers ────────────────────────────────────────────────────────

def _has_all_layers(token_dir: Path, key: str) -> bool:
    """True iff all 4 token layers + dates are present for key."""
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

def _encode_temporal(encoder: TerraMindEncoder,
                     sat_zarr: zarr.Group,
                     src_key: str, modality: str,
                     token_root: zarr.Group, zarr_key: str,
                     batch_size: int, device: torch.device) -> int:
    """Batch TerraMind over all acquisitions for one temporal modality.
    Writes l3/l6/l9/l12 + dates. Returns acquisition count."""
    if f"{src_key}/data" not in sat_zarr or f"{src_key}/dates" not in sat_zarr:
        return 0
    data_z  = sat_zarr[f"{src_key}/data"]   # (N, C, H, W)
    dates_z = sat_zarr[f"{src_key}/dates"]
    N = data_z.shape[0]
    if N == 0:
        return 0

    all_layers: dict[str, list] = {lay: [] for lay in LAYERS}

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        chunk = data_z[start:end].astype(np.float32)  # int16→float32 or float16→float32
        patches = [
            torch.from_numpy(_nn_fill_and_sanitize(chunk[i], modality))
            for i in range(end - start)
        ]
        batch_t = torch.stack(patches).to(device)
        with torch.no_grad():
            feats = encoder(batch_t, modality)   # {L3/L6/L9/L12: (B, 196, 768)}
        for lay in LAYERS:
            all_layers[lay].append(feats[lay.upper()].half().cpu().numpy())

    dates_raw = dates_z[:]
    dates_np  = np.array([
        d.decode() if isinstance(d, (bytes, np.bytes_)) else str(d)
        for d in dates_raw
    ], dtype="U8")

    for lay in LAYERS:
        arr = np.concatenate(all_layers[lay], axis=0)   # (N, 196, 768) fp16
        token_root.array(
            f"{zarr_key}/{lay}", arr,
            chunks=(min(T_TOKENS, N), 196, 768),
            dtype=arr.dtype, compressor=COMPRESSOR, overwrite=True,
        )
    token_root.array(f"{zarr_key}/dates", dates_np, chunks=(N,), overwrite=True)
    if "asc" in zarr_key:
        token_root[zarr_key].attrs["orbit"] = "ASC"
    elif "desc" in zarr_key:
        token_root[zarr_key].attrs["orbit"] = "DESC"
    return N


def _encode_dem(encoder: TerraMindEncoder, sat_zarr: zarr.Group,
                token_root: zarr.Group, device: torch.device) -> bool:
    if "dem/data" not in sat_zarr:
        return False
    arr = sat_zarr["dem/data"][:].astype(np.float32)   # (1, H, W)
    arr = _nn_fill_and_sanitize(arr, "DEM")
    with torch.no_grad():
        feats = encoder(torch.from_numpy(arr).unsqueeze(0).to(device), "DEM")
    tok = feats["L12"][0].half().cpu().numpy()
    token_root.array("dem", tok, compressor=COMPRESSOR, overwrite=True)
    return True


def _encode_lulc(encoder: TerraMindEncoder, sat_zarr: zarr.Group,
                 token_root: zarr.Group, device: torch.device) -> bool:
    if "lulc/data" not in sat_zarr:
        return False
    raw = sat_zarr["lulc/data"][-1]                    # (H, W) uint8, last year
    arr = _remap_lulc(raw).astype(np.float32)[np.newaxis]  # (1, H, W)
    arr = _nn_fill_and_sanitize(arr, "LULC")
    with torch.no_grad():
        feats = encoder(torch.from_numpy(arr).unsqueeze(0).to(device), "LULC")
    tok = feats["L12"][0].half().cpu().numpy()
    token_root.array("lulc", tok, compressor=COMPRESSOR, overwrite=True)
    return True


# ── CloudSEN12 inference ──────────────────────────────────────────────────────

def _run_cloud_masks(sensei,
                     sat_zarr: zarr.Group,
                     token_root: zarr.Group,
                     batch_size: int,
                     device: torch.device) -> int:
    """Run SEnSeIv2 on raw S2 int16 pixels → write cm/masks + cm/dates."""
    if "s2/data" not in sat_zarr or "s2/dates" not in sat_zarr:
        return 0
    data_z  = sat_zarr["s2/data"]    # (N, 12, H, W) int16
    dates_z = sat_zarr["s2/dates"]
    N = data_z.shape[0]
    if N == 0:
        return 0

    all_masks = []
    for start in range(0, N, batch_size):
        end      = min(start + batch_size, N)
        raw_ints = data_z[start:end]                  # (B, 12, H, W) int16
        # Nodata = all bands == 0 in raw DN
        nodata_masks = (raw_ints == 0).all(axis=1)    # (B, H, W)
        # Harmonised reflectance: (DN - 1000) / 10000, clipped to ≥ 0
        arr_f = np.clip(raw_ints.astype(np.int32) - 1000, 0, None).astype(np.float32) / 10_000.0
        batch_t    = torch.from_numpy(arr_f).to(device)
        desc_batch = [S2_L2A_DESCRIPTORS for _ in range(batch_t.shape[0])]
        with torch.no_grad():
            preds = sensei.model(batch_t, desc_batch)
        preds_np = preds.cpu().numpy()
        for i, pred in enumerate(preds_np):
            mask = sensei.postprocess(pred).astype(np.uint8)
            mask[nodata_masks[i]] = 255
            all_masks.append(mask)

    masks_arr = np.stack(all_masks, axis=0)           # (N, H, W) uint8
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

def _fill_tabular(token_root: zarr.Group,
                  cat: str, dir_name: str,
                  start_date: str, end_date: str) -> None:
    era5_dir  = SCRATCH_ROOT / cat / dir_name / "ERA5Land"
    sif_dir   = SCRATCH_ROOT / cat / dir_name / "SIF"
    soil_path = SCRATCH_ROOT / cat / dir_name / "soil" / "soil_patch.tif"

    if "era5/values" not in token_root and era5_dir.exists():
        write_era5(token_root, era5_dir)

    if "sif/values" not in token_root and sif_dir.exists():
        write_sif(token_root, sif_dir)

    if "soil" not in token_root and soil_path.exists():
        with rasterio.open(soil_path) as src:
            patch  = src.read().astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            patch[patch == nodata] = np.nan
        token_root.array("soil", patch, compressor=COMPRESSOR, overwrite=True)

    if "labels/sm" not in token_root:
        write_labels(token_root, dir_name, cat, start_date, end_date)

    if cat in ("flux_only", "sm_and_flux"):
        station_dir = DATA_ROOT / cat / dir_name
        if station_dir.exists() and "labels/le" not in token_root:
            write_flux_labels(token_root, station_dir)


# ── per-station driver ────────────────────────────────────────────────────────

def process_one(dir_name: str, cat: str,
                start_date: str, end_date: str,
                mode: str, batch_size: int,
                encoder,  # TerraMindEncoder | None
                sensei,   # CloudMask | None
                device: torch.device) -> str:
    sat_path  = SAT_ZARR_ROOT / f"{dir_name}.zarr"
    token_dir = ZARR_ROOT / cat / dir_name
    sentinel  = token_dir / ".complete"

    if not sat_path.exists():
        return f"MISS   {dir_name}: no satellite_zarr"

    try:
        t0 = time.perf_counter()

        sat_zarr = zarr.open_group(str(sat_path), mode="r")
        token_dir.mkdir(parents=True, exist_ok=True)
        store      = zarr.DirectoryStore(str(token_dir))
        token_root = zarr.open_group(store=store, mode="a")

        n_s2 = n_s1 = n_cm = 0

        if mode == "all" and encoder is not None:
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
            if "dem" not in token_root:
                _encode_dem(encoder, sat_zarr, token_root, device)
            if "lulc" not in token_root:
                _encode_lulc(encoder, sat_zarr, token_root, device)

        if sensei is not None and not _has_cm(token_dir):
            n_cm = _run_cloud_masks(sensei, sat_zarr, token_root, batch_size, device)

        _fill_tabular(token_root, cat, dir_name, start_date, end_date)

        zarr.consolidate_metadata(store)
        sentinel.touch()

        elapsed = time.perf_counter() - t0
        return f"OK     {dir_name}  s2={n_s2} s1={n_s1} cm={n_cm}  ({elapsed:.0f}s)"

    except Exception as exc:
        import traceback
        return f"ERR    {dir_name}: {exc}\n{traceback.format_exc()}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["all", "cm-only"], required=True,
                        help="all=Group B (TerraMind+CM), cm-only=Group C (CM only)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start",      type=int, default=0)
    parser.add_argument("--end",        type=int, default=None)
    parser.add_argument("--station",    type=str, default=None)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--execute",    action="store_true")
    args = parser.parse_args()

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
        # Auto-filter to only stations that need work
        if args.mode == "all":
            # Group B: .complete exists but s2/l12 missing
            def _needs_all(r):
                td = ZARR_ROOT / r["cat"] / r["dir_name"]
                return (td / ".complete").exists() and not _has_all_layers(td, "s2")
            df = df[df.apply(_needs_all, axis=1)]
        else:
            # Group C: s2 tokens present but no .complete yet
            def _needs_cm(r):
                td = ZARR_ROOT / r["cat"] / r["dir_name"]
                return _has_all_layers(td, "s2") and not (td / ".complete").exists()
            df = df[df.apply(_needs_cm, axis=1)]

    stations = [(r["dir_name"], r["cat"], str(r["start_date"]), str(r["end_date"]))
                for _, r in df.iterrows()]

    print(f"Stations : {len(stations)}")
    print(f"Mode     : {args.mode}")
    print(f"Execute  : {args.execute}")
    print(f"Device   : {args.device}")

    if not args.execute:
        for dn, cat, _, _ in stations:
            print(f"  DRY {cat}/{dn}")
        return

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    # Load models once
    encoder = None
    if args.mode == "all":
        print("Loading TerraMindEncoder...")
        encoder = TerraMindEncoder(frozen=True).to(device).eval()

    print("Loading SEnSeIv2 (CloudSEN12)...")
    sensei = load_sensei(str(device))

    for i, (dn, cat, start, end_d) in enumerate(stations):
        result = process_one(dn, cat, start, end_d,
                             args.mode, args.batch_size,
                             encoder, sensei, device)
        print(f"[{i+1:4d}/{len(stations)}] {result}")

    n_complete = sum(1 for _ in ZARR_ROOT.rglob(".complete"))
    print(f"\nTotal .complete: {n_complete}")


if __name__ == "__main__":
    main()
