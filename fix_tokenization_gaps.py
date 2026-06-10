"""
fix_tokenization_gaps.py

For the 36 stations found by scan_tokenization_gaps.py where
satellite_zarr/{station}.zarr has S2/S1 dates that were never tokenized into
the production token zarr (s2/s1_asc/s1_desc all do a full overwrite from
satellite_zarr each time, so simply re-running the encoder with the existing
"already done" check bypassed regenerates the complete token set).

Two phases (mirrors retokenize_satellite_zarr.py):
  --mode terramind-force : re-encode s2/s1_asc/s1_desc (whichever has
                            n_extra > 0) from satellite_zarr. (terramind env, GPU)
  --mode cm-force        : for stations where s2 was re-encoded, re-run cloud
                            masks (cm/masks, cm/dates) so they match the new
                            s2/dates. (sensei env, GPU)

Usage:
    conda run --no-capture-output -n terramind python fix_tokenization_gaps.py --mode terramind-force --execute
    conda run --no-capture-output -n sensei    python fix_tokenization_gaps.py --mode cm-force        --execute
"""

import argparse
from pathlib import Path

import pandas as pd
import zarr

from retokenize_satellite_zarr import (
    ZARR_ROOT, SAT_ZARR_ROOT,
    _encode_temporal, _encode_dem, _encode_lulc,
    _run_cloud_masks, _load_encoder, _load_sensei,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["terramind-force", "cm-force"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    import torch
    gaps = pd.read_csv("csvs/tokenization_gap_scan.csv")

    if args.mode == "terramind-force":
        targets = gaps[
            (gaps["s2_n_extra"] > 0)
            | (gaps["s1_asc_n_extra"] > 0)
            | (gaps["s1_desc_n_extra"] > 0)
        ]
    else:
        targets = gaps[gaps["s2_n_extra"] > 0]

    print(f"Stations : {len(targets)}")
    print(f"Mode     : {args.mode}")
    print(f"Execute  : {args.execute}")
    if not args.execute:
        for _, r in targets.iterrows():
            print(f"  DRY {r['category']}/{r['station']}")
        return

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    print(f"Device   : {device}")

    if args.mode == "terramind-force":
        print("Loading TerraMindEncoder...")
        encoder = _load_encoder(device)
        for i, r in targets.iterrows():
            station, cat = r["station"], r["category"]
            sat_path  = SAT_ZARR_ROOT / f"{station}.zarr"
            token_dir = ZARR_ROOT / cat / station
            try:
                sat_zarr   = zarr.open_group(str(sat_path), mode="r")
                store      = zarr.DirectoryStore(str(token_dir))
                token_root = zarr.open_group(store=store, mode="a")

                n_s2 = n_asc = n_desc = 0
                if r["s2_n_extra"] > 0:
                    n_s2 = _encode_temporal(encoder, sat_zarr, "s2", "S2L2A",
                                             token_root, "s2", args.batch_size, device)
                if r["s1_asc_n_extra"] > 0:
                    n_asc = _encode_temporal(encoder, sat_zarr, "s1_asc", "S1RTC",
                                              token_root, "s1_asc", args.batch_size, device)
                if r["s1_desc_n_extra"] > 0:
                    n_desc = _encode_temporal(encoder, sat_zarr, "s1_desc", "S1RTC",
                                               token_root, "s1_desc", args.batch_size, device)
                if "dem"  not in token_root: _encode_dem(encoder,  sat_zarr, token_root, device)
                if "lulc" not in token_root: _encode_lulc(encoder, sat_zarr, token_root, device)
                zarr.consolidate_metadata(store)
                print(f"[{i+1:4d}/{len(gaps)}] OK-TM  {station}  s2={n_s2} s1_asc={n_asc} s1_desc={n_desc}", flush=True)
            except Exception as exc:
                import traceback
                print(f"[{i+1:4d}/{len(gaps)}] ERR    {station}: {exc}\n{traceback.format_exc()}", flush=True)

    else:  # cm-force
        print("Loading SEnSeIv2 (CloudSEN12)...")
        sensei = _load_sensei(str(device))
        for i, r in targets.iterrows():
            station, cat = r["station"], r["category"]
            sat_path  = SAT_ZARR_ROOT / f"{station}.zarr"
            token_dir = ZARR_ROOT / cat / station
            try:
                sat_zarr   = zarr.open_group(str(sat_path), mode="r")
                store      = zarr.DirectoryStore(str(token_dir))
                token_root = zarr.open_group(store=store, mode="a")

                n_cm = _run_cloud_masks(sensei, sat_zarr, token_root, args.batch_size, device)
                zarr.consolidate_metadata(store)
                print(f"[{i+1:4d}/{len(gaps)}] OK-CM  {station}  cm={n_cm}", flush=True)
            except Exception as exc:
                import traceback
                print(f"[{i+1:4d}/{len(gaps)}] ERR    {station}: {exc}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
