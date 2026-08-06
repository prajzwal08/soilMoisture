"""
Dump per-sample predictions for the held-out evaluation splits (§22).

One GPU pass over OOS / OOT / OOST writes a long-format parquet per split:
one row per (station, date, depth).  Every metric and figure downstream is a
CPU-only pass over these tables -- no GPU, no dataset rebuild.

Outputs to eval_output/:
    predictions_{oos,oot,oost}.parquet
    manifest.json                        -- run, checkpoint, epoch, split defs, counts

Usage:
    python eval_predict.py --run-name cls_depth_star_reg --ckpt best.pt
    python eval_predict.py --run-name cls_depth_star_reg --splits val   # metric gate
    python eval_predict.py --run-name cls_depth_star_reg --max-stations 5 --splits oos
"""
import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel
from train import CudaPrefetcher
from ckpt_utils import load_checkpoint

CKPT_ROOT  = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
ERA5_STATS = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json")
OUT_DIR    = Path("/gpfs/work3/0/prjs1968/soilMoisture/eval_output")

# §22.2.  "val" is not a held-out split -- it exists only to reproduce the
# training-time numbers (§22.6 hard gate) and is never run by default.
EVAL_SPLITS = {
    "oos":  dict(split_filter=["oos"],          years=list(range(2016, 2023))),
    "oot":  dict(split_filter=["train", "val"], years=[2023]),
    "oost": dict(split_filter=["oos"],          years=[2023]),
    "val":  dict(split_filter=["val"],          years=list(range(2016, 2023))),
}

# Expected station counts from the zarr probe (§22.2).  A large miss means the
# zarr or the year gating changed and the numbers must not be trusted.
EXPECTED_STATIONS = {"oos": 180, "oot": 399, "oost": 98, "val": 74}


def worker_init_fn(worker_id):
    np.random.seed(os.getpid() + worker_id)


def _make_key(r) -> str:
    """station_key as used by dataset.py (the zarr directory name)."""
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


@torch.no_grad()
def run_split(model, loader, device) -> dict:
    """Collect station-pixel predictions, targets and sample identity."""
    srow, scol = SoilMoistureModel.STATION_ROW, SoilMoistureModel.STATION_COL
    preds, targets, keys, years, doys = [], [], [], [], []

    t0, n_batches = time.time(), len(loader)
    for i, batch in enumerate(CudaPrefetcher(loader, device)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(batch)
        preds.append(mu[:, :, srow, scol].float().cpu().numpy())
        targets.append(batch["label"].float().cpu().numpy())
        keys.extend(batch["station_key"])
        years.append(np.asarray(batch["year"].cpu() if torch.is_tensor(batch["year"])
                                else batch["year"], dtype=np.int32))
        doys.append(np.asarray(batch["doy"].cpu() if torch.is_tensor(batch["doy"])
                               else batch["doy"], dtype=np.int32))

        if i % 200 == 0 and i:
            rate = (i + 1) / (time.time() - t0)
            eta  = (n_batches - i - 1) / max(rate, 1e-9) / 60
            print(f"    batch {i+1}/{n_batches}  {rate:.1f} b/s  ETA {eta:.1f} min",
                  flush=True)

    return dict(
        preds   = np.concatenate(preds,   axis=0),   # (N, n_depths)
        targets = np.concatenate(targets, axis=0),   # (N, n_depths)
        keys    = np.asarray(keys),                  # (N,)
        years   = np.concatenate(years,   axis=0),   # (N,)
        doys    = np.concatenate(doys,    axis=0),   # (N,)
    )


def to_long_frame(res: dict, split_name: str, train_split_map: dict) -> pd.DataFrame:
    """Wide (N, n_depths) arrays -> long format, one row per (sample, depth).

    Rows where the observation is NaN (depth absent at that station) are dropped.
    """
    n_samples, n_depths = res["preds"].shape
    frames = []
    for d, depth in enumerate(SM_DEPTHS[:n_depths]):
        obs = res["targets"][:, d]
        keep = ~np.isnan(obs)
        if not keep.any():
            continue
        frames.append(pd.DataFrame({
            "station_key": res["keys"][keep],
            "year":        res["years"][keep],
            "doy":         res["doys"][keep],
            "depth":       depth,
            "pred":        res["preds"][keep, d].astype(np.float32),
            "obs":         obs[keep].astype(np.float32),
        }))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # (year, doy) -> calendar date.  doy is 1-indexed, so subtract one day.
    df["date"] = (pd.to_datetime(df["year"].astype(str) + "-01-01")
                  + pd.to_timedelta(df["doy"] - 1, unit="D"))
    df["eval_split"]  = split_name
    df["train_split"] = df["station_key"].map(train_split_map).astype("string")
    df["depth"]       = df["depth"].astype("category")
    df["station_key"] = df["station_key"].astype("string")

    return df[["station_key", "year", "doy", "date", "depth",
               "pred", "obs", "eval_split", "train_split"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name",     default="cls_depth_star_reg")
    p.add_argument("--ckpt",         default="best.pt")
    p.add_argument("--batch-size",   type=int, default=128)
    p.add_argument("--num-workers",  type=int, default=8)
    p.add_argument("--splits",       nargs="+", default=["oos", "oot", "oost"],
                   choices=list(EVAL_SPLITS))
    p.add_argument("--max-stations", type=int, default=None,
                   help="Cap stations per split (smoke-test mode)")
    p.add_argument("--out-dir",      default=str(OUT_DIR))
    # Station chunking -- same pattern as precompute_terramind.py.  The dataset
    # preloads L12 into RAM, so OOT (774 stations, ~156 GB) spends ~65 min in
    # init.  Splitting it across parallel jobs cuts both peak RAM and wall time
    # at the same total GPU cost.  eval_metrics.py globs the chunks back together.
    p.add_argument("--csv-start-idx", type=int, default=None,
                   help="First row of the split-filtered station CSV (inclusive)")
    p.add_argument("--csv-end-idx",   type=int, default=None,
                   help="Last row of the split-filtered station CSV (exclusive)")
    p.add_argument("--tag",           default=None,
                   help="Suffix for output files, e.g. --tag c0 -> "
                        "predictions_oot_c0.parquet")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        raise SystemExit("CUDA required -- CudaPrefetcher and autocast assume a GPU")

    ckpt_path = CKPT_ROOT / args.run_name / args.ckpt
    model, cfg, epoch = load_checkpoint(ckpt_path, device)

    splits_df = pd.read_csv(SPLITS_CSV)
    train_split_map = dict(zip(splits_df.apply(_make_key, axis=1), splits_df["split"]))

    # Splits may be produced by separate jobs (OOT needs ~156 GB of L12 in RAM,
    # the others ~39 GB, so they are submitted separately).  Merge into any
    # existing manifest rather than clobbering the other job's entries.
    manifest_path = out_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            print(f"  WARNING -- unreadable {manifest_path}, starting fresh")
    prior_splits = manifest.get("splits", {})

    manifest = {
        "run_name":        args.run_name,
        "checkpoint":      str(ckpt_path),
        "epoch":           int(epoch),
        "best_val_loss":   float(cfg.get("best_val_loss", float("nan")))
                           if isinstance(cfg.get("best_val_loss"), (int, float)) else None,
        "category_filter": cfg.get("category_filter", ["sm_only"]),
        "depths":          SM_DEPTHS,
        "generated":       datetime.now().isoformat(timespec="seconds"),
        "max_stations":    args.max_stations,
        "splits":          dict(prior_splits),   # keep the other job's entries
    }

    for split_name in args.splits:
        scfg = EVAL_SPLITS[split_name]
        print(f"\n{'='*66}")
        print(f"Split: {split_name.upper()}  filter={scfg['split_filter']}  "
              f"years={scfg['years'][0]}-{scfg['years'][-1]}")

        # The dataset preloads every station's L12 tokens into RAM: ~39 GB for
        # oos, ~156 GB for oot (train+val).  Without freeing the previous split
        # first, `ds = SoilMoistureDataset(...)` would hold both at once and
        # peak near 195 GB.  Free explicitly before constructing the next.
        ds = loader = res = df = None
        gc.collect()

        # Chunking: slice the split-filtered rows and hand the dataset a
        # temporary CSV holding only this chunk's stations, so it preloads
        # only their L12 tokens.
        active_csv = str(SPLITS_CSV)
        if args.csv_start_idx is not None or args.csv_end_idx is not None:
            sub = splits_df[splits_df["split"].isin(scfg["split_filter"])]
            lo  = args.csv_start_idx or 0
            hi  = args.csv_end_idx if args.csv_end_idx is not None else len(sub)
            sub = sub.iloc[lo:hi]
            chunk_csv = out_dir / f"_chunk_{split_name}_{lo}_{hi}.csv"
            sub.to_csv(chunk_csv, index=False)
            active_csv = str(chunk_csv)
            print(f"  Chunk rows [{lo}:{hi}] of {split_name} → {len(sub)} stations")

        ds = SoilMoistureDataset(
            splits_csv      = active_csv,
            era5_stats_path = str(ERA5_STATS),
            years           = scfg["years"],
            category_filter = cfg.get("category_filter", ["sm_only"]),
            split_filter    = scfg["split_filter"],
            training        = False,
            use_mmap        = True,
            max_stations    = args.max_stations,
        )
        n_stations = len({s["station_key"] for s in ds.samples})
        expected   = EXPECTED_STATIONS.get(split_name)
        chunked    = args.csv_start_idx is not None or args.csv_end_idx is not None
        print(f"  {len(ds):,} samples | {n_stations} stations "
              f"(expected ~{expected})")
        if (args.max_stations is None and not chunked
                and expected and abs(n_stations - expected) > 5):
            print(f"  WARNING -- station count differs from the §22.2 probe by "
                  f"{n_stations - expected:+d}. Investigate before trusting metrics.")

        if len(ds) == 0:
            print("  No samples -- skipping")
            continue

        loader = DataLoader(
            ds,
            batch_size         = args.batch_size,
            shuffle            = False,
            num_workers        = args.num_workers,
            pin_memory         = True,
            worker_init_fn     = worker_init_fn,
            persistent_workers = False,
            prefetch_factor    = 2 if args.num_workers > 0 else None,
        )

        t0  = time.time()
        res = run_split(model, loader, device)
        df  = to_long_frame(res, split_name, train_split_map)
        mins = (time.time() - t0) / 60

        if df.empty:
            print("  All observations NaN -- nothing written")
            continue

        n_nan_pred = int(df["pred"].isna().sum())
        if n_nan_pred:
            print(f"  WARNING -- {n_nan_pred} NaN predictions")

        suffix   = f"_{args.tag}" if args.tag else ""
        out_path = out_dir / f"predictions_{split_name}{suffix}.parquet"
        df.to_parquet(out_path, index=False, compression="snappy")

        print(f"  {len(df):,} rows | {df['station_key'].nunique()} stations | "
              f"{mins:.1f} min")
        print(f"  rows per depth: "
              f"{df.groupby('depth', observed=True).size().to_dict()}")
        print(f"  Saved: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

        manifest["splits"][f"{split_name}{suffix}"] = {
            "split_filter": scfg["split_filter"],
            "years":        [int(scfg["years"][0]), int(scfg["years"][-1])],
            "n_stations":   int(df["station_key"].nunique()),
            "n_rows":       int(len(df)),
            "n_samples":    int(len(ds)),
            "rows_by_depth": {str(k): int(v) for k, v in
                              df.groupby("depth", observed=True).size().items()},
            "nan_pred":     n_nan_pred,
            "minutes":      round(mins, 1),
        }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {manifest_path}")
    print("\n=== DONE ===")
    for name, m in manifest["splits"].items():
        print(f"  {name:>5s}  {m['n_stations']:>4d} stations  "
              f"{m['n_rows']:>9,d} rows  {m['minutes']:>5.1f} min")


if __name__ == "__main__":
    main()
