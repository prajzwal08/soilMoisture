"""
Evaluate baseline_huber best.pt on OOS / OOT / OOST splits.

Outputs to meeting_output/:
  metrics_summary.csv              -- split × depth × {ubRMSE, RMSE, MAE, R, bias, n_stations}
  per_station_oos.csv
  per_station_oot.csv
  per_station_oost.csv

Usage:
    python evaluate_splits.py [--run-name baseline_huber] [--ckpt best.pt]
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel
from train import CudaPrefetcher

CKPT_ROOT  = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
ERA5_STATS = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json")
OUT_DIR    = Path("/gpfs/work3/0/prjs1968/soilMoisture/meeting_output")

EVAL_SPLITS = {
    "oos":  dict(split_filter=["oos"],          years=list(range(2016, 2023))),
    "oot":  dict(split_filter=["train", "val"],  years=[2023]),
    "oost": dict(split_filter=["oos"],           years=[2023]),
}

META_COLS = [
    "station_key", "latitude", "longitude", "IGBP", "igbp_macro",
    "koppen_geiger", "kg_macro", "elevation_band", "n_years",
    "start_date", "end_date", "split", "oot_eligible", "oost_eligible",
]


def worker_init_fn(worker_id):
    np.random.seed(os.getpid() + worker_id)


def load_checkpoint(ckpt_path: Path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = SoilMoistureModel(
        n_depths      = cfg.get("n_depths", 3),
        d_model       = cfg.get("d_model",  768),
        n_heads       = cfg.get("n_heads",  12),
        n_layers      = cfg.get("n_layers", 6),
        drop_path_rate= cfg.get("drop_path_rate", 0.0),
        use_cls_depth = cfg.get("use_cls_depth", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Epoch {ckpt['epoch']}  best_val_loss={ckpt.get('best_val_loss', 'N/A')}")
    return model, cfg, ckpt["epoch"]


@torch.no_grad()
def run_eval(model, loader, device):
    """Collect all predictions and targets across the loader."""
    SROW = SoilMoistureModel.STATION_ROW
    SCOL = SoilMoistureModel.STATION_COL
    all_preds, all_targets, all_keys = [], [], []

    for batch in CudaPrefetcher(loader, device):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(batch)
        all_preds.append(mu[:, :, SROW, SCOL].float().cpu().numpy())
        all_targets.append(batch["label"].cpu().numpy())
        all_keys.extend(batch["station_key"])

    preds   = np.concatenate(all_preds,   axis=0)   # (N, n_depths)
    targets = np.concatenate(all_targets, axis=0)   # (N, n_depths)
    return preds, targets, all_keys


def compute_station_metrics(preds, targets, station_keys):
    """Per-station {ubRMSE, RMSE, MAE, R, bias, n} for each depth."""
    groups = defaultdict(lambda: {"p": [], "t": []})
    for p, t, k in zip(preds, targets, station_keys):
        groups[k]["p"].append(p)
        groups[k]["t"].append(t)

    per_station = {}
    for station, data in groups.items():
        p_arr = np.array(data["p"])   # (T, n_depths)
        t_arr = np.array(data["t"])   # (T, n_depths)
        per_station[station] = {}
        for i, depth in enumerate(SM_DEPTHS):
            pi, ti = p_arr[:, i], t_arr[:, i]
            valid = ~np.isnan(ti)
            n = int(valid.sum())
            if n < 5:
                continue
            pv, tv = pi[valid], ti[valid]
            bias   = float(np.mean(pv - tv))
            rmse   = float(np.sqrt(np.mean((pv - tv) ** 2)))
            mae    = float(np.mean(np.abs(pv - tv)))
            p_anom = pv - pv.mean()
            t_anom = tv - tv.mean()
            ubrmse = float(np.sqrt(np.mean((p_anom - t_anom) ** 2)))
            r_val  = float(np.corrcoef(pv, tv)[0, 1]) if n > 2 else float("nan")
            per_station[station][depth] = {
                "ubRMSE": ubrmse, "RMSE": rmse, "MAE": mae,
                "R":      r_val,  "bias": bias, "n":  n,
            }
    return per_station


def global_metrics(per_station):
    """Mean-across-stations metrics per depth."""
    result = {}
    for depth in SM_DEPTHS:
        vals = [v[depth] for v in per_station.values() if depth in v]
        if not vals:
            continue
        result[depth] = {
            "ubRMSE":     float(np.mean([v["ubRMSE"] for v in vals])),
            "RMSE":       float(np.mean([v["RMSE"]   for v in vals])),
            "MAE":        float(np.mean([v["MAE"]    for v in vals])),
            "R":          float(np.nanmean([v["R"]   for v in vals])),
            "bias":       float(np.mean([v["bias"]   for v in vals])),
            "n_stations": len(vals),
        }
    return result


def build_station_df(per_station, splits_df, split_name):
    """Flatten per-station metrics to a DataFrame joined with metadata."""
    rows = []
    for station, depths in per_station.items():
        row = {"station_key": station}
        for depth in SM_DEPTHS:
            tag = depth.replace("-", "_")
            if depth in depths:
                for metric, val in depths[depth].items():
                    row[f"{metric}_{tag}"] = val
        rows.append(row)

    df = pd.DataFrame(rows)

    meta = splits_df.copy()
    def _make_key(r):
        if str(r["source_network"]) == "ISMN":
            return f"ISMN_{r['network']}_{r['station_name']}"
        return f"{r['source_network']}_{r['station_id']}"
    meta["station_key"] = meta.apply(_make_key, axis=1)
    keep = [c for c in META_COLS if c in meta.columns]
    df = df.merge(meta[keep], on="station_key", how="left")
    df.insert(0, "eval_split", split_name)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",     default="baseline_huber")
    parser.add_argument("--ckpt",         default="best.pt")
    parser.add_argument("--batch-size",   type=int, default=128)
    parser.add_argument("--num-workers",  type=int, default=8)
    parser.add_argument("--max-stations", type=int, default=None,
                        help="Cap stations per split (smoke-test mode)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = CKPT_ROOT / args.run_name / args.ckpt
    model, cfg, epoch = load_checkpoint(ckpt_path, device)

    splits_df    = pd.read_csv(SPLITS_CSV)
    summary_rows = []

    for split_name, split_cfg in EVAL_SPLITS.items():
        print(f"\n{'='*60}")
        print(f"Split: {split_name.upper()}  "
              f"filter={split_cfg['split_filter']}  years={split_cfg['years']}")

        ds = SoilMoistureDataset(
            splits_csv      = str(SPLITS_CSV),
            era5_stats_path = str(ERA5_STATS),
            years           = split_cfg["years"],
            category_filter = cfg.get("category_filter", ["sm_only"]),
            split_filter    = split_cfg["split_filter"],
            training        = False,
            use_mmap        = True,
            max_stations    = args.max_stations,
        )
        n_stations = len({s["station_key"] for s in ds.samples})
        print(f"  {len(ds):,} samples  |  {n_stations} stations")

        if len(ds) == 0:
            print("  No samples — skipping")
            continue

        loader = DataLoader(
            ds,
            batch_size         = args.batch_size,
            shuffle            = False,
            num_workers        = args.num_workers,
            pin_memory         = True,
            worker_init_fn     = worker_init_fn,
            persistent_workers = False,
            prefetch_factor    = 2,
        )

        preds, targets, keys = run_eval(model, loader, device)
        per_station = compute_station_metrics(preds, targets, keys)
        glob        = global_metrics(per_station)

        print(f"\n  Global metrics (mean across {len(per_station)} stations):")
        for depth, m in glob.items():
            print(f"    {depth:>8s}  ubRMSE={m['ubRMSE']:.4f}  RMSE={m['RMSE']:.4f}"
                  f"  MAE={m['MAE']:.4f}  R={m['R']:.3f}  bias={m['bias']:+.4f}"
                  f"  n={m['n_stations']}")
            summary_rows.append({"split": split_name, "depth": depth, **m})

        df       = build_station_df(per_station, splits_df, split_name)
        out_path = OUT_DIR / f"per_station_{split_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    print(f"\nSummary → {OUT_DIR / 'metrics_summary.csv'}")
    print("\n=== FINAL SUMMARY ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
