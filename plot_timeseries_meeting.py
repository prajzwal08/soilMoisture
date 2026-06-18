"""
Clean publication-quality predicted vs observed time series.

One figure per station: 3 stacked subplots (one per depth).
Predicted = coloured line.  Observed = black scatter dots.
All available years plotted on a continuous date axis.

Requires:
    meeting_output/per_station_oos.csv   (from evaluate_splits.py)

Outputs:
    meeting_output/timeseries/{station}_ts.png

Usage:
    python plot_timeseries_meeting.py [--run-name baseline_huber] [--n 5]
"""
import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

try:
    import scienceplots        # noqa: F401
    plt.style.use(["science", "nature"])
except ImportError:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10})

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel

CKPT_ROOT  = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
ERA5_STATS = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json")
OUT_DIR    = Path("meeting_output/timeseries")

DEPTH_COLORS = {"0-10": "#e74c3c", "10-30": "#2980b9", "30-100": "#27ae60"}
DEPTH_LABELS = {"0-10": "0–10 cm", "10-30": "10–30 cm", "30-100": "30–100 cm"}


def worker_init_fn(wid):
    np.random.seed(os.getpid() + wid)


def load_checkpoint(ckpt_path: Path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
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
    return model, cfg, ckpt["epoch"]


@torch.no_grad()
def infer_station(model, dataset, station_key: str, device, batch_size: int = 64):
    """Batched inference over all samples for one station.

    Returns dict with dates (list[datetime]), preds (T, n_depths), labels (T, n_depths).
    """
    indices = sorted(
        [i for i, s in enumerate(dataset.samples) if s["station_key"] == station_key],
        key=lambda i: (dataset.samples[i]["year"], dataset.samples[i]["doy"]),
    )
    if not indices:
        return None

    SROW = SoilMoistureModel.STATION_ROW
    SCOL = SoilMoistureModel.STATION_COL

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True,
                        worker_init_fn=worker_init_fn, prefetch_factor=2)

    all_dates, all_preds, all_labels = [], [], []
    offset = 0
    for batch in loader:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(batch_gpu)
        preds_cpu  = mu[:, :, SROW, SCOL].float().cpu().numpy()
        labels_cpu = batch["label"].cpu().numpy()
        for j in range(preds_cpu.shape[0]):
            s  = dataset.samples[indices[offset + j]]
            dt = datetime(s["year"], 1, 1) + timedelta(days=s["doy"] - 1)
            all_dates.append(dt)
            all_preds.append(preds_cpu[j])
            all_labels.append(labels_cpu[j])
        offset += preds_cpu.shape[0]

    return {
        "station": station_key,
        "dates":   all_dates,
        "preds":   np.array(all_preds),   # (T, n_depths)
        "labels":  np.array(all_labels),  # (T, n_depths)
    }


def plot_timeseries(result: dict, meta_row: dict, out_dir: Path, epoch: int,
                    rank_label: str = ""):
    """One figure per station: 3 depth subplots, predicted vs observed."""
    station = result["station"]
    dates   = pd.DatetimeIndex(result["dates"])
    preds   = result["preds"]    # (T, n_depths)
    labels  = result["labels"]   # (T, n_depths)

    igbp    = meta_row.get("IGBP",         "")
    climate = meta_row.get("koppen_geiger", "")
    n_yrs   = meta_row.get("n_years",       "")
    lat     = meta_row.get("latitude",  float("nan"))
    lon     = meta_row.get("longitude", float("nan"))

    station_short = station.replace("ISMN_", "").replace("_", " / ", 1)
    rank_str      = f"  [{rank_label}]" if rank_label else ""

    fig, axes = plt.subplots(
        len(SM_DEPTHS), 1,
        figsize=(12, 2.8 * len(SM_DEPTHS)),
        sharex=True,
    )
    if len(SM_DEPTHS) == 1:
        axes = [axes]

    fig.suptitle(
        f"{station_short}{rank_str}   |   IGBP: {igbp}   Climate: {climate}"
        f"   |   {lat:.2f}°N, {lon:.2f}°E   |   epoch {epoch}",
        fontsize=9.5, fontweight="bold",
    )

    for i, (ax, depth) in enumerate(zip(axes, SM_DEPTHS)):
        col   = DEPTH_COLORS[depth]
        label = DEPTH_LABELS[depth]
        pred  = preds[:, i]
        obs   = labels[:, i]
        valid = ~np.isnan(obs)

        # Light alternating year bands
        if len(dates) > 1:
            for yr in range(dates[0].year, dates[-1].year + 1):
                if yr % 2 == 0:
                    ax.axvspan(datetime(yr, 1, 1), datetime(yr, 12, 31),
                               color="#f7f7f7", zorder=0, lw=0)

        ax.plot(dates, pred, color=col, lw=1.1, label="Predicted", zorder=3)
        if valid.any():
            ax.scatter(dates[valid], obs[valid],
                       s=6, c="k", alpha=0.55, linewidths=0,
                       zorder=4, label="Observed")

        # Metric annotation
        if valid.sum() > 5:
            pv, tv = pred[valid], obs[valid]
            ub = np.sqrt(np.mean(((pv - pv.mean()) - (tv - tv.mean())) ** 2))
            rv = np.corrcoef(pv, tv)[0, 1]
            bv = np.mean(pv - tv)
            ax.set_title(
                f"{label}   ubRMSE = {ub:.3f}   R = {rv:.3f}   bias = {bv:+.3f}"
                f"   (n = {valid.sum()})",
                fontsize=8.5, loc="left", pad=3,
            )
        else:
            ax.set_title(f"{label}   (no observations)", fontsize=8.5, loc="left")

        ax.set_ylabel("SM  (m³/m³)", fontsize=8)
        ax.set_ylim(-0.02, 0.65)
        ax.legend(fontsize=7.5, loc="upper right", frameon=False,
                  markerscale=2.0)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.tick_params(axis="y", labelsize=7.5)

    # Shared x-axis formatting
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].tick_params(axis="x", labelsize=8, rotation=0)
    axes[-1].set_xlabel("Date", fontsize=8)

    fig.tight_layout()
    fname = out_dir / f"{station}_ts.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="baseline_huber")
    parser.add_argument("--ckpt",     default="best.pt")
    parser.add_argument("--split",    default="oos")
    parser.add_argument("--n",        type=int, default=5,
                        help="Number of best + worst stations to plot")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csv_path = Path("meeting_output") / f"per_station_{args.split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run evaluate_splits.py first")

    rank_df  = pd.read_csv(csv_path).dropna(subset=["ubRMSE_0_10"]).sort_values("ubRMSE_0_10")
    best_df  = rank_df.head(args.n).assign(_rank="BEST")
    worst_df = rank_df.tail(args.n).iloc[::-1].assign(_rank="WORST")
    selected = pd.concat([best_df, worst_df], ignore_index=True)

    print(f"\nBest {args.n}:")
    print(best_df[["station_key", "ubRMSE_0_10", "IGBP", "koppen_geiger"]].to_string(index=False))
    print(f"\nWorst {args.n}:")
    print(worst_df[["station_key", "ubRMSE_0_10", "IGBP", "koppen_geiger"]].to_string(index=False))

    ckpt_path = CKPT_ROOT / args.run_name / args.ckpt
    model, cfg, epoch = load_checkpoint(ckpt_path, device)

    ds = SoilMoistureDataset(
        splits_csv      = str(SPLITS_CSV),
        era5_stats_path = str(ERA5_STATS),
        years           = list(range(2016, 2024)),
        category_filter = cfg.get("category_filter", ["sm_only"]),
        split_filter    = None,
        training        = False,
        use_mmap        = True,
    )
    print(f"Dataset: {len(ds):,} total samples")

    for _, row in selected.iterrows():
        station = row["station_key"]
        rank    = row["_rank"]
        print(f"\n[{rank}] {station}")
        result = infer_station(model, ds, station, device)
        if result is None:
            print("  No samples — skipping")
            continue
        plot_timeseries(result, row.to_dict(), OUT_DIR, epoch, rank_label=rank)

    print(f"\nAll time series saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
