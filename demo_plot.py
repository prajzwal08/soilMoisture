"""
Quick meeting demo: load best.pt and plot predicted vs observed SM
for a handful of validation stations.

Usage:
    python demo_plot.py [--checkpoint PATH] [--n-stations 4] [--year 2021]
                        [--out demo_output/]

Produces per-station plots:
  - Predicted vs observed time series for all 3 depths
  - Predicted uncertainty band (σ = exp(0.5 * log_var))
  - 224×224 spatial SM map for the target day (shows spatial context)
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel

# ─────────────────────────────────────────────────────────────────────────────

CONFIG_DEFAULTS = {
    "splits_csv"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv",
    "data_root"     : "/gpfs/work3/0/prjs1968/data",
    "era5_stats"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json",
    "checkpoint_dir": "/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only",
    "category_filter": ["sm_only"],
    "years"         : list(range(2016, 2024)),
    "run_name"      : "baseline_nll",
}

DEPTH_COLORS = {"0-10": "#e74c3c", "10-30": "#2980b9", "30-100": "#27ae60"}


def load_checkpoint(ckpt_path: Path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = SoilMoistureModel(
        n_depths            = cfg.get("n_depths", 3),
        d_model             = cfg.get("d_model",  768),
        n_heads             = cfg.get("n_heads",  12),
        n_layers            = cfg.get("n_layers", 6),
        predict_uncertainty = cfg.get("predict_uncertainty", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Epoch {ckpt['epoch']}  |  best_val_loss={ckpt['best_val_loss']:.4f}")
    return model, cfg


@torch.no_grad()
def run_station_year(model, dataset, station_key: str, year: int, device):
    """Run inference for all days in one station-year. Returns arrays aligned to DoY."""
    indices = [
        i for i, s in enumerate(dataset.samples)
        if s["station_key"] == station_key and s["year"] == year
    ]
    if not indices:
        return None

    doys, mu_list, sigma_list, label_list = [], [], [], []

    for idx in sorted(indices, key=lambda i: dataset.samples[i]["doy"]):
        s     = dataset.samples[idx]
        batch = dataset[idx]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        mu, log_var = model(batch)
        mu_pt = mu[0, :, model.STATION_ROW, model.STATION_COL].cpu().numpy()

        if log_var is not None:
            sigma_pt = (0.5 * log_var[0, :, model.STATION_ROW, model.STATION_COL]).exp().cpu().numpy()
        else:
            sigma_pt = np.zeros_like(mu_pt)

        doys.append(s["doy"])
        mu_list.append(mu_pt)
        sigma_list.append(sigma_pt)
        label_list.append(batch["label"][0].cpu().numpy())

        # Also save the spatial map for the middle day
        if idx == indices[len(indices) // 2]:
            spatial_mu = mu[0].cpu().numpy()   # (n_depths, 224, 224)

    return {
        "doys"   : np.array(doys),
        "mu"     : np.array(mu_list),     # (T, n_depths)
        "sigma"  : np.array(sigma_list),  # (T, n_depths)
        "labels" : np.array(label_list),  # (T, n_depths)
        "spatial": spatial_mu,            # (n_depths, 224, 224)
        "station": station_key,
        "year"   : year,
    }


def plot_station(result: dict, out_dir: Path, epoch: int):
    station = result["station"]
    year    = result["year"]
    doys    = result["doys"]
    mu      = result["mu"]
    sigma   = result["sigma"]
    labels  = result["labels"]
    spatial = result["spatial"]

    n_depths = len(SM_DEPTHS)
    fig = plt.figure(figsize=(16, 4 * n_depths + 4))
    gs  = gridspec.GridSpec(n_depths + 1, 2, figure=fig,
                            width_ratios=[2, 1], hspace=0.45, wspace=0.3)

    fig.suptitle(f"{station}  |  {year}  |  epoch {epoch}", fontsize=13, fontweight="bold")

    for i, depth in enumerate(SM_DEPTHS):
        ax = fig.add_subplot(gs[i, 0])
        col = DEPTH_COLORS[depth]

        obs  = labels[:, i]
        pred = mu[:, i]
        sig  = sigma[:, i]
        obs_mask = ~np.isnan(obs)

        ax.fill_between(doys, pred - sig, pred + sig, alpha=0.25, color=col, label="±1σ")
        ax.plot(doys, pred, color=col, lw=1.5, label="Predicted")
        if obs_mask.any():
            ax.scatter(doys[obs_mask], obs[obs_mask], s=8, color="k", zorder=5,
                       label="Observed", alpha=0.7)

        # compute metrics where observed
        if obs_mask.sum() > 5:
            p, t = pred[obs_mask], obs[obs_mask]
            ubrmse = np.sqrt(np.mean(((p - p.mean()) - (t - t.mean())) ** 2))
            bias   = np.mean(p - t)
            ax.set_title(f"Depth {depth} cm  |  ubRMSE={ubrmse:.3f}  bias={bias:+.3f}",
                         fontsize=9)
        else:
            ax.set_title(f"Depth {depth} cm  (insufficient obs)", fontsize=9)

        ax.set_xlabel("Day of year")
        ax.set_ylabel("SM (m³/m³)")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(1, 365)

    # Spatial SM maps (one per depth in a subplot row at the bottom)
    ax_map = fig.add_subplot(gs[n_depths, :])
    n = n_depths
    imgs = np.concatenate([spatial[i] for i in range(n)], axis=1)  # side-by-side
    im = ax_map.imshow(imgs, cmap="YlOrBr_r", vmin=0.0, vmax=0.5, aspect="auto")
    plt.colorbar(im, ax=ax_map, fraction=0.015, pad=0.01, label="SM (m³/m³)")
    # depth labels
    for i, depth in enumerate(SM_DEPTHS):
        ax_map.text(224 * i + 112, 5, f"{depth} cm", ha="center", va="top",
                    color="white", fontsize=9, fontweight="bold")
    # station pixel
    for i in range(n):
        ax_map.axvline(224 * i + 112, color="cyan", lw=0.8, ls="--", alpha=0.6)
    ax_map.axhline(112, color="cyan", lw=0.8, ls="--", alpha=0.6)
    ax_map.set_title("Spatial SM map (mid-year snapshot) — cyan lines = station pixel")
    ax_map.axis("off")

    fname = out_dir / f"{station}_{year}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best.pt (default: {checkpoint_dir}/{run_name}/best.pt)")
    parser.add_argument("--run-name",   type=str, default=CONFIG_DEFAULTS["run_name"])
    parser.add_argument("--n-stations", type=int, default=4,
                        help="Number of val stations to plot")
    parser.add_argument("--year",       type=int, default=2021)
    parser.add_argument("--out",        type=str, default="demo_output")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = (Path(CONFIG_DEFAULTS["checkpoint_dir"])
                     / args.run_name / "best.pt")

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Has training started? Check with: ls -lh "
              f"{Path(CONFIG_DEFAULTS['checkpoint_dir']) / args.run_name}/")
        return

    model, cfg = load_checkpoint(ckpt_path, device)
    epoch = torch.load(ckpt_path, map_location="cpu", weights_only=False)["epoch"]

    # Build val dataset
    print("Building val dataset...")
    val_ds = SoilMoistureDataset(
        splits_csv       = CONFIG_DEFAULTS["splits_csv"],
        data_root        = CONFIG_DEFAULTS["data_root"],
        era5_stats_path  = CONFIG_DEFAULTS["era5_stats"],
        years            = [args.year],
        category_filter  = CONFIG_DEFAULTS["category_filter"],
        split_filter     = ["val"],
        training         = False,
    )

    # Pick stations that have data in the requested year
    available = sorted({s["station_key"] for s in val_ds.samples
                        if s["year"] == args.year})
    if not available:
        print(f"No val stations have data in {args.year}. Try --year 2020 or 2019.")
        return

    random.seed(42)
    chosen = random.sample(available, min(args.n_stations, len(available)))
    print(f"Plotting {len(chosen)} stations for year {args.year}: {chosen}")

    for station in chosen:
        print(f"\n→ {station}")
        result = run_station_year(model, val_ds, station, args.year, device)
        if result is None:
            print("  No samples — skipping")
            continue
        plot_station(result, out_dir, epoch)

    print(f"\nAll plots saved to {out_dir}/")
    print("Copy to local machine with:")
    print(f"  scp snellius:{out_dir.resolve()}/*.png .")


if __name__ == "__main__":
    main()
