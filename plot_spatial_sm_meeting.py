"""
SM spatial maps: n_depths rows × N_dates columns for best & worst OOS stations.

For each station, picks 6 dates spread across one representative year (every ~60 days),
runs model inference per date → (n_depths, 224, 224) SM map, and plots a grid.

Observed SM value at the station pixel is annotated on each panel.

Requires:
    meeting_output/per_station_oos.csv   (from evaluate_splits.py)

Outputs:
    meeting_output/spatial/{station}_spatial.png

Usage:
    python plot_spatial_sm_meeting.py [--run-name baseline_huber] [--n 5]
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "nature"])
except ImportError:
    plt.rcParams.update({"font.size": 9})

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel

CKPT_ROOT  = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
ERA5_STATS = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json")
OUT_DIR    = Path("meeting_output/spatial")

# Dates to sample: ~every 60 days, DOY 30/90/150/210/270/330
SAMPLE_DOYS = [30, 90, 150, 210, 270, 330]

DEPTH_CMAPS   = {"0-10": "YlOrBr", "10-30": "YlGnBu", "30-100": "PuBu"}
DEPTH_LABELS  = {"0-10": "0–10 cm", "10-30": "10–30 cm", "30-100": "30–100 cm"}
SM_VMIN, SM_VMAX = 0.0, 0.5

SROW = SoilMoistureModel.STATION_ROW
SCOL = SoilMoistureModel.STATION_COL


def load_checkpoint(ckpt_path: Path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = SoilMoistureModel(
        n_depths       = cfg.get("n_depths", 3),
        d_model        = cfg.get("d_model",  768),
        n_heads        = cfg.get("n_heads",  12),
        n_layers       = cfg.get("n_layers", 6),
        drop_path_rate = cfg.get("drop_path_rate", 0.0),
        use_cls_depth  = cfg.get("use_cls_depth", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt["epoch"]


@torch.no_grad()
def infer_spatial_for_dates(model, dataset, station_key: str,
                            target_year: int, target_doys: list[int],
                            device) -> list[dict | None]:
    """
    For each DOY in target_doys, find the closest sample for (station, year),
    run inference, and return a list of dicts with {date, sm_map (n_depths,224,224),
    obs_sm (n_depths,)}.
    """
    # Index all samples for this station in this year by DOY
    cands = {
        dataset.samples[i]["doy"]: i
        for i, s in enumerate(dataset.samples)
        if s["station_key"] == station_key and s["year"] == target_year
    }
    if not cands:
        # Try any available year
        years_avail = sorted({s["year"] for s in dataset.samples
                               if s["station_key"] == station_key})
        if not years_avail:
            return [None] * len(target_doys)
        target_year = years_avail[len(years_avail) // 2]
        cands = {
            dataset.samples[i]["doy"]: i
            for i, s in enumerate(dataset.samples)
            if s["station_key"] == station_key and s["year"] == target_year
        }

    available_doys = np.array(sorted(cands.keys()))
    results = []
    for doy in target_doys:
        diffs = np.abs(available_doys - doy)
        best_doy = int(available_doys[np.argmin(diffs)])
        if diffs.min() > 45:   # more than 45 days away — skip
            results.append(None)
            continue

        idx  = cands[best_doy]
        item = dataset[idx]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v]
                 for k, v in item.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(batch)
        sm_map = mu[0].float().cpu().numpy()   # (n_depths, 224, 224)
        obs_sm = item["label"].numpy()          # (n_depths,)
        date_dt = datetime(target_year, 1, 1) + timedelta(days=best_doy - 1)
        results.append({"date": date_dt, "sm_map": sm_map, "obs_sm": obs_sm,
                        "doy": best_doy, "target_doy": doy})

    return results


def plot_spatial_grid(station_key: str, rank_label: str, meta_row: dict,
                      date_results: list[dict | None], out_dir: Path):
    """
    n_depths rows × N_dates columns.
    Each cell: SM heatmap with crosshair at (112,112) and obs value annotated.
    """
    n_depths = len(SM_DEPTHS)
    valid    = [(i, r) for i, r in enumerate(date_results) if r is not None]
    if not valid:
        print(f"  No valid inference results for {station_key} — skipping")
        return

    n_cols  = len(valid)
    igbp    = meta_row.get("IGBP",         "")
    climate = meta_row.get("koppen_geiger", "")
    ubrmse  = meta_row.get("ubRMSE_0_10",  float("nan"))

    station_short = station_key.replace("ISMN_", "").replace("_", " / ", 1)

    fig_w = max(8, n_cols * 2.4)
    fig_h = n_depths * 2.4 + 0.7
    fig, axes = plt.subplots(n_depths, n_cols,
                              figsize=(fig_w, fig_h),
                              squeeze=False)

    fig.suptitle(
        f"[{rank_label}]  {station_short}   IGBP: {igbp}   Climate: {climate}"
        f"   ubRMSE₀₋₁₀ = {ubrmse:.3f}",
        fontsize=9, fontweight="bold", y=1.01,
    )

    for col_idx, (_, res) in enumerate(valid):
        date_str = res["date"].strftime("%b %d %Y")
        for row_idx, depth in enumerate(SM_DEPTHS):
            ax   = axes[row_idx][col_idx]
            cmap = DEPTH_CMAPS[depth]
            sm   = res["sm_map"][row_idx]       # (224, 224)
            obs  = res["obs_sm"][row_idx]

            im = ax.imshow(sm, cmap=cmap,
                           vmin=SM_VMIN, vmax=SM_VMAX,
                           interpolation="bilinear", aspect="equal")

            # Crosshair at station pixel
            ax.axhline(SROW, color="cyan", lw=0.8, ls="--", alpha=0.8)
            ax.axvline(SCOL, color="cyan", lw=0.8, ls="--", alpha=0.8)
            ax.plot(SCOL, SROW, "c^", ms=4, zorder=5)

            # Observed value annotation
            if not np.isnan(obs):
                ax.text(SCOL + 7, SROW - 7, f"obs={obs:.2f}",
                        fontsize=5.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="k", alpha=0.45, lw=0))

            ax.set_xticks([])
            ax.set_yticks([])

            # Row label (left column only)
            if col_idx == 0:
                ax.set_ylabel(DEPTH_LABELS[depth], fontsize=7.5, labelpad=3)

            # Column header (top row only)
            if row_idx == 0:
                ax.set_title(date_str, fontsize=7.5, pad=4)

        # Colourbar on rightmost column
        if col_idx == n_cols - 1:
            for row_idx, depth in enumerate(SM_DEPTHS):
                ax  = axes[row_idx][col_idx]
                cax = ax.inset_axes([1.03, 0, 0.05, 1])
                sm_map_dummy = plt.cm.ScalarMappable(
                    cmap=DEPTH_CMAPS[depth],
                    norm=plt.Normalize(SM_VMIN, SM_VMAX))
                sm_map_dummy.set_array([])
                cb = plt.colorbar(sm_map_dummy, cax=cax)
                cb.set_label("m³/m³", fontsize=6, labelpad=2)
                cb.ax.tick_params(labelsize=5.5)

    fig.tight_layout(rect=[0, 0, 0.97, 1])
    fname = out_dir / f"{station_key}_spatial.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="baseline_huber")
    parser.add_argument("--ckpt",     default="best.pt")
    parser.add_argument("--n",        type=int, default=5,
                        help="Number of best + worst stations")
    parser.add_argument("--year",     type=int, default=None,
                        help="Force specific year (default: median available year)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csv_path = Path("meeting_output/per_station_oos.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run evaluate_splits.py first")

    rank_df  = pd.read_csv(csv_path).dropna(subset=["ubRMSE_0_10"]).sort_values("ubRMSE_0_10")
    best_df  = rank_df.head(args.n).assign(_rank="BEST")
    worst_df = rank_df.tail(args.n).iloc[::-1].assign(_rank="WORST")
    selected = pd.concat([best_df, worst_df], ignore_index=True)

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
    print(f"Dataset: {len(ds):,} samples")

    for _, row in selected.iterrows():
        station    = row["station_key"]
        rank_label = row["_rank"]
        print(f"\n[{rank_label}] {station}")

        if args.year is not None:
            target_year = args.year
        else:
            yrs = sorted({s["year"] for s in ds.samples if s["station_key"] == station})
            if not yrs:
                print("  No samples — skipping")
                continue
            # Prefer OOS years (2016-2022)
            oos_yrs = [y for y in yrs if 2016 <= y <= 2022]
            pool    = oos_yrs if oos_yrs else yrs
            target_year = pool[len(pool) // 2]

        print(f"  Year: {target_year}  DOYs: {SAMPLE_DOYS}")
        date_results = infer_spatial_for_dates(
            model, ds, station, target_year, SAMPLE_DOYS, device)

        n_valid = sum(1 for r in date_results if r is not None)
        print(f"  Valid dates: {n_valid}/{len(SAMPLE_DOYS)}")

        plot_spatial_grid(station, rank_label, row.to_dict(), date_results, OUT_DIR)

    print(f"\nAll spatial figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
