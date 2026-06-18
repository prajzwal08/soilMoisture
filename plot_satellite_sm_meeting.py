"""
Satellite imagery (PCA pseudo-RGB/SAR) + SM spatial maps for best & worst OOS stations.

For each station, picks the mid-year date with best S2 coverage, applies PCA on
TerraMind L3 tokens (196 patches → 3 PCA components → 14×14 → bicubic 224×224),
runs model inference to get the SM spatial map, then plots a 5-panel figure.

Requires:
    meeting_output/per_station_oos.csv  (from evaluate_splits.py)

Outputs:
    meeting_output/satellite/{station}_{date}_BEST.png
    meeting_output/satellite/{station}_{date}_WORST.png

Usage:
    python plot_satellite_sm_meeting.py [--run-name baseline_huber] [--n 5]
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "nature"])
except ImportError:
    plt.rcParams.update({"font.size": 9})

from dataset import SoilMoistureDataset, SM_DEPTHS, ZARR_ROOT
from model import SoilMoistureModel

CKPT_ROOT  = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
ERA5_STATS = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json")
OUT_DIR    = Path("meeting_output/satellite")


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


def _pca_pseudo_image(tokens: np.ndarray) -> np.ndarray:
    """
    tokens : (196, 768) float16 or float32
    Returns: (224, 224, 3) uint8  — bicubic-upscaled pseudo-RGB
    """
    from sklearn.decomposition import PCA

    t = tokens.astype(np.float32)
    # Standardise each dimension (important for meaningful PCA)
    t = (t - t.mean(0)) / (t.std(0) + 1e-6)

    pca  = PCA(n_components=3, random_state=0)
    proj = pca.fit_transform(t)           # (196, 3)

    # Normalise each channel to [0, 1]
    for c in range(3):
        lo, hi = proj[:, c].min(), proj[:, c].max()
        proj[:, c] = (proj[:, c] - lo) / (hi - lo + 1e-8)

    img_14 = proj.reshape(14, 14, 3)      # (14, 14, 3) in [0, 1]

    # Bicubic upscale to 224×224 via torch
    t_img  = torch.from_numpy(img_14).permute(2, 0, 1).unsqueeze(0).float()
    t_up   = torch.nn.functional.interpolate(t_img, size=(224, 224), mode="bicubic",
                                              align_corners=False)
    arr    = t_up.squeeze(0).permute(1, 2, 0).numpy()
    arr    = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def load_l3_tokens_for_date(station_key: str, category: str, modality: str,
                             target_year: int, target_doy: int) -> np.ndarray | None:
    """
    Load L3 tokens from zarr for the closest date to target_doy within +/-60 days.
    Returns (196, 768) float16 array or None if unavailable.
    """
    import zarr

    zarr_dir = ZARR_ROOT / category / station_key
    try:
        zg = zarr.open(str(zarr_dir), mode="r")
    except Exception:
        return None

    # Prefer memmap .npy if available
    npy_key  = f"{modality}_l3"
    json_key = f"{npy_key}.json"
    npy_path = zarr_dir / f"{npy_key}.npy"
    json_path = zarr_dir / f"{json_key}"

    if npy_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text())
        arr  = np.memmap(str(npy_path), dtype="float16",
                         mode="r", shape=tuple(meta["shape"]))
        # Find date index
        try:
            date_ints = np.array(zg[modality]["date_ints"])
        except Exception:
            return None
        target_dt   = datetime(target_year, 1, 1) + timedelta(days=target_doy - 1)
        target_int  = int(target_dt.strftime("%Y%m%d"))
        diffs       = np.abs(date_ints - target_int)
        best_idx    = int(np.argmin(diffs))
        if diffs[best_idx] > 600:    # > ~2 months
            return None
        return np.array(arr[best_idx])    # force copy from memmap

    # Fallback: zarr
    try:
        l3       = zg[modality]["l3"]
        dates_z  = zg[modality]["date_ints"][:]
        target_dt  = datetime(target_year, 1, 1) + timedelta(days=target_doy - 1)
        target_int = int(target_dt.strftime("%Y%m%d"))
        diffs      = np.abs(dates_z - target_int)
        best_idx   = int(np.argmin(diffs))
        if diffs[best_idx] > 600:
            return None
        return l3[best_idx].astype(np.float16)
    except Exception as e:
        print(f"    L3 zarr load failed ({modality}): {e}")
        return None


@torch.no_grad()
def infer_sm_map(model, dataset, station_key: str, target_year: int,
                 target_doy: int, device) -> np.ndarray | None:
    """Run inference for a specific (station, year, doy) sample. Returns (n_depths, 224, 224)."""
    SROW = SoilMoistureModel.STATION_ROW
    SCOL = SoilMoistureModel.STATION_COL

    idx_cands = [i for i, s in enumerate(dataset.samples)
                 if s["station_key"] == station_key
                 and s["year"] == target_year
                 and abs(s["doy"] - target_doy) < 10]
    if not idx_cands:
        # Fall back to any sample from that year
        idx_cands = [i for i, s in enumerate(dataset.samples)
                     if s["station_key"] == station_key and s["year"] == target_year]
    if not idx_cands:
        return None

    idx  = min(idx_cands, key=lambda i: abs(dataset.samples[i]["doy"] - target_doy))
    item = dataset[idx]
    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v]
             for k, v in item.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mu = model(batch)
    return mu[0].float().cpu().numpy()   # (n_depths, 224, 224)


def make_satellite_sm_figure(station_key: str, rank_label: str,
                              meta_row: dict, spatial_sm: np.ndarray,
                              s2_img: np.ndarray | None, s1_img: np.ndarray | None,
                              target_date: datetime, out_dir: Path):
    """5-panel figure: [S2 | S1 | SM 0-10 | SM 10-30 | SM 30-100]"""
    n_depths = len(SM_DEPTHS)
    n_panels = 2 + n_depths    # S2 + S1 + SM depths

    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 3.2, 3.8))

    station_short = station_key.replace("ISMN_", "").replace("_", " / ", 1)
    igbp    = meta_row.get("IGBP",         "—")
    climate = meta_row.get("koppen_geiger", "—")
    ubrmse  = meta_row.get("ubRMSE_0_10",  float("nan"))

    fig.suptitle(
        f"{rank_label} station: {station_short}  ·  {target_date.strftime('%Y-%m-%d')}"
        f"  ·  IGBP: {igbp}  ·  Climate: {climate}  ·  ubRMSE₀₋₁₀ = {ubrmse:.3f}",
        fontsize=9, fontweight="bold", y=1.02,
    )

    def _imshow_panel(ax, img, title, cmap="viridis", vmin=None, vmax=None, cb_label=None):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(title, fontsize=8.5)
        ax.axis("off")
        # Cyan crosshair at station pixel
        ax.axhline(112, color="cyan", lw=0.8, ls="--", alpha=0.7)
        ax.axvline(112, color="cyan", lw=0.8, ls="--", alpha=0.7)
        if cb_label:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.85)
            cbar.set_label(cb_label, fontsize=7)
            cbar.ax.tick_params(labelsize=6.5)
        return im

    # Panel 0: S2 pseudo-RGB
    if s2_img is not None:
        axes[0].imshow(s2_img, interpolation="bilinear")
        axes[0].set_title("S2 (L3 PCA pseudo-RGB)", fontsize=8.5)
    else:
        axes[0].set_facecolor("#cccccc")
        axes[0].text(0.5, 0.5, "S2 unavailable", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=8)
        axes[0].set_title("S2 pseudo-RGB", fontsize=8.5)
    axes[0].axis("off")
    axes[0].axhline(112, color="cyan", lw=0.8, ls="--", alpha=0.7)
    axes[0].axvline(112, color="cyan", lw=0.8, ls="--", alpha=0.7)

    # Panel 1: S1 pseudo-SAR (single channel → greys)
    if s1_img is not None:
        s1_grey = s1_img.mean(axis=2)    # average PCA channels → grayscale
        axes[1].imshow(s1_grey, cmap="Greys_r", interpolation="bilinear")
        axes[1].set_title("S1 (L3 PCA pseudo-SAR)", fontsize=8.5)
    else:
        axes[1].set_facecolor("#cccccc")
        axes[1].text(0.5, 0.5, "S1 unavailable", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=8)
        axes[1].set_title("S1 pseudo-SAR", fontsize=8.5)
    axes[1].axis("off")
    axes[1].axhline(112, color="cyan", lw=0.8, ls="--", alpha=0.7)
    axes[1].axvline(112, color="cyan", lw=0.8, ls="--", alpha=0.7)

    # Panels 2–4: SM depth maps
    sm_cmaps = ["YlOrBr", "BrBG", "PuBu"]
    for i, (depth, cmap) in enumerate(zip(SM_DEPTHS, sm_cmaps)):
        _imshow_panel(axes[2 + i], spatial_sm[i],
                      f"SM {depth} cm  (predicted)",
                      cmap=cmap, vmin=0.0, vmax=0.5,
                      cb_label="SM (m³/m³)")

    fig.tight_layout()
    fname = out_dir / f"{station_key}_{target_date.strftime('%Y%m%d')}_{rank_label}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="baseline_huber")
    parser.add_argument("--ckpt",     default="best.pt")
    parser.add_argument("--n",        type=int, default=5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csv_path = Path("meeting_output/per_station_oos.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run evaluate_splits.py first.")
    rank_df = pd.read_csv(csv_path).dropna(subset=["ubRMSE_0_10"]).sort_values("ubRMSE_0_10")
    best_df  = rank_df.head(args.n)
    worst_df = rank_df.tail(args.n).iloc[::-1]
    selected = (pd.concat([best_df.assign(_rank="BEST"),
                            worst_df.assign(_rank="WORST")])
                .reset_index(drop=True))

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

    for _, row in selected.iterrows():
        station    = row["station_key"]
        rank_label = row["_rank"]
        print(f"\n[{rank_label}] {station}")

        # Choose target year: median year in station's data
        station_samples = [s for s in ds.samples if s["station_key"] == station]
        if not station_samples:
            print("  No samples in dataset — skipping")
            continue
        years_avail = sorted({s["year"] for s in station_samples})
        target_year = years_avail[len(years_avail) // 2]
        target_doy  = 182    # ~July 1st

        target_date = datetime(target_year, 1, 1) + timedelta(days=target_doy - 1)
        print(f"  Target: {target_date.strftime('%Y-%m-%d')}")

        # Build pseudo-RGB images from zarr L3 tokens
        # Determine category
        meta_row = row.to_dict()
        has_sm   = True   # we're evaluating sm_only stations
        category = "sm_only"

        print("  Loading S2 L3 tokens…")
        s2_tokens = load_l3_tokens_for_date(station, category, "s2",
                                             target_year, target_doy)
        s2_img = _pca_pseudo_image(s2_tokens) if s2_tokens is not None else None
        print(f"  S2: {'ok' if s2_img is not None else 'unavailable'}")

        print("  Loading S1 L3 tokens…")
        s1_tokens = load_l3_tokens_for_date(station, category, "s1_asc",
                                             target_year, target_doy)
        if s1_tokens is None:
            s1_tokens = load_l3_tokens_for_date(station, category, "s1_desc",
                                                 target_year, target_doy)
        s1_img = _pca_pseudo_image(s1_tokens) if s1_tokens is not None else None
        print(f"  S1: {'ok' if s1_img is not None else 'unavailable'}")

        print("  Running SM inference…")
        sm_map = infer_sm_map(model, ds, station, target_year, target_doy, device)
        if sm_map is None:
            print("  Inference failed — skipping")
            continue

        make_satellite_sm_figure(
            station_key  = station,
            rank_label   = rank_label,
            meta_row     = meta_row,
            spatial_sm   = sm_map,
            s2_img       = s2_img,
            s1_img       = s1_img,
            target_date  = target_date,
            out_dir      = OUT_DIR,
        )

    print(f"\nAll satellite+SM figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
