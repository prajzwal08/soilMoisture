"""
Spatial figure: rows = selected dates, cols = [S2 pseudo-RGB | S1 pseudo-SAR | SM 0-10 | SM 10-30 | SM 30-100].

All panels are 224×224 pixels at equal physical size.
S2/S1 pseudo-images are reconstructed from L3 tokens via PCA (14×14 → bicubic 224×224).

Requires:
    meeting_output/per_station_oos.csv   (from evaluate_splits.py)

Outputs:
    meeting_output/spatial/{station}_spatial.png

Usage:
    python plot_spatial_sm_meeting.py [--run-name baseline_huber] [--n 5]
    python plot_spatial_sm_meeting.py --station DWDBerlin-Spaeth
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zarr
from PIL import Image
from torch.utils.data import DataLoader, Subset

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "nature"])
except ImportError:
    plt.rcParams.update({"font.size": 9})

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel
from ckpt_utils import load_checkpoint

CKPT_ROOT      = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")
SPLITS_CSV     = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
ERA5_STATS     = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json")
ZARR_ROOT      = Path("/gpfs/scratch1/shared/pkhanal/zarr")   # token zarr (model input)
SAT_ZARR_ROOT  = Path("/projects/prjs1968/satellite_zarr")    # raw satellite imagery
OUT_DIR        = Path("meeting_output/spatial")

SAMPLE_DOYS = [30, 90, 150, 210, 270, 330]

DEPTH_CMAPS  = {"0-10": "YlOrBr", "10-30": "YlGnBu", "30-100": "PuBu"}
DEPTH_LABELS = {"0-10": "SM  0–10 cm", "10-30": "SM 10–30 cm", "30-100": "SM 30–100 cm"}
SM_VMIN, SM_VMAX = 0.0, 0.5

SROW = SoilMoistureModel.STATION_ROW
SCOL = SoilMoistureModel.STATION_COL


# ── Real satellite image helpers ──────────────────────────────────────────────

# S2 band indices in the 12-band array (B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12)
_S2_RED, _S2_GREEN, _S2_BLUE = 3, 2, 1   # B4, B3, B2

def _sat_zarr_path(station_key: str) -> Path:
    """Flat layout: /projects/prjs1968/satellite_zarr/{station_key}.zarr"""
    return SAT_ZARR_ROOT / f"{station_key}.zarr"


def _closest_idx(dates_bytes, target_date: datetime, max_gap_days: int = 30):
    """Return index of the closest acquisition, or None if beyond max_gap_days."""
    acq = pd.to_datetime([d.decode() if isinstance(d, bytes) else str(d)
                          for d in dates_bytes], format="%Y%m%d")
    diffs = np.abs((acq - pd.Timestamp(target_date)).days)
    idx   = int(np.argmin(diffs))
    return idx if diffs[idx] <= max_gap_days else None


def get_s2_rgb(station_key: str, target_date: datetime,
               max_gap_days: int = 30) -> np.ndarray | None:
    """
    Load S2 true-colour (B4/B3/B2) 224×224 image closest to target_date.
    Returns (224, 224, 3) uint8, or None if unavailable.
    """
    path = _sat_zarr_path(station_key)
    if not path.exists():
        return None
    try:
        zg = zarr.open_group(str(path), mode="r")
    except Exception:
        return None
    if "s2" not in zg:
        return None

    idx = _closest_idx(zg["s2/dates"][:], target_date, max_gap_days)
    if idx is None:
        return None

    img = zg["s2/data"][idx].astype(np.float32)   # (12, 224, 224)
    rgb = np.stack([img[_S2_RED], img[_S2_GREEN], img[_S2_BLUE]], axis=-1)  # (224,224,3)

    # 2-98 percentile stretch per channel for a natural-looking result
    out = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, :, c], (2, 98))
        out[:, :, c] = np.clip((rgb[:, :, c] - p2) / max(p98 - p2, 1) * 255, 0, 255).astype(np.uint8)
    return out


def get_s1_vv(station_key: str, target_date: datetime,
              max_gap_days: int = 30) -> np.ndarray | None:
    """
    Load S1 VV channel 224×224 image closest to target_date.
    Returns (224, 224) float32 (dB, clipped to [-20, 0]), or None if unavailable.
    Picks whichever orbit (asc/desc) has more acquisitions.
    """
    path = _sat_zarr_path(station_key)
    if not path.exists():
        return None
    try:
        zg = zarr.open_group(str(path), mode="r")
    except Exception:
        return None

    n_asc  = zg["s1_asc/data"].shape[0]  if "s1_asc"  in zg else 0
    n_desc = zg["s1_desc/data"].shape[0] if "s1_desc" in zg else 0
    mod    = "s1_asc" if n_asc >= n_desc else "s1_desc"
    if mod not in zg:
        return None

    idx = _closest_idx(zg[f"{mod}/dates"][:], target_date, max_gap_days)
    if idx is None:
        return None

    return np.clip(zg[f"{mod}/data"][idx, 0].astype(np.float32), -20.0, 0.0)


def get_dem(station_key: str) -> np.ndarray | None:
    """Load DEM patch (224, 224) float32 in metres, or None."""
    path = _sat_zarr_path(station_key)
    if not path.exists():
        return None
    try:
        zg = zarr.open_group(str(path), mode="r")
        return zg["dem/data"][0].astype(np.float32)   # (224, 224)
    except Exception:
        return None


def get_lulc(station_key: str) -> np.ndarray | None:
    """Load most-recent LULC patch (224, 224) uint8, or None."""
    path = _sat_zarr_path(station_key)
    if not path.exists():
        return None
    try:
        zg  = zarr.open_group(str(path), mode="r")
        return zg["lulc/data"][-1].astype(np.uint8)   # most recent year, (224, 224)
    except Exception:
        return None


# ── Model inference ───────────────────────────────────────────────────────────

@torch.no_grad()
def infer_spatial_for_dates(model, dataset, station_key: str,
                             target_year: int, target_doys: list,
                             device) -> list:
    """
    For each DOY in target_doys find the closest sample, run inference.
    Returns list of dicts: {date, sm_map (n_depths,224,224), obs_sm (n_depths,), doy}
    or None for missing dates.
    """
    cands = {
        dataset.samples[i]["doy"]: i
        for i, s in enumerate(dataset.samples)
        if s["station_key"] == station_key and s["year"] == target_year
    }
    if not cands:
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
        diffs    = np.abs(available_doys - doy)
        best_doy = int(available_doys[np.argmin(diffs)])
        if diffs.min() > 45:
            results.append(None)
            continue

        idx  = cands[best_doy]
        item = dataset[idx]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v]
                 for k, v in item.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(batch)
        sm_map  = mu[0].float().cpu().numpy()   # (n_depths, 224, 224)
        obs_sm  = item["label"].numpy()          # (n_depths,)
        date_dt = datetime(target_year, 1, 1) + timedelta(days=best_doy - 1)
        results.append({"date": date_dt, "sm_map": sm_map, "obs_sm": obs_sm,
                        "doy": best_doy, "target_doy": doy})
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_spatial_grid(station_key: str, rank_label: str, meta_row: dict,
                      date_results: list, out_dir: Path):
    """
    Rows = valid dates, Cols = [S2 RGB | S1 VV | DEM | LULC | SM 0-10 | SM 10-30 | SM 30-100].
    All panels are 224×224 pixels at equal physical size.
    DEM and LULC are static — same image shown in every row.
    """
    valid = [(i, r) for i, r in enumerate(date_results) if r is not None]
    if not valid:
        print(f"  No valid inference results for {station_key} — skipping")
        return

    n_rows   = len(valid)
    n_depths = len(SM_DEPTHS)
    n_cols   = 4 + n_depths   # S2, S1, DEM, LULC, then one per depth

    igbp    = meta_row.get("IGBP",         "")
    climate = meta_row.get("koppen_geiger", "")
    ubrmse  = meta_row.get("ubRMSE_0_10",  float("nan"))
    station_short = station_key.replace("ISMN_", "").replace("_", " / ", 1)

    # Pre-load static layers once
    dem_img  = get_dem(station_key)
    lulc_img = get_lulc(station_key)

    panel_in = 2.3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_in + 0.5, n_rows * panel_in + 0.9),
        squeeze=False,
    )

    rank_str   = f"[{rank_label}]  " if rank_label else ""
    try:
        ubrmse_str = f"  ubRMSE₀₋₁₀ = {ubrmse:.3f}"
    except (ValueError, TypeError):
        ubrmse_str = ""
    fig.suptitle(
        f"{rank_str}{station_short}   IGBP: {igbp}   Climate: {climate}{ubrmse_str}",
        fontsize=9.5, fontweight="bold",
    )

    col_headers = ["S2  RGB", "S1  VV  (dB)", "DEM  (m)", "LULC"] + \
                  [DEPTH_LABELS[d] for d in SM_DEPTHS]
    for col_idx, hdr in enumerate(col_headers):
        axes[0][col_idx].set_title(hdr, fontsize=8, pad=4)

    for row_idx, (_, res) in enumerate(valid):
        date_str = res["date"].strftime("%Y-%m-%d")
        axes[row_idx][0].set_ylabel(date_str, fontsize=7.5, labelpad=4)

        # ── S2 true-colour RGB ────────────────────────────────────────────────
        ax = axes[row_idx][0]
        s2 = get_s2_rgb(station_key, res["date"])
        if s2 is not None:
            ax.imshow(s2, interpolation="bilinear", aspect="equal")
        else:
            _no_data(ax, "no S2")
        _crosshair(ax)

        # ── S1 VV greyscale ───────────────────────────────────────────────────
        ax   = axes[row_idx][1]
        s1   = get_s1_vv(station_key, res["date"])
        if s1 is not None:
            im_s1 = ax.imshow(s1, cmap="gray", vmin=-20, vmax=0,
                              interpolation="bilinear", aspect="equal")
            if row_idx == n_rows - 1:
                _hcbar(ax, im_s1, "dB")
        else:
            _no_data(ax, "no S1")
        _crosshair(ax)

        # ── DEM (static) ──────────────────────────────────────────────────────
        ax = axes[row_idx][2]
        if dem_img is not None:
            im_dem = ax.imshow(dem_img, cmap="terrain",
                               interpolation="bilinear", aspect="equal")
            if row_idx == n_rows - 1:
                _hcbar(ax, im_dem, "m")
        else:
            _no_data(ax, "no DEM")
        _crosshair(ax)

        # ── LULC (static) ─────────────────────────────────────────────────────
        ax = axes[row_idx][3]
        if lulc_img is not None:
            ax.imshow(lulc_img, cmap="tab20", interpolation="nearest",
                      vmin=0, vmax=20, aspect="equal")
        else:
            _no_data(ax, "no LULC")
        _crosshair(ax)

        # ── SM depth maps ─────────────────────────────────────────────────────
        for d_idx, depth in enumerate(SM_DEPTHS):
            ax  = axes[row_idx][4 + d_idx]
            sm  = res["sm_map"][d_idx]
            obs = res["obs_sm"][d_idx]

            im_sm = ax.imshow(sm, cmap=DEPTH_CMAPS[depth],
                              vmin=SM_VMIN, vmax=SM_VMAX,
                              interpolation="bilinear", aspect="equal")
            _crosshair(ax)

            if not np.isnan(obs):
                ax.text(SCOL + 7, SROW - 7, f"obs={obs:.2f}",
                        fontsize=5.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="k", alpha=0.45, lw=0))

            if row_idx == n_rows - 1:
                _hcbar(ax, im_sm, "m³/m³")

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fname = out_dir / f"{station_key}_spatial.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def _crosshair(ax):
    ax.axhline(SROW, color="cyan", lw=0.7, ls="--", alpha=0.75)
    ax.axvline(SCOL, color="cyan", lw=0.7, ls="--", alpha=0.75)
    ax.plot(SCOL, SROW, "c^", ms=3.5, zorder=5)


def _no_data(ax, msg: str):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=7, color="grey")
    ax.set_facecolor("#111111")


def _hcbar(ax, mappable, label: str):
    """Thin horizontal colourbar below the axis."""
    cax = ax.inset_axes([0, -0.06, 1, 0.04])
    cb  = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=5.5, labelpad=1)
    cb.ax.tick_params(labelsize=5)


# ── Station selection ─────────────────────────────────────────────────────────

def _make_key(r):
    if str(r.get("source_network", "")) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def get_zarr_path(station_key: str, splits_df: pd.DataFrame) -> Path:
    """Resolve zarr directory path from station_key using splits metadata."""
    hits = splits_df[splits_df.apply(_make_key, axis=1) == station_key]
    if hits.empty:
        # fallback: assume sm_only
        return ZARR_ROOT / "sm_only" / station_key
    r   = hits.iloc[0]
    sm  = str(r.get("has_soil_moisture", "False")).lower() == "true"
    fl  = str(r.get("has_flux",          "False")).lower() == "true"
    cat = "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")
    return ZARR_ROOT / cat / station_key


def resolve_stations(args, ds) -> list:
    if args.station:
        all_keys = sorted({s["station_key"] for s in ds.samples})
        selected = []
        for name in args.station:
            matches = [k for k in all_keys if name in k]
            if not matches:
                print(f"  Warning: no station matching '{name}' — skipping")
                continue
            selected.append({"station_key": matches[0], "_rank": ""})
        return selected

    csv_path = Path("meeting_output/per_station_oos.csv")
    if csv_path.exists():
        rank_df  = pd.read_csv(csv_path).dropna(subset=["ubRMSE_0_10"]).sort_values("ubRMSE_0_10")
        best_df  = rank_df.head(args.n).assign(_rank="BEST")
        worst_df = rank_df.tail(args.n).iloc[::-1].assign(_rank="WORST")
        return pd.concat([best_df, worst_df], ignore_index=True).to_dict("records")

    print(f"No CSV found — sampling {args.n} random stations from dataset")
    all_keys = sorted({s["station_key"] for s in ds.samples})
    rng      = np.random.default_rng(42)
    sampled  = rng.choice(all_keys, size=min(args.n, len(all_keys)), replace=False)
    return [{"station_key": k, "_rank": ""} for k in sampled]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="baseline_huber")
    parser.add_argument("--ckpt",     default="best.pt")
    parser.add_argument("--n",        type=int, default=5)
    parser.add_argument("--year",     type=int, default=None)
    parser.add_argument("--station",  nargs="+", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = CKPT_ROOT / args.run_name / args.ckpt
    model, cfg, epoch = load_checkpoint(ckpt_path, device)

    splits_df = pd.read_csv(SPLITS_CSV)

    # When --station is given, filter splits CSV to just those rows so the
    # dataset only opens memmaps for the requested stations (fast init).
    if args.station:
        keys_mask = splits_df.apply(_make_key, axis=1)
        matched   = splits_df[keys_mask.apply(
            lambda k: any(s in k for s in args.station))]
        if matched.empty:
            print(f"No stations matched {args.station} in splits CSV — exiting")
            return
        import tempfile, os
        tmp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        matched.to_csv(tmp_csv.name, index=False)
        tmp_csv.close()
        active_splits_csv = tmp_csv.name
        active_split_filter = None   # matched rows may be any split
        print(f"Station filter: {len(matched)} row(s) → temp splits CSV")
    else:
        active_splits_csv  = str(SPLITS_CSV)
        active_split_filter = ["oos"]
        tmp_csv = None

    ds = SoilMoistureDataset(
        splits_csv      = active_splits_csv,
        era5_stats_path = str(ERA5_STATS),
        years           = list(range(2016, 2024)),
        category_filter = cfg.get("category_filter", ["sm_only"]),
        split_filter    = active_split_filter,
        training        = False,
        use_mmap        = True,
    )
    print(f"Dataset: {len(ds):,} samples")

    selected = resolve_stations(args, ds)
    if not selected:
        print("No stations to plot — exiting")
        return

    for row in selected:
        station    = row["station_key"]
        rank_label = row.get("_rank", "")
        sat_exists = _sat_zarr_path(station).exists()
        print(f"\n[{rank_label or 'USER'}] {station}  sat_zarr={'OK' if sat_exists else 'MISSING'}")

        if args.year is not None:
            target_year = args.year
        else:
            yrs     = sorted({s["year"] for s in ds.samples if s["station_key"] == station})
            oos_yrs = [y for y in yrs if 2016 <= y <= 2022]
            pool    = oos_yrs if oos_yrs else yrs
            if not pool:
                print("  No samples — skipping")
                continue
            target_year = pool[len(pool) // 2]

        print(f"  Year: {target_year}  DOYs: {SAMPLE_DOYS}")
        date_results = infer_spatial_for_dates(
            model, ds, station, target_year, SAMPLE_DOYS, device)

        n_valid = sum(1 for r in date_results if r is not None)
        print(f"  Valid dates: {n_valid}/{len(SAMPLE_DOYS)}")

        plot_spatial_grid(station, rank_label or "USER", row,
                          date_results, OUT_DIR)

    print(f"\nAll spatial figures saved to {OUT_DIR}/")
    if tmp_csv is not None:
        import os
        os.unlink(tmp_csv.name)


if __name__ == "__main__":
    main()
