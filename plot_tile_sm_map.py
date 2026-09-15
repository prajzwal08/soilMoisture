"""One tile at 160 m: static inputs, the predicted SM field through the year, six stations.

§35.33.6 found the model predicts nearly the same soil moisture at every station in a tile.
This shows the field it actually paints, rather than inferring it from six point readouts.

`token_sel='all'` makes the dataset emit all 196 patches instead of only the station's, and
the patchwise model is patch-agnostic -- `_build_patch_seq` runs per patch with shared
weights -- so one forward pass returns (196, n_depths): a real 14x14 map at 160 m over the
2.24 km tile. No model change needed; this is the §28.9 readout taken from the input side
instead of the head.

The six TxSON stations inside ISMN_TxSON_CR200-18 land on tokens 105, 62, 100, 20, 44, 172,
so all six series come from ONE forward pass on ONE tile -- the comparison §26 asked for.

Three figures:
    {tile}_inputs.{png,pdf}   DEM / LULC / soil / S1 VV at 10 m, with the 160 m token grid
    {tile}_sm_maps.{png,pdf}  predicted SM at 160 m, --per-year dates per year, ONE shared
                              colour scale across every panel
    {tile}_series.{png,pdf}   six panels, one station each, predicted vs observed + metrics

Usage
-----
    python plot_tile_sm_map.py --tile ISMN_TxSON_CR200-18 --years 2019 2020
    python plot_tile_sm_map.py --cache-only          # re-plot, no GPU
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import BoundaryNorm, ListedColormap

warnings.filterwarnings("ignore")

REPO       = Path(__file__).resolve().parent
SAT_ZARR   = Path("/projects/prjs1968/satellite_zarr")
TOK_ZARR   = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SPLITS     = REPO / "csvs" / "station_splits.csv"
CKPT_ROOT  = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")
ERA5_STATS = REPO / "csvs" / "era5_stats.json"

PATCH_PX, TOKEN_PX, RES_M, GRID = 224, 16, 10, 14
SM_DEPTHS = ["0-10", "10-30", "30-100"]

# Stored values are TerraMind indices, NOT raw ESRI classes (download_s1_lulc_mpc.py:50-54).
LULC_NAMES = {0: "NoData", 1: "Water", 2: "Trees", 3: "Flooded veg.", 4: "Crops",
              5: "Built area", 6: "Bare ground", 7: "Snow/Ice", 8: "Clouds",
              9: "Rangeland"}
LULC_COLOURS = {0: "#ffffff", 1: "#1f6cb0", 2: "#1a7a3a", 3: "#4c9fd4", 4: "#e8c34a",
                5: "#c43c3c", 6: "#cfc6b8", 7: "#f0f0f0", 8: "#dddddd", 9: "#a8bf6a"}
STATION_COLOURS = ["#111111", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
INK, MUTED, GRIDC = "#1a1a1a", "#6b6b6b", "#d9d9d9"


def _dstr(a) -> list[str]:
    return [d.decode() if isinstance(d, bytes) else str(d) for d in a]


def token_of(row: int, col: int) -> int:
    """Pixel (row, col) in the 224x224 patch -> index in the 14x14 token grid."""
    return int(row // TOKEN_PX) * GRID + int(col // TOKEN_PX)


# ---------------------------------------------------------------------------
# Stage 1 -- inference over all 196 patches
# ---------------------------------------------------------------------------
def predict_tile(tile: str, run_name: str, ckpt: str, years: list[int],
                 cache: Path, batch_size: int = 4) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from dataset import SoilMoistureDataset
    from ckpt_utils import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, epoch = load_checkpoint(CKPT_ROOT / run_name / ckpt, device)
    print(f"  checkpoint epoch {epoch}, trained token_sel={cfg.get('token_sel')!r}")

    splits = pd.read_csv(SPLITS)
    key = splits.apply(
        lambda r: (f"ISMN_{r['network']}_{r['station_name']}"
                   if str(r["source_network"]) == "ISMN"
                   else f"{r['source_network']}_{r['station_id']}"), axis=1)
    sub = splits[key == tile]
    if sub.empty:
        raise SystemExit(f"{tile} not in {SPLITS}")
    split_name = str(sub.iloc[0]["split"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = cache.parent / f"_tile_{tile}.csv"
    sub.to_csv(tmp_csv, index=False)
    print(f"  {tile} is split={split_name}")

    # token_sel='all' is the point: K=196 instead of 1. It is NOT what the model trained
    # with, and that is fine -- the patch blocks share weights across k, so patch 20 is
    # scored by exactly the machinery that scored patch 105 in training. Only the number
    # of patches asked for changes.
    ds = SoilMoistureDataset(
        splits_csv      = str(tmp_csv),
        era5_stats_path = str(ERA5_STATS),
        years           = years,
        category_filter = cfg.get("category_filter", ["sm_only"]),
        split_filter    = [split_name],
        training        = False,
        token_sel       = "all",
        shm_dir         = None,   # the staged cache is narrowed to K=1 and cannot serve this
    )
    if len(ds) == 0:
        raise SystemExit(f"no samples for {tile} in {years}")
    print(f"  {len(ds)} dates")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    maps, ys, dys = [], [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                mu = model(batch)                        # (B, 196, D)
            if mu.ndim != 3 or mu.shape[1] != GRID * GRID:
                raise SystemExit(f"expected (B,{GRID*GRID},D), got {tuple(mu.shape)}")
            maps.append(mu.float().cpu().numpy())
            ys.append(np.asarray(batch["year"]))
            dys.append(np.asarray(batch["doy"]))

    out = dict(maps=np.concatenate(maps), year=np.concatenate(ys),
               doy=np.concatenate(dys))
    np.savez_compressed(cache, **out)
    tmp_csv.unlink(missing_ok=True)
    print(f"  cached {out['maps'].shape} -> {cache}")
    return out


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def mark_px(ax, st: pd.DataFrame, labels: bool = True):
    """Station markers + the 160 m token grid, on a 224x224 (10 m) panel."""
    for k in range(TOKEN_PX, PATCH_PX, TOKEN_PX):
        ax.axhline(k - .5, color="white", lw=.3, alpha=.30)
        ax.axvline(k - .5, color="white", lw=.3, alpha=.30)
    for i, (_, r) in enumerate(st.iterrows()):
        ax.plot(r["col"], r["row"], "o", ms=6, mfc=STATION_COLOURS[i % 6],
                mec="white", mew=1.0, zorder=5)
        if labels:
            ax.annotate(r["station"].replace("ISMN_TxSON_", ""), (r["col"], r["row"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=6,
                        color="white", zorder=6)
    ax.set_xticks([]); ax.set_yticks([])


def mark_tok(ax, st: pd.DataFrame):
    """The same stations on a 14x14 (160 m) panel."""
    for i, (_, r) in enumerate(st.iterrows()):
        ax.plot(r["col"] / TOKEN_PX - .5, r["row"] / TOKEN_PX - .5, "o", ms=6,
                mfc=STATION_COLOURS[i % 6], mec="white", mew=1.0, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])


def hillshade(dem, az=315.0, alt=45.0):
    gy, gx = np.gradient(dem, RES_M)
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    az_r, alt_r = np.radians(360.0 - az + 90.0), np.radians(alt)
    return np.clip(np.sin(alt_r) * np.cos(slope)
                   + np.cos(alt_r) * np.sin(slope) * np.cos(az_r - aspect), 0, 1)


def pick_dates(date: pd.Series, years: list[int], per_year: int) -> list[int]:
    """`per_year` dates spread evenly through EACH year, so the panels sample the
    seasonal cycle rather than clustering wherever acquisitions happen to be dense."""
    picks = []
    for y in years:
        idx = np.where(date.dt.year.to_numpy() == y)[0]
        if len(idx) == 0:
            continue
        doy = date.iloc[idx].dt.dayofyear.to_numpy()
        for target in np.linspace(1, 365, per_year + 2)[1:-1]:
            picks.append(int(idx[np.argmin(np.abs(doy - target))]))
    return sorted(set(picks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile",     default="ISMN_TxSON_CR200-18")
    ap.add_argument("--run-name", default="pw_stage2a_L3")
    ap.add_argument("--ckpt",     default="best.pt")
    ap.add_argument("--years",    type=int, nargs=2, default=[2019, 2020])
    ap.add_argument("--per-year", type=int, default=4,
                    help="SM map panels per year (>=4 samples the seasonal cycle)")
    ap.add_argument("--readouts", default="csvs/txson_readouts.csv")
    ap.add_argument("--preds",
                    default="eval_output/pw_stage2a_L3/predictions_val.parquet",
                    help="observed series come from this parquet's obs column")
    ap.add_argument("--soil-channel", type=int, default=3)
    ap.add_argument("--depth",    default="0-10", choices=SM_DEPTHS)
    ap.add_argument("--out-dir",  default="figures/eval/pw_stage2a_L3/tile")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--dpi",      type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"{args.tile}_patchmap.npz"
    years = list(range(args.years[0], args.years[1] + 1))
    di = SM_DEPTHS.index(args.depth)

    ro = pd.read_csv(args.readouts)
    st = ro[ro["tile"] == args.tile].copy().sort_values("dist_m").reset_index(drop=True)
    if st.empty:
        raise SystemExit(f"no readout rows for {args.tile} in {args.readouts}")
    st["token"] = [token_of(r, c) for r, c in zip(st["row"], st["col"])]
    print(f"{args.tile}: {len(st)} stations, tokens {list(st['token'])}")

    if args.cache_only or cache.exists():
        z = np.load(cache); pred = {k: z[k] for k in z.files}
        print(f"  loaded cache {pred['maps'].shape}")
    else:
        pred = predict_tile(args.tile, args.run_name, args.ckpt, years, cache)

    maps = pred["maps"][:, :, di]                                  # (N, 196)
    date = (pd.to_datetime(pd.Series(pred["year"]).astype(str) + "-01-01")
            + pd.to_timedelta(pred["doy"] - 1, unit="D"))
    o = np.argsort(date.values)
    maps, date = maps[o], date.iloc[o].reset_index(drop=True)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.dpi": args.dpi, "savefig.bbox": "tight"})

    raw = zarr.open_group(str(SAT_ZARR / f"{args.tile}.zarr"), mode="r")

    # ── FIGURE 1: the four static inputs ─────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.8), constrained_layout=True)

    dem = raw["dem/data"][0].astype(np.float32)
    axes[0].imshow(hillshade(dem), cmap="gray", alpha=.55)
    im = axes[0].imshow(dem, cmap="terrain", alpha=.65)
    fig.colorbar(im, ax=axes[0], shrink=.8, pad=.02)
    axes[0].set_title(f"DEM  {dem.min():.0f}–{dem.max():.0f} m  (sd {dem.std():.1f})",
                      loc="left")
    mark_px(axes[0], st)

    yrs = list(raw["lulc/years"][:])
    li = int(np.argmin([abs(int(y) - years[-1]) for y in yrs]))
    lulc = raw["lulc/data"][li].astype(np.uint8)
    present = sorted(np.unique(lulc).tolist())
    cmap = ListedColormap([LULC_COLOURS.get(v, "#888888") for v in present])
    axes[1].imshow(np.searchsorted(present, lulc), cmap=cmap,
                   norm=BoundaryNorm(np.arange(len(present) + 1) - .5, len(present)))
    frac = {v: float((lulc == v).mean()) for v in present}
    # Two classes, not three: at this panel width a third entry overruns into the soil
    # panel's title. The rest of the breakdown is on the colour patches themselves.
    lab = "  ".join(f"{LULC_NAMES.get(v, v)} {100*frac[v]:.0f}%"
                    for v in sorted(present, key=lambda v: -frac[v])[:2])
    axes[1].set_title(f"Land cover {yrs[li]} · {lab}", loc="left", fontsize=7)
    mark_px(axes[1], st)

    try:
        s = zarr.open_consolidated(str(TOK_ZARR / "sm_only" / args.tile))["soil"][
            args.soil_channel].astype(np.float32)
        idx = np.clip((np.arange(PATCH_PX) * s.shape[0] / PATCH_PX).astype(int),
                      0, s.shape[0] - 1)
        soil = s[np.ix_(idx, idx)]
        im = axes[2].imshow(soil, cmap="YlOrBr")
        fig.colorbar(im, ax=axes[2], shrink=.8, pad=.02)
        axes[2].set_title(f"Soil ch{args.soil_channel} (SOC 0–30 cm)  "
                          f"{soil.min():.0f}–{soil.max():.0f}", loc="left")
    except Exception as e:
        axes[2].text(.5, .5, f"no soil layer\n({e})", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=7, color=MUTED)
        axes[2].set_title("Soil", loc="left")
    mark_px(axes[2], st)

    s1d = _dstr(raw["s1_asc/dates"][:])
    j = int(np.argmin([abs(int(d) - int(f"{years[0]}0101")) for d in s1d]))
    vv = np.clip(raw["s1_asc/data"][j, 0].astype(np.float32), -20, 0)
    im = axes[3].imshow(vv, cmap="gray")
    fig.colorbar(im, ax=axes[3], shrink=.8, pad=.02)
    axes[3].set_title(f"S1 VV (dB)  {s1d[j]}", loc="left")
    mark_px(axes[3], st)

    fig.suptitle(f"{args.tile} — static inputs at 10 m over the 2.24 km tile.  "
                 f"White grid = the 14×14 TerraMind tokens the model predicts on (160 m).",
                 fontsize=9)
    for ext in ("png", "pdf"):
        p = out_dir / f"{args.tile}_inputs.{ext}"; fig.savefig(p); print(f"wrote {p}")
    plt.close(fig)

    # ── FIGURE 2: predicted SM field through the year ────────────────────
    picks = pick_dates(date, years, args.per_year)
    ncol = args.per_year
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.5 * nrow),
                             constrained_layout=True, squeeze=False)
    # ONE colour scale across every panel. Per-panel autoscaling would make a field that
    # barely moves look richly varied, which is the exact question being asked.
    sel = maps[picks]
    vmin, vmax = float(np.percentile(sel, 1)), float(np.percentile(sel, 99))
    for ax, p in zip(axes.ravel(), picks):
        g = maps[p].reshape(GRID, GRID)
        im = ax.imshow(g, cmap="YlGnBu", vmin=vmin, vmax=vmax, interpolation="nearest")
        mark_tok(ax, st)
        ax.set_title(f"{date[p]:%Y-%m-%d}\nrange {g.min():.3f}–{g.max():.3f}  "
                     f"(spread {g.max()-g.min():.3f})", loc="left")
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=.6, pad=.02, label=f"predicted SM {args.depth} cm")
    fig.suptitle(f"{args.tile} — predicted soil moisture at 160 m, {args.per_year} dates "
                 f"per year, {years[0]}–{years[-1]}.  All 196 patches from one forward "
                 f"pass per date; shared colour scale.", fontsize=9)
    for ext in ("png", "pdf"):
        p = out_dir / f"{args.tile}_sm_maps.{ext}"; fig.savefig(p); print(f"wrote {p}")
    plt.close(fig)

    # ── FIGURE 3: six separate station series ────────────────────────────
    obs_df = pd.read_parquet(args.preds)
    obs_df = obs_df[obs_df["depth"] == args.depth]

    fig, axes = plt.subplots(len(st), 1, figsize=(11.0, 2.05 * len(st)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    rows = []
    for i, ((_, r), ax) in enumerate(zip(st.iterrows(), axes)):
        name = r["station"]
        p = pd.DataFrame({"date": date, "pred": maps[:, int(r["token"])]})
        ob = (obs_df[obs_df["station_key"] == name][["date", "obs"]]
              .assign(date=lambda d: pd.to_datetime(d["date"])))
        m = p.merge(ob, on="date", how="inner").dropna()

        ax.plot(p["date"], p["pred"], "-", lw=1.0, color=STATION_COLOURS[i % 6],
                label=f"predicted · token {int(r['token'])}", zorder=3)
        if not m.empty:
            ax.plot(m["date"], m["obs"], ".", ms=2.0, color="black",
                    label="observed", zorder=4)
            e  = m["pred"] - m["obs"]
            ub = float(np.sqrt(np.mean(((m["pred"] - m["pred"].mean())
                                        - (m["obs"] - m["obs"].mean())) ** 2)))
            rmse = float(np.sqrt((e ** 2).mean()))
            rr = float(np.corrcoef(m["pred"], m["obs"])[0, 1]) if len(m) > 2 else np.nan
            txt = (f"ubRMSE {ub:.4f}   RMSE {rmse:.4f}   r {rr:+.3f}   "
                   f"bias {float(e.mean()):+.4f}   n {len(m)}")
            rows.append(dict(station=name, token=int(r["token"]),
                             dist_m=float(r["dist_m"]), ubRMSE=ub, RMSE=rmse, r=rr,
                             bias=float(e.mean()), n=len(m),
                             obs_mean=float(m["obs"].mean()),
                             pred_mean=float(m["pred"].mean())))
        else:
            txt = "no overlapping observations"
        ax.text(0.005, 0.95, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=7, bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.4))
        ax.set_ylabel(f"{name.replace('ISMN_TxSON_','')}\nSM (m³/m³)", fontsize=7)
        ax.set_ylim(0, 0.55)
        ax.grid(True, color=GRIDC, lw=.5); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=6, frameon=False, loc="upper right", ncol=2)

    axes[-1].set_xlabel("date")
    fig.suptitle(f"{args.tile} — {len(st)} stations, {args.depth} cm.  Every series is read "
                 f"from the SAME forward pass on the SAME tile, at its own 160 m token.",
                 fontsize=9)
    for ext in ("png", "pdf"):
        p = out_dir / f"{args.tile}_series.{ext}"; fig.savefig(p); print(f"wrote {p}")
    plt.close(fig)

    if rows:
        mt = pd.DataFrame(rows)
        csv = out_dir / f"{args.tile}_station_metrics.csv"
        mt.to_csv(csv, index=False)
        print(f"wrote {csv}\n")
        print(mt.to_string(index=False))
        print(f"\nobserved  mean-level spread "
              f"{mt['obs_mean'].max() - mt['obs_mean'].min():.4f}")
        print(f"predicted mean-level spread "
              f"{mt['pred_mean'].max() - mt['pred_mean'].min():.4f}")
        if len(mt) > 2:
            print(f"r(obs_mean, pred_mean) over {len(mt)} stations "
                  f"{np.corrcoef(mt['obs_mean'], mt['pred_mean'])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
