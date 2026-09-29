"""One tile at 20 m: static inputs, the predicted SM field through the year, every station.

§35.33.6 found the model predicts nearly the same soil moisture at every station in a tile.
This shows the field it actually paints, rather than inferring it from point readouts.

§48: the model emits the whole map itself -- model(batch)["sm"] is (B, 3, 112, 112) at 20 m
over the 2.24 km tile, supervised only at the station pixel (56, 56). One forward pass per
date is the full field; no token gather is needed. The readout table (csvs/txson_readouts.csv)
is on the 224 x 10 m grid, so a station at (row, col) sits at map pixel (row // 2, col // 2).
The six TxSON stations inside ISMN_TxSON_CR200-18 all come from ONE forward pass on ONE tile.

Four figures:
    {tile}_inputs.{png,pdf}   DEM / LULC / soil / S1 VV at 10 m, with the stations
    {tile}_sm_maps.{png,pdf}  predicted SM at 20 m, --per-year dates per year, ONE shared
                              colour scale across every panel
    {tile}_series.{png,pdf}   one panel per station, predicted (its own map pixel) vs observed
    {tile}_field.{png,pdf}    time-mean map with stations coloured by OBSERVED mean on the same
                              scale, and the within-tile spread through time (is the map flat?)

Observations are QC=0 days from the token zarr (combine_network.load_observations), so
train stations -- absent from every eval split parquet -- are shown too, and flagged.

Usage
-----
    python plot_tile_sm_map.py --run-name s48_full_20260929 --tile ISMN_TxSON_CR200-18 \
        --years 2019 2020 --out-dir figures/eval/s48_full_20260929/tile
    python plot_tile_sm_map.py ... --cache-only          # re-plot, no GPU
    python plot_tile_sm_map.py ... --check-preds eval_output/<run>/predictions_oos.parquet
        # asserts the map's station pixel equals eval_predict's value on shared dates
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
ERA5_STATS = REPO / "csvs" / "era5_stats18.json"      # §47: 18 columns, as in training

PATCH_PX, RES_M = 224, 10          # raw inputs and the readout table: 224 x 10 m
MAP_PX, STRIDE  = 112, 2           # model map: 112 x 20 m; map pixel = readout pixel // 2
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


def _key(r) -> str:
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def _category(r) -> str:
    sm, fl = bool(r["has_soil_moisture"]), bool(r["has_flux"])
    return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")


# ---------------------------------------------------------------------------
# Stage 1 -- one forward pass per date; the model returns the full 20 m map
# ---------------------------------------------------------------------------
def predict_tile(tile: str, run_name: str, ckpt: str, years: list[int],
                 cache: Path, batch_size: int = 32) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from dataset import SoilMoistureDataset
    from ckpt_utils import load_checkpoint
    from splits_config import SM_CATEGORIES

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU (bf16 autocast, as in eval_predict.py); "
                         "use --cache-only to re-plot")
    device = torch.device("cuda")
    model, cfg, epoch = load_checkpoint(CKPT_ROOT / run_name / ckpt, device)
    model.eval()
    print(f"  checkpoint {run_name}/{ckpt}: epoch {epoch}")

    splits = pd.read_csv(SPLITS)
    sub = splits[splits.apply(_key, axis=1) == tile]
    if sub.empty:
        raise SystemExit(f"{tile} not in {SPLITS}")
    split_name = str(sub.iloc[0]["split"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = cache.parent / f"_tile_{tile}.csv"
    sub.to_csv(tmp_csv, index=False)
    print(f"  {tile} is split={split_name}")

    # Exactly eval_predict.py's construction, restricted to one station.
    ds = SoilMoistureDataset(
        splits_csv      = str(tmp_csv),
        era5_stats_path = str(ERA5_STATS),
        years           = years,
        category_filter = cfg.get("category_filter", list(SM_CATEGORIES)),
        split_filter    = [split_name],
        training        = False,
    )
    tmp_csv.unlink(missing_ok=True)
    if len(ds) == 0:
        raise SystemExit(f"no samples for {tile} in {years}")
    print(f"  {len(ds)} dates")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    maps, lsts, ys, dys = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
            sm = out["sm"]
            if sm.ndim != 4 or tuple(sm.shape[-2:]) != (MAP_PX, MAP_PX):
                raise SystemExit(f"expected (B,3,{MAP_PX},{MAP_PX}), got {tuple(sm.shape)}")
            maps.append(sm.float().cpu().numpy().astype(np.float16))
            lsts.append(out["lst"][:, 0].float().cpu().numpy().astype(np.float16))
            ys.append(np.asarray(batch["year"]))
            dys.append(np.asarray(batch["doy"]))

    res = dict(maps=np.concatenate(maps), lst=np.concatenate(lsts),
               year=np.concatenate(ys), doy=np.concatenate(dys),
               epoch=np.asarray(epoch), run=np.asarray(run_name))
    np.savez_compressed(cache, **res)
    print(f"  cached maps {res['maps'].shape} lst {res['lst'].shape} -> {cache}")
    return res


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def mark(ax, st: pd.DataFrame, scale: float = 1.0, labels: bool = True):
    """Station markers. scale=1 on a 224 x 10 m panel, 1/STRIDE on the 112 x 20 m map.
    Train stations are squares: their own label was seen through their own tile."""
    for i, (_, r) in enumerate(st.iterrows()):
        x, y = r["col"] * scale, r["row"] * scale
        ax.plot(x, y, "s" if r["cur_split"] == "train" else "o", ms=6,
                mfc=STATION_COLOURS[i % 6], mec="white", mew=1.0, zorder=5)
        if labels:
            ax.annotate(f"{r['station_name']} ({r['cur_split']})", (x, y),
                        textcoords="offset points", xytext=(6, 4), fontsize=6,
                        color="white", zorder=6)
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


def _ub(p, o) -> float:
    return float(np.sqrt(np.mean(((p - p.mean()) - (o - o.mean())) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile",     default="ISMN_TxSON_CR200-18")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt",     default="best.pt")
    ap.add_argument("--years",    type=int, nargs=2, default=[2019, 2020])
    ap.add_argument("--per-year", type=int, default=4,
                    help="SM map panels per year (>=4 samples the seasonal cycle)")
    ap.add_argument("--readouts", default="csvs/txson_readouts.csv")
    ap.add_argument("--check-preds", default=None,
                    help="eval_predict parquet holding the tile station; asserts the map's "
                         "station pixel reproduces it")
    ap.add_argument("--soil-channel", type=int, default=3)
    ap.add_argument("--depth",    default="0-10", choices=SM_DEPTHS)
    ap.add_argument("--out-dir",  required=True)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--dpi",      type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"{args.tile}_map20m.npz"
    years = list(range(args.years[0], args.years[1] + 1))
    di = SM_DEPTHS.index(args.depth)

    splits = pd.read_csv(SPLITS)
    splits["key"] = splits.apply(_key, axis=1)
    split_map = dict(zip(splits.key, splits.split))
    cat_map = {r["key"]: _category(r) for _, r in splits.iterrows()}

    ro = pd.read_csv(args.readouts)
    st = ro[ro["tile"] == args.tile].copy().sort_values("dist_m").reset_index(drop=True)
    if st.empty:
        raise SystemExit(f"no readout rows for {args.tile} in {args.readouts}")
    # The readout table predates §47; the split that matters is today's.
    st["cur_split"] = st["station"].map(split_map).fillna("n/a")
    st["mrow"], st["mcol"] = st["row"] // STRIDE, st["col"] // STRIDE
    if not ((st["mrow"] == 56) & (st["mcol"] == 56))[st["is_centre"]].all():
        raise SystemExit("centre station does not map to pixel (56, 56) -- readout table wrong")
    print(f"{args.tile}: {len(st)} stations at map pixels "
          f"{list(zip(st['mrow'], st['mcol']))}  splits {list(st['cur_split'])}")

    if args.cache_only or cache.exists():
        z = np.load(cache); pred = {k: z[k] for k in z.files}
        print(f"  loaded cache {pred['maps'].shape}")
    else:
        pred = predict_tile(args.tile, args.run_name, args.ckpt, years, cache)

    maps = pred["maps"][:, di].astype(np.float32)                 # (N, 112, 112)
    date = (pd.to_datetime(pd.Series(pred["year"]).astype(str) + "-01-01")
            + pd.to_timedelta(pred["doy"] - 1, unit="D"))
    o = np.argsort(date.values)
    maps, date = maps[o], date.iloc[o].reset_index(drop=True)

    # ── map/eval agreement: the map's station pixel must BE eval_predict's prediction ──
    if args.check_preds:
        ev = pd.read_parquet(args.check_preds)
        if "station_key" in ev.columns:          # a per-split parquet
            ev = ev[(ev["station_key"] == args.tile) & (ev["depth"] == args.depth)]
        else:                                    # the §26 network parquet: own-tile centre
            ev = ev[(ev["tile"].astype(str) == args.tile) & ev["is_centre"].astype(bool)
                    & (ev["depth"] == args.depth)]
        ev = ev.assign(date=pd.to_datetime(ev["date"]))[["date", "pred"]]
        m = pd.DataFrame({"date": date, "map": maps[:, 56, 56]}).merge(ev, on="date")
        if m.empty:
            print(f"  CHECK  no shared dates with {args.check_preds}")
        else:
            d = float(np.abs(m["map"] - m["pred"]).max())
            # fp16 cache + bf16 autocast: agreement to ~1e-3 m3/m3, not bit-identical
            print(f"  CHECK  map pixel (56,56) vs eval_predict on {len(m)} dates: "
                  f"max |diff| {d:.2e}  {'PASS' if d < 2e-3 else 'FAIL'}")

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
    mark(axes[0], st)

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
    mark(axes[1], st)

    try:
        s = zarr.open_consolidated(str(TOK_ZARR / cat_map.get(args.tile, "sm_only")
                                       / args.tile))["soil"][args.soil_channel]
        s = s.astype(np.float32)
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
    mark(axes[2], st)

    s1d = _dstr(raw["s1_asc/dates"][:])
    j = int(np.argmin([abs(int(d) - int(f"{years[0]}0101")) for d in s1d]))
    vv = np.clip(raw["s1_asc/data"][j, 0].astype(np.float32), -20, 0)
    im = axes[3].imshow(vv, cmap="gray")
    fig.colorbar(im, ax=axes[3], shrink=.8, pad=.02)
    axes[3].set_title(f"S1 VV (dB)  {s1d[j]}", loc="left")
    mark(axes[3], st)

    fig.suptitle(f"{args.tile} — static inputs at 10 m over the 2.24 km tile.  "
                 f"Squares = train stations, circles = held out (current §47 split).",
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
        g = maps[p]
        im = ax.imshow(g, cmap="YlGnBu", vmin=vmin, vmax=vmax, interpolation="nearest")
        mark(ax, st, scale=1 / STRIDE, labels=False)
        ax.set_title(f"{date[p]:%Y-%m-%d}\np5–p95 {np.percentile(g, 5):.3f}–"
                     f"{np.percentile(g, 95):.3f}  (sd {g.std():.4f})", loc="left")
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=.6, pad=.02, label=f"predicted SM {args.depth} cm")
    fig.suptitle(f"{args.tile} — predicted soil moisture at 20 m, {args.per_year} dates "
                 f"per year, {years[0]}–{years[-1]}.  One forward pass per date; "
                 f"shared colour scale.", fontsize=9)
    for ext in ("png", "pdf"):
        p = out_dir / f"{args.tile}_sm_maps.{ext}"; fig.savefig(p); print(f"wrote {p}")
    plt.close(fig)

    # ── FIGURE 3: one series per station ─────────────────────────────────
    from combine_network import load_observations
    obs_all = load_observations(sorted(st["station"].unique()), cat_map)
    obs_all = obs_all[obs_all["depth"] == args.depth].assign(
        date=lambda d: pd.to_datetime(d["date"]))

    fig, axes = plt.subplots(len(st), 1, figsize=(11.0, 2.05 * len(st)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    rows = []
    for i, ((_, r), ax) in enumerate(zip(st.iterrows(), axes)):
        name = r["station"]
        p = pd.DataFrame({"date": date, "pred": maps[:, int(r["mrow"]), int(r["mcol"])]})
        ob = obs_all[obs_all["station"] == name][["date", "obs"]]
        m = p.merge(ob, on="date", how="inner").dropna()

        ax.plot(p["date"], p["pred"], "-", lw=1.0, color=STATION_COLOURS[i % 6],
                label=f"predicted · pixel ({int(r['mrow'])},{int(r['mcol'])})", zorder=3)
        if not m.empty:
            ax.plot(m["date"], m["obs"], ".", ms=2.0, color="black",
                    label="observed", zorder=4)
            e    = m["pred"] - m["obs"]
            ub   = _ub(m["pred"], m["obs"])
            rmse = float(np.sqrt((e ** 2).mean()))
            rr   = float(np.corrcoef(m["pred"], m["obs"])[0, 1]) if len(m) > 2 else np.nan
            txt = (f"ubRMSE {ub:.4f}   RMSE {rmse:.4f}   r {rr:+.3f}   "
                   f"bias {float(e.mean()):+.4f}   n {len(m)}")
            rows.append(dict(station=name, split=r["cur_split"],
                             is_centre=bool(r["is_centre"]),
                             mrow=int(r["mrow"]), mcol=int(r["mcol"]),
                             dist_m=float(r["dist_m"]), ubRMSE=ub, RMSE=rmse, r=rr,
                             bias=float(e.mean()), n=len(m),
                             obs_mean=float(m["obs"].mean()),
                             pred_mean=float(m["pred"].mean())))
        else:
            txt = "no overlapping observations"
        ax.text(0.005, 0.95, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=7, bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.4))
        ax.set_ylabel(f"{r['station_name']}\n({r['cur_split']})\nSM (m³/m³)", fontsize=7)
        ax.set_ylim(0, 0.55)
        ax.grid(True, color=GRIDC, lw=.5); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=6, frameon=False, loc="upper right", ncol=2)

    axes[-1].set_xlabel("date")
    fig.suptitle(f"{args.tile} — {len(st)} stations, {args.depth} cm.  Every series is read "
                 f"from the SAME forward pass on the SAME tile, at its own 20 m pixel; only "
                 f"(56,56) is supervised.", fontsize=9)
    for ext in ("png", "pdf"):
        p = out_dir / f"{args.tile}_series.{ext}"; fig.savefig(p); print(f"wrote {p}")
    plt.close(fig)

    mt = pd.DataFrame(rows)

    # ── FIGURE 4: is the field flat? ─────────────────────────────────────
    # (a) time-mean map, stations filled with their OBSERVED mean on the SAME scale: a map
    #     that has learned the field shows dots that blend in; a flat map shows dots that
    #     stand out. Means are over each station's own observed dates, so they compare
    #     observed and predicted over the same days.
    # (b) within-tile spatial SD per date, against the observed spread between stations.
    tmean = maps.mean(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1.5]))
    lo_hi = [tmean.min(), tmean.max()]
    if not mt.empty:
        lo_hi += [mt["obs_mean"].min(), mt["obs_mean"].max()]
    vmin, vmax = float(min(lo_hi)), float(max(lo_hi))
    im = axes[0].imshow(tmean, cmap="YlGnBu", vmin=vmin, vmax=vmax, interpolation="nearest")
    for _, r in mt.iterrows():
        axes[0].scatter(r["mcol"], r["mrow"], s=70, c=[r["obs_mean"]], cmap="YlGnBu",
                        vmin=vmin, vmax=vmax, marker="s" if r["split"] == "train" else "o",
                        edgecolors="black", linewidths=1.0, zorder=5)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], shrink=.8, pad=.02, label=f"SM {args.depth} cm")
    axes[0].set_title(f"(a) predicted time-mean {years[0]}–{years[-1]}; dots = OBSERVED "
                      f"station mean,\nsame scale.  map p5–p95 "
                      f"{np.percentile(tmean, 5):.3f}–{np.percentile(tmean, 95):.3f}",
                      loc="left")

    sd_t = maps.reshape(len(maps), -1).std(1)
    axes[1].plot(date, sd_t, "-", lw=1.0, color="#4363d8", label="predicted: within-tile SD")
    if len(mt) > 1:
        obs_sd = float(mt["obs_mean"].std(ddof=0))
        prd_sd = float(mt["pred_mean"].std(ddof=0))
        axes[1].axhline(obs_sd, color="black", ls="--", lw=1.0,
                        label=f"observed: SD of station means ({obs_sd:.4f})")
        axes[1].axhline(prd_sd, color="#e6194b", ls=":", lw=1.2,
                        label=f"predicted at the stations: SD of means ({prd_sd:.4f})")
    axes[1].set_ylabel("m³/m³"); axes[1].set_ylim(bottom=0)
    axes[1].grid(True, color=GRIDC, lw=.5)
    for side in ("top", "right"):
        axes[1].spines[side].set_visible(False)
    axes[1].legend(fontsize=7, frameon=True, framealpha=.9, loc="center right")
    axes[1].set_title("(b) spatial spread of the predicted map through time", loc="left")
    fig.suptitle(f"{args.tile} — does the 20 m map carry the station-to-station field?",
                 fontsize=9)
    for ext in ("png", "pdf"):
        p = out_dir / f"{args.tile}_field.{ext}"; fig.savefig(p); print(f"wrote {p}")
    plt.close(fig)

    if not mt.empty:
        csv = out_dir / f"{args.tile}_station_metrics.csv"
        mt.to_csv(csv, index=False)
        print(f"wrote {csv}\n")
        print(mt.to_string(index=False))
        print(f"\nobserved  mean-level spread (max-min) "
              f"{mt['obs_mean'].max() - mt['obs_mean'].min():.4f}")
        print(f"predicted mean-level spread (max-min) "
              f"{mt['pred_mean'].max() - mt['pred_mean'].min():.4f}")
        if len(mt) > 2:
            print(f"r(obs_mean, pred_mean) over {len(mt)} stations "
                  f"{np.corrcoef(mt['obs_mean'], mt['pred_mean'])[0, 1]:+.3f}")
        print(f"within-tile map SD, median over dates {np.median(sd_t):.4f}")


if __name__ == "__main__":
    main()
