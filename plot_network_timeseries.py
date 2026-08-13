"""Per-tile time series: several stations predicted from ONE 224x224 map (§26).

One forward pass on a tile produces a full (3, 224, 224) soil-moisture map.  This plots,
for a single tile, every station that falls inside that map — the one at the supervised
centre pixel (112, 112) and the neighbours at pixels that received no supervision — each
against its own observations, with metrics per panel.

The question the figure answers: does the map track soil moisture away from the pixel the
model was trained on, or does it only work at the centre?

Usage
-----
    python plot_network_timeseries.py --ts eval_output/txson_timeseries.parquet \
                                      --tile ISMN_TxSON_CR200-18 --year 2019
    python plot_network_timeseries.py --ts ... --top 3          # 3 densest tiles
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from eval_metrics import metrics_from_arrays

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent
PATCH_PX = 224
CENTRE = 112

C_OBS = "#333333"
C_CENTRE = "#1a9850"     # supervised readout
C_OFF = "#d62728"        # never-supervised readout


def panel_layout(ax, station: str, is_centre: bool, off_px: int, row: int, col: int):
    tag = ("centre pixel (112,112) — SUPERVISED" if is_centre
           else f"pixel ({row},{col}) — {off_px} px = {off_px*10} m off centre, NEVER SUPERVISED")
    ax.set_title(f"{station.replace('ISMN_','').replace('TxSON_','')}   ·   {tag}",
                 loc="left", fontsize=9,
                 weight="bold" if is_centre else "normal",
                 color=C_CENTRE if is_centre else C_OFF)


def plot_tile(ts: pd.DataFrame, tile: str, depth: str, year_lo: int, year_hi: int,
              out: Path, dpi: int):
    sub = ts[(ts.tile == tile) & (ts.depth == depth)].copy()
    if sub.empty:
        print(f"  no rows for {tile} / {depth}")
        return
    sub = sub[(sub.date.dt.year >= year_lo) & (sub.date.dt.year <= year_hi)]
    # supervised readout first, then by distance from the centre
    order = (sub.groupby("station")
                .agg(is_centre=("is_centre", "first"), off=("offset_px", "first"))
                .sort_values(["is_centre", "off"], ascending=[False, True]))
    stations = list(order.index)
    n = len(stations)

    # Across-pixel spread: the diagnostic that decides whether the panels below mean
    # what they look like.  ubRMSE removes each series' own mean, and in one small
    # catchment every station shares the same wetting/drying signal -- so a model that
    # paints ONE time series over the whole tile still scores well at every pixel.
    # Only the spatial spread separates "resolves the tile" from "repeats the centre".
    P = sub.pivot_table(index="date", columns="station", values="pred")
    O = sub.pivot_table(index="date", columns="station", values="obs")
    idx = P.dropna().index.intersection(O.dropna().index)
    spread = None
    if len(idx) > 30 and P.shape[1] > 1:
        P, O = P.loc[idx], O.loc[idx]
        cp = P.corr().to_numpy()
        co = O.corr().to_numpy()
        iu = np.triu_indices_from(cp, 1)
        spread = dict(pred=float(P.std(axis=1).mean()), obs=float(O.std(axis=1).mean()),
                      pred_r=float(np.median(cp[iu])), obs_r=float(np.median(co[iu])),
                      pred_range=float(P.mean().max() - P.mean().min()),
                      obs_range=float(O.mean().max() - O.mean().min()),
                      n_dates=len(idx))

    head = 2.3 if spread else 1.5
    H = 1.9 * n + head
    fig = plt.figure(figsize=(15, H))
    gs = GridSpec(n, 2, figure=fig, width_ratios=[3.5, 1], hspace=.42, wspace=.16,
                  left=.055, right=.985, top=1 - (head - .35) / H, bottom=.05)

    rows_summary = []
    for i, st in enumerate(stations):
        g = sub[sub.station == st].sort_values("date")
        obs_g = g[g.obs.notna()]
        is_c = bool(g.is_centre.iloc[0])
        colour = C_CENTRE if is_c else C_OFF

        ax = fig.add_subplot(gs[i, 0])
        ax.plot(g.date, g.pred, lw=1.0, color=colour, label="predicted", zorder=3)
        if len(obs_g):
            ax.plot(obs_g.date, obs_g.obs, lw=1.0, color=C_OBS, alpha=.75,
                    label="observed", zorder=2)
        panel_layout(ax, st, is_c, int(g.offset_px.iloc[0]),
                     int(g.row.iloc[0]), int(g.col.iloc[0]))
        ax.set_ylabel("SM (m³/m³)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, lw=.3, alpha=.35)
        ax.margins(x=.01)
        if i == 0:
            ax.legend(fontsize=7.5, ncol=2, loc="upper right", frameon=False)
        if i < n - 1:
            ax.set_xticklabels([])

        # metrics panel
        axm = fig.add_subplot(gs[i, 1])
        axm.set_axis_off()
        if len(obs_g) >= 30:
            m = metrics_from_arrays(obs_g.pred.to_numpy(np.float64),
                                    obs_g.obs.to_numpy(np.float64))
            txt = (f"n         {m['n']:>6d}\n"
                   f"ubRMSE    {m['ubRMSE']:>6.4f}\n"
                   f"RMSE      {m['RMSE']:>6.4f}\n"
                   f"bias      {m['bias']:>+6.4f}\n"
                   f"R         {m['R']:>6.3f}\n"
                   f"NSE_anom  {m['NSE_anom']:>6.3f}")
            rows_summary.append(dict(station=st, is_centre=is_c,
                                     offset_px=int(g.offset_px.iloc[0]), **m))
        else:
            txt = f"n = {len(obs_g)}\n(too few paired days\n for metrics)"
        axm.text(0, .96, txt, family="monospace", fontsize=8.2, va="top",
                 transform=axm.transAxes,
                 bbox=dict(fc="#f7f7f7", ec=colour, lw=1.1, pad=5))
        # station position inside the tile
        axi = axm.inset_axes([.60, .06, .38, .78])
        axi.add_patch(mpatches.Rectangle((0, 0), PATCH_PX, PATCH_PX, fc="#eef2f6",
                                         ec="#888", lw=.7))
        axi.plot(CENTRE, CENTRE, marker="+", ms=8, mew=1.4, color=C_CENTRE)
        axi.plot(int(g.col.iloc[0]), int(g.row.iloc[0]), marker="o", ms=6,
                 color=colour, mec="white", mew=.8)
        axi.set_xlim(0, PATCH_PX); axi.set_ylim(PATCH_PX, 0)
        axi.set_aspect("equal"); axi.set_xticks([]); axi.set_yticks([])
        axi.set_title("2.24 km tile", fontsize=6.2, pad=2)

    n_off = int((~order.is_centre).sum())
    tsplit = sub.tile_split.iloc[0]
    fig.suptitle(
        f"{tile.replace('ISMN_','')} — one 224×224 map, {n} stations, depth {depth}\n"
        f"1 supervised centre pixel + {n_off} never-supervised off-centre pixels"
        f"   ·   tile split: {tsplit}   ·   {year_lo}–{year_hi}",
        x=.055, ha="left", fontsize=12.5, weight="bold", y=.995)

    if spread:
        ratio = spread["pred"] / spread["obs"] if spread["obs"] else np.nan
        verdict = ("the map resolves the tile" if ratio > .6 else
                   "the map repeats ~one series over the whole tile" if ratio < .35
                   else "the map partly resolves the tile")
        fig.text(
            .055, 1 - .70 / (1.9 * n + head),
            f"ACROSS-PIXEL SPREAD (mean over {spread['n_dates']} dates where all "
            f"{n} stations report)   —   predicted {spread['pred']:.4f}   vs   "
            f"observed {spread['obs']:.4f}   =   {100*ratio:.0f}% of the real spatial "
            f"variability\n"
            f"per-pixel mean level spans {spread['pred_range']:.3f} predicted vs "
            f"{spread['obs_range']:.3f} observed   ·   median r between the predicted "
            f"series {spread['pred_r']:.3f} vs {spread['obs_r']:.3f} between the observed "
            f"series   →   {verdict}",
            ha="left", va="top", fontsize=9.2, family="monospace",
            bbox=dict(fc="#fff4f4" if ratio < .35 else "#f4f8ff",
                      ec="#d62728" if ratio < .35 else "#2166ac", lw=1.1, pad=6))

    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.png")

    if rows_summary:
        s = pd.DataFrame(rows_summary)
        print(s[["station", "is_centre", "offset_px", "n",
                 "ubRMSE", "bias", "R", "NSE_anom"]].round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts", required=True, help="timeseries parquet from combine_network.py")
    ap.add_argument("--tile", nargs="*", default=None, help="tile key(s) to plot")
    ap.add_argument("--top", type=int, default=None,
                    help="instead of --tile, plot the N tiles holding the most stations")
    ap.add_argument("--depth", default="0-10")
    ap.add_argument("--years", type=int, nargs=2, default=[2016, 2022])
    ap.add_argument("--out-dir", default=str(REPO / "figures" / "network_timeseries"))
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    ts = pd.read_parquet(args.ts)
    ts["date"] = pd.to_datetime(ts["date"])

    tiles = args.tile
    if not tiles:
        counts = (ts[ts.depth == args.depth].groupby("tile").station.nunique()
                  .sort_values(ascending=False))
        tiles = list(counts.index[: (args.top or 1)])
        print(f"tiles by station count:\n{counts.head(8).to_string()}\n")

    out_dir = Path(args.out_dir)
    for t in tiles:
        print(f"\n=== {t} ===")
        plot_tile(ts, t, args.depth, args.years[0], args.years[1],
                  out_dir / f"{t}_{args.depth}", args.dpi)


if __name__ == "__main__":
    main()
