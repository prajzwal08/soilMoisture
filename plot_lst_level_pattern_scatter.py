"""
plot_lst_level_pattern_scatter.py — one station: LST level vs pattern, scattered against SM
===========================================================================================

Reads csvs/probe_lst_level_pattern/scenes.csv (probe_lst_level_pattern.py). Rows: LST tile −
T2m (level) and LST(station cell) − LST tile mean (pattern). Columns: the three SM depths.
Points coloured by season, because a shared seasonal cycle alone can make both look related
to SM; each panel gives r raw and r with the station's monthly means removed.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SEASON_COLOR = {"DJF": "#3b6fb6", "MAM": "#4f9a5e", "JJA": "#d0822c", "SON": "#8a5aa8"}
ROWS = [("dT_mean", "LST tile − T2m mean (K)"),
        ("P_stn", "LST pattern at station: cell − tile mean (K)")]
DEPTHS = ["0-10"]          # CR200-18 has no observed 10-30 / 30-100 on scene days


def r_of(x, y):
    m = x.notna() & y.notna()
    return (np.corrcoef(x[m], y[m])[0, 1], int(m.sum())) if m.sum() >= 5 else (np.nan, int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="ISMN_TxSON_CR200-18")
    ap.add_argument("--scenes", default="csvs/probe_lst_level_pattern/scenes.csv")
    ap.add_argument("--out-dir", default="figures/probe_lst_level_pattern")
    a = ap.parse_args()

    s = pd.read_csv(a.scenes)
    s = s[s["station"] == a.station].copy()
    if s.empty:
        raise SystemExit(f"{a.station} not in {a.scenes}")
    s["month"] = (s["date"] // 100) % 100
    s["season"] = s["month"].map({m: k for k, ms in SEASONS.items() for m in ms})
    cols = [c for c, _ in ROWS] + [f"sm_{d}" for d in DEPTHS]
    ds = s.copy()
    for c in cols:
        ds[c] = ds[c] - ds.groupby("month")[c].transform("mean")

    fig, axes = plt.subplots(len(DEPTHS), 2, figsize=(13, 5.2 * len(DEPTHS)), squeeze=False)
    for i, (x, xlab) in enumerate(ROWS):
        for j, d in enumerate(DEPTHS):
            ax, y = axes[j, i], f"sm_{d}"
            for k, col in SEASON_COLOR.items():
                g = s[s["season"] == k]
                ax.scatter(g[y], g[x], s=22, color=col, alpha=0.8, edgecolors="white",
                           linewidths=0.4, label=k)
            r, n = r_of(s[x], s[y])
            rd, _ = r_of(ds[x], ds[y])
            ax.set_title(f"SM {d} cm   r = {r:+.2f}   deseasonalised r = {rd:+.2f}   n = {n}",
                         fontsize=10)
            ax.set_xlabel(f"observed SM {d} cm (m³/m³)")
            ax.set_ylabel(xlab)
            ax.axhline(0, color="#999999", lw=0.8) if x == "P_stn" else None
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axes[0, 0].legend(title="season", fontsize=8, title_fontsize=8, frameon=False)
    fig.suptitle(f"{a.station.replace('ISMN_', '')} — Landsat scene days 2016-2022: "
                 "thermal level vs pattern against soil moisture", fontsize=12)
    fig.tight_layout()
    out = Path(a.out_dir) / f"B3_scatter_{a.station.split('_')[-1]}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
