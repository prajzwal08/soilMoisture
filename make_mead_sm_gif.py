# ============================================================
# Mead: animated GIF of the daily predicted SM maps, one frame per day (runbook §69)
# ============================================================
# Input: maps_<tile>_2017-2024.npz from predict_mead_pixels.py --save-maps (frozen §66 model).
# Frame: rows = Ne1 tile, Ne2 tile (2.24 km, offset 546 m east-west); cols = 0-10 / 10-30 /
# 30-100 cm; both towers marked in both tiles; fixed colour scale so frames compare.
# Title carries the date, plus "S2" / "S1" on acquisition days (embed_cos_*_<year>_alldates.csv,
# written by plot_mead_embed.py --all-dates) and "Ne2 irrigation" on logged irrigation days.
#
# Writes figures/mead/mead_sm_maps_<year>.gif
#
# Usage:
#   python make_mead_sm_gif.py [--year 2019] [--fps 5]
# ============================================================

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

import plot_style_bw

PRED    = Path("/gpfs/work3/0/prjs1968/data_mead/pred")
BIF     = Path("/gpfs/work3/0/prjs1968/data_mead/bif")
FIG_DIR = Path("/gpfs/work3/0/prjs1968/soilMoisture/figures/mead")
TILES   = {"Ne1": "AmeriFlux_US-Ne1", "Ne2": "AmeriFlux_US-Ne2"}
DEPTHS  = ["0-10", "10-30", "30-100"]
VMIN, VMAX = 0.10, 0.45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--fps", type=float, default=5)
    a = ap.parse_args()
    y = a.year

    pix = json.loads((PRED / "mead_pixels_2017-2024.json").read_text())["pixels"]
    maps, days = {}, None
    for short, st in TILES.items():
        z = np.load(PRED / f"maps_{st}_2017-2024.npz")
        d = pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
        i = np.where(d.year == y)[0]
        maps[short] = z["sm"][i].astype(np.float32)
        assert days is None or days.equals(d[i]), "Ne1 and Ne2 map dates differ"
        days = d[i]
    acq = {}
    for s in ("S2", "S1"):
        f = PRED / f"embed_cos_{s}_{y}_alldates.csv"
        acq[s] = pd.read_csv(f, index_col=0, parse_dates=True).index if f.exists() else pd.DatetimeIndex([])
    irr = pd.read_csv(BIF / "US-Ne2_irrigation_events.csv", parse_dates=["date_start", "date_end"])
    irr = irr[~irr["cancelled"].fillna(False).astype(bool)]
    irr_days = set()
    for r in irr.itertuples():
        irr_days.update(pd.date_range(r.date_start, r.date_end))
    print(f"{y}: {len(days)} days, S2 {len(acq['S2'])} / S1 {len(acq['S1'])} acquisition days, "
          f"{sum(d in irr_days for d in days)} irrigation days")

    plot_style_bw.apply(globals(), "paper")
    fig, ax = plt.subplots(2, 3, figsize=(9, 6.4))
    ims = {}
    for r, short in enumerate(TILES):
        for c, dep in enumerate(DEPTHS):
            x = ax[r, c]
            ims[r, c] = x.imshow(maps[short][0, c], vmin=VMIN, vmax=VMAX, cmap="YlGnBu",
                                 interpolation="nearest")
            for st, (pr, pc, _) in pix[TILES[short]].items():
                x.plot(pc, pr, "o" if st.endswith("Ne1") else "^", ms=6, mfc="white", mec="black", mew=1.0)
            x.set_xticks([])
            x.set_yticks([])
            x.text(0.02, 0.98, f"{short} tile, {dep} cm", transform=x.transAxes, va="top",
                   fontsize="small", bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
    fig.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.08, wspace=0.04, hspace=0.06)
    fig.colorbar(ims[0, 0], cax=fig.add_axes([0.90, 0.15, 0.02, 0.68]), label="SM (m$^3$ m$^{-3}$)")
    title = fig.text(0.45, 0.95, "", ha="center", va="center", fontsize="large", fontweight="bold")
    h = [plt.Line2D([], [], ls="", marker="o", mfc="white", mec="black", label="Ne1 tower"),
         plt.Line2D([], [], ls="", marker="^", mfc="white", mec="black", label="Ne2 tower")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False)

    def frame(t):
        dd = days[t]
        for r, short in enumerate(TILES):
            for c in range(3):
                ims[r, c].set_data(maps[short][t, c])
        tags = [s for s in ("S2", "S1") if dd in acq[s]] + (["Ne2 irrigation"] if dd in irr_days else [])
        title.set_text(f"{dd:%Y-%m-%d}" + (f"   [{', '.join(tags)}]" if tags else ""))
        return list(ims.values()) + [title]

    out = FIG_DIR / f"mead_sm_maps_{y}.gif"
    FuncAnimation(fig, frame, frames=len(days), blit=False).save(
        out, writer=PillowWriter(fps=a.fps), dpi=90)
    print(f"figure: {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(days)} frames at {a.fps} fps)")


if __name__ == "__main__":
    main()
