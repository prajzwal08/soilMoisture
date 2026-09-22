#!/usr/bin/env python
"""
plot_era5_radiation.py
======================
Look at the `rad_{year}.nc` files rather than only trusting the range checks in
`check_era5_radiation.py`.  §43.12 run-order step 2.

Four panels, all in MJ m-2 day-1 so nothing needs a second y-axis:

  (a) ssrd_sum annual cycle   -- day-of-year mean across years, +/-1 SD band
  (b) strd_sum annual cycle   -- same
  (c) ssrd_sum daily series   -- every day on disk, to show interannual repeat
  (d) ssrd vs strd for the highest-latitude station -- the winter crossover

What the panels are meant to prove, beyond "the numbers are in range":

  * (a) must be a clean sinusoid peaking near DOY 172 in the north.  A flat or
    noisy curve is the ACCUMULATED-band mistake, which no range check catches if
    the magnitude happens to land in bounds.
  * amplitude must ORDER BY LATITUDE -- PortGraham (59N) swings hardest, Combate
    (18N) barely at all.  Nothing in the download knows latitude, so this is an
    independent check.
  * (b) strd is much flatter than ssrd and orders by temperature, not insolation.
  * (d) at 59N, downward longwave EXCEEDS shortwave for most of the winter.  That
    crossover is real physics and would be destroyed by a broken de-accumulation.

Colours are Okabe-Ito (the standard colourblind-safe scientific triple), assigned
in fixed order by latitude so the identity never moves between figures.

Usage
-----
    python plot_era5_radiation.py                 # -> fig/era5_radiation/
    sbatch slurm/plot_era5_radiation.sh

Env: `soilmoisture` or `terramind`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

DATA_ROOT = Path("/gpfs/work3/0/prjs1968/data")
FIG       = Path("/gpfs/work3/0/prjs1968/soilMoisture/fig/era5_radiation")

# Fixed order = increasing latitude.  Okabe-Ito blue / vermillion / bluish-green.
STATIONS = [
    ("ISMN_SCAN_Combate",             "Combate, PR  18°N",        "#0072B2"),
    ("ISMN_USCRN_Cape-Charles-5-ENE", "Cape Charles, VA  37°N",   "#D55E00"),
    ("ISMN_SNOTEL_PortGraham",        "Port Graham, AK  59°N",    "#009E73"),
]
MJ = 1e6


def load(folder: str) -> pd.DataFrame | None:
    files = sorted(DATA_ROOT.glob(f"*/{folder}/ERA5Land/rad_????.nc"))
    if not files:
        return None
    frames = []
    for f in files:
        with xr.open_dataset(f) as ds:
            frames.append(ds[["ssrd_sum", "strd_sum"]].to_dataframe())
    d = pd.concat(frames).sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d.index = pd.DatetimeIndex(d.index)
    d["doy"] = d.index.dayofyear
    return d[["ssrd_sum", "strd_sum", "doy"]] / [MJ, MJ, 1]


def _style(ax) -> None:
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors="#444444", labelsize=9)


def _cycle(ax, data, col, title, ylab):
    for folder, label, colour in STATIONS:
        d = data.get(folder)
        if d is None:
            continue
        g = d.groupby("doy")[col]
        m, sd = g.mean(), g.std()
        ax.fill_between(m.index, m - sd, m + sd, color=colour, alpha=0.15, lw=0)
        ax.plot(m.index, m.values, color=colour, lw=2, label=label)
    ax.set_xlim(1, 366)
    ax.set_xticks([1, 60, 121, 182, 244, 305, 366])
    ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Jan"])
    ax.set_title(title, fontsize=11, color="#222222", loc="left", pad=8)
    ax.set_ylabel(ylab, fontsize=9, color="#444444")
    _style(ax)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=FIG)
    args = ap.parse_args()

    data = {f: load(f) for f, _, _ in STATIONS}
    have = {k: v for k, v in data.items() if v is not None}
    if not have:
        print("no rad_*.nc found")
        return 1
    for k, v in have.items():
        print(f"  {k}: {len(v)} days  {v.index.min().date()} .. {v.index.max().date()}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.patch.set_facecolor("white")

    # (a) + (b) annual cycles, shared y-limits so the two are comparable by eye
    _cycle(axes[0, 0], data, "ssrd_sum",
           "a.  Incoming shortwave · annual cycle", "ssrd_sum  (MJ m$^{-2}$ d$^{-1}$)")
    _cycle(axes[0, 1], data, "strd_sum",
           "b.  Incoming longwave · annual cycle", "strd_sum  (MJ m$^{-2}$ d$^{-1}$)")
    top = max(ax.get_ylim()[1] for ax in (axes[0, 0], axes[0, 1]))
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_ylim(0, top)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor="#444444")

    # (c) every day on disk
    ax = axes[1, 0]
    for folder, label, colour in STATIONS:
        d = data.get(folder)
        if d is None:
            continue
        ax.plot(d.index, d["ssrd_sum"].values, color=colour, lw=0.5, alpha=0.85)
    ax.set_title("c.  Incoming shortwave · every day on disk",
                 fontsize=11, color="#222222", loc="left", pad=8)
    ax.set_ylabel("ssrd_sum  (MJ m$^{-2}$ d$^{-1}$)", fontsize=9, color="#444444")
    _style(ax)

    # (d) the high-latitude crossover
    ax = axes[1, 1]
    folder, label, _ = STATIONS[-1]
    d = data.get(folder)
    if d is not None:
        for col, colour, nm in (("ssrd_sum", "#E69F00", "shortwave (ssrd)"),
                                ("strd_sum", "#56B4E9", "longwave  (strd)")):
            g = d.groupby("doy")[col].mean()
            ax.plot(g.index, g.values, color=colour, lw=2, label=nm)
        s = d.groupby("doy")["ssrd_sum"].mean()
        t = d.groupby("doy")["strd_sum"].mean()
        ax.fill_between(s.index, s, t, where=(t > s), color="#56B4E9", alpha=0.12, lw=0)
        ax.set_xlim(1, 366)
        ax.set_xticks([1, 60, 121, 182, 244, 305, 366])
        ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Jan"])
        ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor="#444444")
    ax.set_title(f"d.  {label} · longwave exceeds shortwave all winter",
                 fontsize=11, color="#222222", loc="left", pad=8)
    ax.set_ylabel("MJ m$^{-2}$ d$^{-1}$", fontsize=9, color="#444444")
    _style(ax)

    fig.suptitle("ERA5-Land downward radiation · §43.12 smoke  "
                 "(de-accumulated *_hourly bands, daily sums)",
                 fontsize=12.5, color="#222222", x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = args.out / "era5_radiation_smoke.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\n-> {out}")

    # A compact table beside the figure, so the numbers travel with it.
    rows = []
    for folder, label, _ in STATIONS:
        d = data.get(folder)
        if d is None:
            continue
        cyc = d.groupby("doy")["ssrd_sum"].mean()
        rows.append({
            "station": label, "n_days": len(d),
            "ssrd_mean": d.ssrd_sum.mean(), "ssrd_max": d.ssrd_sum.max(),
            "ssrd_peak_doy": int(cyc.idxmax()),
            "ssrd_amp": cyc.max() - cyc.min(),
            "strd_mean": d.strd_sum.mean(),
            "strd_amp": (lambda g: g.max() - g.min())(d.groupby("doy")["strd_sum"].mean()),
        })
    t = pd.DataFrame(rows)
    print()
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n(MJ m-2 d-1.  Peak DOY near 172 = June solstice in the north.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
