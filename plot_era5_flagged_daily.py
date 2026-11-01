#!/usr/bin/env python
"""
plot_era5_flagged_daily.py
==========================
Every day on disk, for every station the §43.12 radiation check flagged.  The
companion to `plot_era5_flagged_temperature.py`, which plotted a day-of-year
CLIMATOLOGY -- and a climatology is an average over years, so it smooths away
exactly the artifacts a daily series exposes:

  * a month-boundary step.  `download_era5_radiation.py` builds each year from
    MONTHLY GEE queries and `pd.concat`s them.  A de-accumulation that resets per
    request shows up as a jump at the 1st of the month and nowhere else.
  * a flat run, a zero run, or a duplicated year.
  * whether the sub-floor `strd` days are isolated cold snaps or a whole season.

Outputs
-------
  fig/era5_radiation/era5_flagged_daily.pdf   one page per flagged station (35)
  fig/era5_radiation/era5_ngari_daily.png     the 6 Ngari/Naqu stations, the
                                              unresolved ones, on a single sheet

Each panel: ssrd_sum (top) and strd_sum (bottom), full record, year boundaries
marked, sub-10 MJ strd days highlighted, Brutsaert clear-sky overlaid on strd.

The month-boundary test
-----------------------
For each station compare the mean |day-to-day change| ACROSS a month boundary
with the mean |day-to-day change| WITHIN months.  Clean data: ratio ~1.0, because
nothing physical knows about the calendar.  A per-request de-accumulation: ratio
>> 1.  This is printed as a table and is the real reason to run this script.

Usage
-----
    sbatch slurm/plot_era5_flagged_daily.sh

Env: `soilmoisture` or `terramind`.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from plot_era5_flagged_temperature import (
    COLD, FLAT, FIG, MJ, STRD_FLOOR_MJ, load_station,
)

NGARI = ["ISMN_NGARI_ALI01", "ISMN_NGARI_ALI02", "ISMN_NGARI_ALI03",
         "ISMN_NGARI_SQ19", "ISMN_NGARI_SQ20", "ISMN_NGARI_SQ21"]

C_SSRD = "#D55E00"   # Okabe-Ito vermillion
C_STRD = "#0072B2"   # Okabe-Ito blue
C_CLR  = "#009E73"   # Okabe-Ito bluish-green


def _with_dates(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["date"] = pd.to_datetime(d["date_int"].astype(str), format="%Y%m%d")
    return d.sort_values("date").reset_index(drop=True)


def month_boundary_ratio(d: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """mean |diff| across month boundaries vs within months, and their ratio.

    Gaps longer than one day are excluded so a missing stretch cannot masquerade
    as a boundary jump.
    """
    dd = _with_dates(d)
    step = dd[col].diff().abs().values[1:]
    gap1 = (dd["date"].diff().dt.days.values[1:] == 1)
    is_first = (dd["date"].dt.day.values[1:] == 1)
    across = step[gap1 & is_first]
    within = step[gap1 & ~is_first]
    if across.size == 0 or within.size == 0:
        return np.nan, np.nan, np.nan
    a, w = float(np.nanmean(across)), float(np.nanmean(within))
    return a, w, (a / w if w > 0 else np.nan)


def _panel(axes, d: pd.DataFrame, station: str) -> None:
    dd = _with_dates(d)
    ax_s, ax_l = axes

    ax_s.plot(dd["date"], dd["ssrd_MJ"], color=C_SSRD, lw=0.5)
    ax_s.set_ylabel("ssrd_sum\n(MJ m-2 d-1)")
    ax_s.set_title(f"{station}   n={len(dd)} days   "
                   f"{dd['date'].min():%Y-%m-%d} to {dd['date'].max():%Y-%m-%d}",
                   loc="left", fontsize=9)

    ax_l.plot(dd["date"], dd["strd_MJ"], color=C_STRD, lw=0.5, label="strd_sum")
    ax_l.plot(dd["date"], dd["clearsky_MJ"], color=C_CLR, lw=0.5, alpha=0.8,
              label="Brutsaert clear sky")
    below = dd["strd_MJ"] < STRD_FLOOR_MJ
    if below.any():
        ax_l.scatter(dd.loc[below, "date"], dd.loc[below, "strd_MJ"],
                     s=9, color="#CC79A7", zorder=5, lw=0,
                     label=f"< {STRD_FLOOR_MJ:.0f} MJ (n={int(below.sum())})")
    ax_l.axhline(STRD_FLOOR_MJ, color="0.3", ls=":", lw=0.8)
    ax_l.set_ylabel("strd_sum\n(MJ m-2 d-1)")
    ax_l.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")

    a, w, r = month_boundary_ratio(d, "strd_MJ")
    if np.isfinite(r):
        ax_l.text(0.99, 0.05,
                  f"month-boundary |step| {a:.3f} vs within {w:.3f}  ->  ratio {r:.2f}",
                  transform=ax_l.transAxes, fontsize=7, ha="right", color="0.3")

    for ax in (ax_s, ax_l):
        for yr in range(dd["date"].dt.year.min(), dd["date"].dt.year.max() + 2):
            ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="0.85", lw=0.6, zorder=0)
        ax.margins(x=0.01)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    stations = COLD + FLAT

    with Pool(16) as pool:
        frames = pool.map(load_station, stations)
    data = {s: d for s, d in zip(stations, frames) if d is not None}
    print(f"loaded {len(data)}/{len(stations)}")
    if not data:
        return 1

    # ── the Ngari sheet ──────────────────────────────────────────────────────
    ngari = [s for s in NGARI if s in data]
    if ngari:
        fig, axes = plt.subplots(len(ngari) * 2, 1,
                                 figsize=(13, 2.6 * len(ngari)), sharex=False)
        for i, s in enumerate(ngari):
            _panel((axes[2 * i], axes[2 * i + 1]), data[s], s)
        fig.suptitle("§43.12 -- daily ssrd and strd, the Ngari / Naqu stations",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        out_png = FIG / "era5_ngari_daily.png"
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"wrote {out_png}")

    # ── one page per flagged station ─────────────────────────────────────────
    out_pdf = FIG / "era5_flagged_daily.pdf"
    with PdfPages(out_pdf) as pdf:
        for s in stations:
            if s not in data:
                continue
            fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
            _panel(axes, data[s], s)
            axes[1].set_xlabel("date")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {out_pdf}  ({len(data)} pages)")

    # ── the month-boundary table ─────────────────────────────────────────────
    rows = []
    for s, d in data.items():
        a_l, w_l, r_l = month_boundary_ratio(d, "strd_MJ")
        a_s, w_s, r_s = month_boundary_ratio(d, "ssrd_MJ")
        rows.append({"station": s, "days": len(d),
                     "strd_across": a_l, "strd_within": w_l, "strd_ratio": r_l,
                     "ssrd_across": a_s, "ssrd_within": w_s, "ssrd_ratio": r_s})
    t = pd.DataFrame(rows).sort_values("strd_ratio", ascending=False)
    pd.set_option("display.width", 220)
    print("\n--- month-boundary step vs within-month step ---")
    print("clean data -> ratio ~1.00 (nothing physical knows the calendar)")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nstrd ratio: median {t['strd_ratio'].median():.3f}  "
          f"max {t['strd_ratio'].max():.3f}  ({t.iloc[0]['station']})")
    print(f"ssrd ratio: median {t['ssrd_ratio'].median():.3f}  "
          f"max {t['ssrd_ratio'].max():.3f}")

    # ── flat / zero runs ─────────────────────────────────────────────────────
    print("\n--- longest constant run (a de-accumulation or a fill would show here) ---")
    for s, d in data.items():
        dd = _with_dates(d)
        for col in ("ssrd_MJ", "strd_MJ"):
            v = dd[col].values
            same = np.concatenate(([False], np.isclose(np.diff(v), 0.0)))
            best = cur = 0
            for f in same:
                cur = cur + 1 if f else 0
                best = max(best, cur)
            if best >= 3:
                print(f"  {s:32s} {col}: {best + 1} identical consecutive days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
