#!/usr/bin/env python
"""
plot_era5_flagged_temperature.py
================================
The §43.12 radiation check (`check_era5_radiation.py`) raised two FAIL flags:

  * `strd_sum` below the 10 MJ floor at 32 stations (62 station-years)
  * near-flat `ssrd_sum` (std/mean < 0.15) at 3 stations (11 station-years)

Both are suspected to be TOO-TIGHT THRESHOLDS rather than a broken download --
the same shape as the Landsat 340 K ceiling that flagged Death Valley as broken.
This figure is the falsification.  It plots the annual temperature cycle of the
flagged stations and puts the radiation next to it.

Panels
------
  (a) t2m_mean day-of-year climatology, all 32 cold-flagged stations
  (b) skt_mean day-of-year climatology, same stations -- the SURFACE temperature,
      which swings wider than the air because it decouples under snow
  (c) THE TEST: observed strd_sum vs Brutsaert clear-sky downward longwave,
      computed from each day's OWN t2m and d2m.  Nothing in the download knows
      the Brutsaert formula, so agreement is independent evidence.  Points must
      sit ON or ABOVE the 1:1 line -- cloud only ever ADDS longwave.
  (d) the 3 near-flat stations: ssrd_sum and t2m annual cycle at ~13 N, where a
      weak solar cycle is correct rather than a de-accumulation failure.

Reads t2m/skt/d2m from the pre-splice `era5/values` (19 cols, skt still present)
and ssrd/strd from `rad_{year}.nc`, joining on YYYYMMDD.  That join is the same
one `splice_era5_radiation.py` will perform, so a clean join here is also a
dry-run of the splice alignment.

Usage
-----
    sbatch slurm/plot_era5_flagged_temperature.sh

Env: `soilmoisture` or `terramind` -- xarray + pandas + matplotlib + zarr 2.x.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr

DATA_ROOT = Path("/gpfs/work3/0/prjs1968/data")
ZARR_ROOT = Path("/gpfs/work3/0/prjs1968/zarr_tokens")
FIG       = Path("/gpfs/work3/0/prjs1968/soilMoisture/fig/era5_radiation")

# era5/values is the PRE-splice 19-column array; skt is still in it.
ERA5_VARS19 = [
    "t2m_mean", "t2m_min", "t2m_max",
    "d2m_mean", "d2m_min", "d2m_max",
    "skt_mean", "skt_min", "skt_max",
    "u10_mean", "u10_min", "u10_max",
    "v10_mean", "v10_min", "v10_max",
    "sp_mean",  "sp_min",  "sp_max",
    "tp_sum",
]
I_T2M = ERA5_VARS19.index("t2m_mean")
I_D2M = ERA5_VARS19.index("d2m_mean")
I_SKT = ERA5_VARS19.index("skt_mean")

# strd_min < 1.0e7 J m-2 day-1 in csvs/era5_radiation_check.csv
COLD = [
    "AmeriFlux_US-ICs", "AmeriFlux_US-ICt", "AmeriFlux_US-MN2",
    "AmeriFlux_US-xDJ", "AmeriFlux_US-xWD",
    "ISMN_NAQU_NQKema", "ISMN_NAQU_NQMS", "ISMN_NAQU_NQNorth",
    "ISMN_NGARI_ALI01", "ISMN_NGARI_ALI02", "ISMN_NGARI_ALI03",
    "ISMN_NGARI_SQ19", "ISMN_NGARI_SQ20", "ISMN_NGARI_SQ21",
    "ISMN_ROMPS_GolubinGlacier",
    "ISMN_SCAN_GlacialRidge", "ISMN_SCAN_Jordan", "ISMN_SCAN_Moccasin",
    "ISMN_SNOTEL_AmericanCreek", "ISMN_SNOTEL_Chisana", "ISMN_SNOTEL_Coldfoot",
    "ISMN_SNOTEL_EagleSummit", "ISMN_SNOTEL_FieldingLake", "ISMN_SNOTEL_GalenaAK",
    "ISMN_SNOTEL_GobblersKnob", "ISMN_SNOTEL_ImnaviatCreek",
    "ISMN_SNOTEL_JackWadeJct", "ISMN_SNOTEL_KellyStation", "ISMN_SNOTEL_McGrath",
    "ISMN_SNOTEL_MooreCreekBridge", "ISMN_SNOTEL_TelaquanaLake", "ISMN_SNOTEL_Tok",
]
# ssrd_season < 0.15
FLAT = ["ISMN_AMMA-CATCH_Banizoumbou", "ISMN_AMMA-CATCH_Tondikiboro",
        "ISMN_SD_DEM_Demokeya"]

# Okabe-Ito.  Exemplars: the global strd minimum, the coldest Alaskan, the glacier.
EXEMPLARS = {
    "ISMN_NGARI_ALI02":          ("Ali-02, Tibet 4.5 km", "#0072B2"),
    "ISMN_SNOTEL_Coldfoot":      ("Coldfoot, AK 67N",     "#D55E00"),
    "ISMN_ROMPS_GolubinGlacier": ("Golubin Glacier, KG",  "#009E73"),
}
FLAT_COLOURS = ["#0072B2", "#D55E00", "#009E73"]

MJ    = 1e6
SIGMA = 5.670374419e-8
SECS  = 86400.0
STRD_FLOOR_MJ = 10.0   # the threshold that fired


def _category(station: str) -> str | None:
    for cat in ("sm_only", "sm_and_flux", "flux_only"):
        if (DATA_ROOT / cat / station / "ERA5Land").is_dir():
            return cat
    return None


def brutsaert_clearsky_MJ(t2m_K: np.ndarray, d2m_K: np.ndarray) -> np.ndarray:
    """Clear-sky downward longwave, MJ m-2 day-1, from air temp + dewpoint.

    Magnus for vapour pressure (hPa), Brutsaert (1975) for effective emissivity:
        eps = 1.24 * (e / T) ** (1/7)
    This is a LOWER bound on the real flux: cloud raises emissivity toward 1.
    """
    td_C = d2m_K - 273.15
    e_hPa = 6.112 * np.exp(17.67 * td_C / (td_C + 243.5))
    eps = 1.24 * np.power(np.clip(e_hPa, 1e-6, None) / t2m_K, 1.0 / 7.0)
    return eps * SIGMA * np.power(t2m_K, 4) * SECS / MJ


def load_station(station: str) -> pd.DataFrame | None:
    """Join zarr temperature with rad_*.nc radiation on YYYYMMDD."""
    cat = _category(station)
    if cat is None:
        return None

    zpath = ZARR_ROOT / cat / station
    if not (zpath / ".complete").exists():
        return None
    try:
        zg = zarr.open_consolidated(str(zpath), mode="r")
    except KeyError:
        zg = zarr.open_group(str(zpath), mode="r")
    if "era5/values" not in zg:
        return None

    vals = np.asarray(zg["era5/values"][:], dtype=np.float64)
    if vals.ndim != 2 or vals.shape[1] != len(ERA5_VARS19):
        return None
    temp = pd.DataFrame({
        "date_int": np.asarray(zg["era5/date_ints"][:], dtype=np.int64),
        "doy":      np.asarray(zg["era5/doys"][:],      dtype=np.int64),
        "t2m":      vals[:, I_T2M],
        "d2m":      vals[:, I_D2M],
        "skt":      vals[:, I_SKT],
    })

    rad_files = sorted((DATA_ROOT / cat / station / "ERA5Land").glob("rad_????.nc"))
    if not rad_files:
        return None
    frames = []
    for f in rad_files:
        with xr.open_dataset(f) as ds:
            t = pd.to_datetime(ds["time"].values)
            frames.append(pd.DataFrame({
                "date_int": t.year * 10000 + t.month * 100 + t.day,
                "ssrd_MJ":  ds["ssrd_sum"].values.astype(np.float64) / MJ,
                "strd_MJ":  ds["strd_sum"].values.astype(np.float64) / MJ,
            }))
    rad = pd.concat(frames, ignore_index=True)

    d = temp.merge(rad, on="date_int", how="inner")
    if d.empty:
        return None
    d["station"] = station
    d["clearsky_MJ"] = brutsaert_clearsky_MJ(d["t2m"].values, d["d2m"].values)
    return d


def doy_climatology(d: pd.DataFrame, col: str) -> pd.Series:
    return d.groupby("doy")[col].mean()


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)

    with Pool(16) as pool:
        cold_frames = pool.map(load_station, COLD)
        flat_frames = pool.map(load_station, FLAT)

    cold = {s: d for s, d in zip(COLD, cold_frames) if d is not None}
    flat = {s: d for s, d in zip(FLAT, flat_frames) if d is not None}
    missing = [s for s, d in zip(COLD + FLAT, cold_frames + flat_frames) if d is None]
    print(f"loaded {len(cold)}/{len(COLD)} cold, {len(flat)}/{len(FLAT)} flat")
    if missing:
        print("NO JOIN (zarr or rad missing): " + ", ".join(missing))

    if not cold:
        print("nothing to plot")
        return 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ── (a) and (b): annual temperature cycle ────────────────────────────────
    for ax, col, title in (
        (ax_a, "t2m", "(a) 2 m air temperature, day-of-year mean"),
        (ax_b, "skt", "(b) skin (surface) temperature, day-of-year mean"),
    ):
        for s, d in cold.items():
            if s in EXEMPLARS:
                continue
            clim = doy_climatology(d, col)
            ax.plot(clim.index, clim.values, color="0.75", lw=0.7, zorder=1)
        for s, (label, colour) in EXEMPLARS.items():
            if s not in cold:
                continue
            clim = doy_climatology(cold[s], col)
            ax.plot(clim.index, clim.values, color=colour, lw=2.0,
                    label=label, zorder=3)
        ax.axhline(273.15, color="k", ls=":", lw=0.8, zorder=2)
        ax.set_xlabel("day of year")
        ax.set_ylabel("K")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(1, 366)
        ax.legend(fontsize=8, frameon=False)
    ax_a.text(0.02, 0.04, f"grey = the other {len(cold) - len(EXEMPLARS)} flagged stations",
              transform=ax_a.transAxes, fontsize=8, color="0.45")

    # ── (c) the falsification ────────────────────────────────────────────────
    allcold = pd.concat(cold.values(), ignore_index=True)
    below = allcold["strd_MJ"] < STRD_FLOOR_MJ
    ax_c.scatter(allcold.loc[~below, "clearsky_MJ"], allcold.loc[~below, "strd_MJ"],
                 s=2, alpha=0.10, color="0.55", lw=0, label="above 10 MJ floor")
    ax_c.scatter(allcold.loc[below, "clearsky_MJ"], allcold.loc[below, "strd_MJ"],
                 s=7, alpha=0.75, color="#D55E00", lw=0,
                 label=f"below the 10 MJ floor (n={int(below.sum())})")
    lim = [min(allcold["clearsky_MJ"].min(), allcold["strd_MJ"].min()) - 1,
           max(allcold["clearsky_MJ"].max(), allcold["strd_MJ"].max()) + 1]
    ax_c.plot(lim, lim, color="k", lw=1.0, ls="--", label="1:1 (clear sky)")
    ax_c.axhline(STRD_FLOOR_MJ, color="#0072B2", lw=1.0,
                 label=f"check floor {STRD_FLOOR_MJ:.0f} MJ")
    ax_c.set_xlim(lim); ax_c.set_ylim(lim)
    ax_c.set_xlabel("Brutsaert clear-sky LW$\\downarrow$  (MJ m-2 day-1)")
    ax_c.set_ylabel("observed strd_sum  (MJ m-2 day-1)")
    ax_c.set_title("(c) observed vs clear-sky floor -- points must sit ON or ABOVE 1:1",
                   loc="left", fontsize=10)
    ax_c.legend(fontsize=8, frameon=False, loc="upper left")

    frac_above = float((allcold["strd_MJ"] >= allcold["clearsky_MJ"] - 1.0).mean())
    ax_c.text(0.97, 0.05, f"{frac_above * 100:.1f}% on/above clear sky\n"
                          f"n = {len(allcold):,} station-days",
              transform=ax_c.transAxes, fontsize=8, ha="right", color="0.25")

    # ── (d) the near-flat Sahel trio ─────────────────────────────────────────
    ax_d2 = ax_d.twinx()
    for (s, d), colour in zip(flat.items(), FLAT_COLOURS):
        short = s.split("_", 1)[1]
        ax_d.plot(doy_climatology(d, "ssrd_MJ").index,
                  doy_climatology(d, "ssrd_MJ").values,
                  color=colour, lw=1.8, label=short)
        ax_d2.plot(doy_climatology(d, "t2m").index,
                   doy_climatology(d, "t2m").values,
                   color=colour, lw=1.0, ls=":")
    ax_d.set_xlabel("day of year")
    ax_d.set_ylabel("ssrd_sum  (MJ m-2 day-1)   [solid]")
    ax_d2.set_ylabel("t2m  (K)   [dotted]")
    ax_d.set_xlim(1, 366)
    ax_d.set_title("(d) the 3 near-flat stations, all ~13 N", loc="left", fontsize=10)
    ax_d.legend(fontsize=8, frameon=False, loc="lower center")

    fig.suptitle("§43.12 -- annual temperature cycle of the radiation-check "
                 "flagged stations", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "era5_flagged_temperature.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # ── numbers for the write-up ─────────────────────────────────────────────
    print("\n--- cold group, per station ---")
    rows = []
    for s, d in cold.items():
        rows.append({
            "station": s,
            "days": len(d),
            "t2m_min": d["t2m"].min(),
            "t2m_mean": d["t2m"].mean(),
            "skt_min": d["skt"].min(),
            "strd_min_MJ": d["strd_MJ"].min(),
            "clearsky_at_strd_min": d.loc[d["strd_MJ"].idxmin(), "clearsky_MJ"],
            "frac_below_clearsky": float((d["strd_MJ"] < d["clearsky_MJ"] - 1.0).mean()),
        })
    g = pd.DataFrame(rows).sort_values("strd_min_MJ")
    pd.set_option("display.width", 200)
    print(g.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
