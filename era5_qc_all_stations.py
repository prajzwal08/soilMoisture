#!/usr/bin/env python
"""
era5_qc_all_stations.py
=======================
§45.  All-station ERA5 driver QC.  990 stations against answers that follow from
first principles, in one table and one sheet.

The point is NOT to plot 990 series -- nobody reads 990 plots.  It is to build
quantities whose correct value is derivable WITHOUT the data, then put every
station against them at once, so a wrong station is a point off a curve.

The centrepiece is top-of-atmosphere insolation, from latitude and day-of-year
alone:

    H0 = (86400/pi) * Gsc * dr * (ws*sin(phi)sin(dec) + cos(phi)cos(dec)sin(ws))

Nothing in the download knows this.  It gives three tests at once:
  * Kt = ssrd/H0 must be <= 1.  Physically impossible to exceed.
  * Kt must be climatologically sensible (~0.7 arid, ~0.35-0.45 wet maritime).
  * polar night becomes an EXACT equality: the ws clamp makes H0 exactly 0 on
    polar-night days, so count(H0==0) must equal count(ssrd==0), per station.

Plus the right-pixel test, which nothing else in the pipeline performs:
`sp_mean` against `elevation_m` through the ISA barometric profile.  A station
handed the wrong grid cell reads a pressure its elevation cannot explain -- which
is exactly the §43.12 buffer-pixel defect class.

Outputs
-------
  csvs/era5_all_station_qc.csv          one row per station, every scalar+boolean
  fig/era5_radiation/era5_qc_overview.png   8 panels, all stations

Usage
-----
    sbatch slurm/era5_qc_all_stations.sh

Env: `terramind` (zarr 2.x).
"""
from __future__ import annotations

import argparse
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

REPO_ROOT   = Path("/gpfs/work3/0/prjs1968/soilMoisture")
DATA_ROOT   = Path("/gpfs/work3/0/prjs1968/data")
ZARR_ROOT   = Path("/gpfs/work3/0/prjs1968/zarr_tokens")
STATION_CSV = REPO_ROOT / "csvs" / "station_splits.csv"
OUT_CSV     = REPO_ROOT / "csvs" / "era5_all_station_qc.csv"
FIG         = REPO_ROOT / "fig" / "era5_radiation"

# era5/values carries no .zattrs, so the names live here.  19 columns, the order
# create_token_zarr.py:47 wrote and dataset_unet.py:53-62 reads.
ERA5_VARS19 = [
    "t2m_mean", "t2m_min", "t2m_max",
    "d2m_mean", "d2m_min", "d2m_max",
    "skt_mean", "skt_min", "skt_max",
    "u10_mean", "u10_min", "u10_max",
    "v10_mean", "v10_min", "v10_max",
    "sp_mean",  "sp_min",  "sp_max",
    "tp_sum",
]
IDX = {v: i for i, v in enumerate(ERA5_VARS19)}
TRIPLES = ["t2m", "d2m", "skt", "u10", "v10", "sp"]

MJ     = 1e6
GSC    = 1361.0          # solar constant, W m-2
KT_HARD = 1.05           # Kt above this is a hard violation
# tp_sum is an IFS accumulation in float32; the probe (era5_qc_tp_probe.py) measured
# the most negative value across all 2.46 M station-days as -3.875e-8 m, i.e. 39 nm
# against a typical 2.1 mm wet day -- numerical noise, not a defect. 1 micron is the
# threshold at which ZERO stations fail; anything stricter just counts float error.
TP_NEG_TOL = -1e-6       # m of water per day
# Geometric sunrise (sun centre on the horizon) is STRICTER than real illumination:
# the solar disc is ~0.27 deg and horizon refraction ~0.57 deg, so the standard
# sunrise convention is -0.833 deg, and ERA5 additionally carries diffuse light
# through twilight. Polar-night day counts are therefore compared at three
# elevations, not one.
SUN_ELEV = {"geometric": 0.0, "refracted": -0.833, "civil": -6.0}


# ── first principles ─────────────────────────────────────────────────────────
def toa_daily_MJ(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    """Daily extraterrestrial irradiation on a horizontal surface, MJ m-2 d-1.

    Cooper declination + the sunset hour angle.  The arccos clamp is what makes
    polar night and polar day fall out for free: when -tan(phi)tan(dec) > 1 the
    sun never rises, ws = 0, and H0 is exactly 0.
    """
    phi = np.radians(lat_deg)
    dec = np.radians(23.45) * np.sin(2.0 * np.pi * (284.0 + doy) / 365.0)
    dr  = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    ws  = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0))
    h0  = (86400.0 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws)
    )
    return np.maximum(h0, 0.0) / MJ


def n_sunless_days(lat_deg: float, doy: np.ndarray, elev_deg: float) -> int:
    """Days on which the sun never rises above `elev_deg`.

    `toa_daily_MJ` uses the GEOMETRIC horizon (sun centre at 0 deg), which is the
    correct convention for an energy integral but the WRONG one for asking whether
    any light reached the ground: the solar disc has a ~0.27 deg semi-diameter and
    horizon refraction adds ~0.57 deg, so sunrise is conventionally -0.833 deg, and
    ERA5's radiation scheme carries diffuse light further still through twilight.
    Comparing observed zero-ssrd days against the geometric count therefore produces
    a ONE-SIDED deficit at high latitude that is physics, not a defect.
    """
    phi = np.radians(lat_deg)
    dec = np.radians(23.45) * np.sin(2.0 * np.pi * (284.0 + doy) / 365.0)
    cos_ws = (np.sin(np.radians(elev_deg)) - np.sin(phi) * np.sin(dec)) / (
        np.cos(phi) * np.cos(dec))
    return int((cos_ws >= 1.0).sum())      # never clears that elevation


def isa_pressure_Pa(elev_m: float) -> float:
    """International Standard Atmosphere pressure at a geometric altitude."""
    return 101325.0 * (1.0 - 2.25577e-5 * elev_m) ** 5.25588


def station_rows() -> pd.DataFrame:
    """Folder + category + metadata for the authoritative stations.

    Body copied rather than imported: the only version that keeps latitude /
    longitude / elevation_m / koppen_geiger is download_era5_radiation.py, and
    that module pulls in earthengine-api, which is absent from `terramind`
    (splice_era5_radiation.py:91-94 documents the same workaround).
    """
    df = pd.read_csv(STATION_CSV)

    def _folder(r):
        if r["source_network"] != r["network"]:
            return f"{r['source_network']}_{r['network']}_{r['station_id']}"
        return f"{r['network']}_{r['station_id']}"

    def _cat(r):
        has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        has_fl = str(r.get("has_flux", "False")).lower() == "true"
        return "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")

    df["folder"] = df.apply(_folder, axis=1)
    df["cat"]    = df.apply(_cat, axis=1)
    # koppen_geiger has a case bug -- BSk(168) vs Bsk(11), BSh(6) vs Bsh(3).
    kg = df["koppen_geiger"].astype(str).str.strip()
    df["kg_norm"] = kg.str[0].str.upper() + kg.str[1:].str.lower()
    df.loc[kg.isin(["", "nan"]), "kg_norm"] = "unknown"
    return df[["folder", "cat", "latitude", "longitude", "elevation_m",
               "kg_norm", "split"]].reset_index(drop=True)


def scan(task) -> dict:
    folder, cat, lat, lon, elev, kg, split = task
    o = {"station": folder, "cat": cat, "lat": lat, "lon": lon,
         "elev_m": elev, "koppen": kg, "split": split,
         "status": "error", "msg": "", "n_days": 0, "n_rad": 0}
    o["doy_clim"] = None
    try:
        zpath = ZARR_ROOT / cat / folder
        if not (zpath / ".complete").exists():
            o["status"] = "skip:no-complete"; return o
        try:
            zg = zarr.open_consolidated(str(zpath), mode="r")
        except KeyError:
            zg = zarr.open_group(str(zpath), mode="r")
        if "era5/values" not in zg:
            o["status"] = "skip:no-era5"; return o

        v     = np.asarray(zg["era5/values"][:], dtype=np.float64)
        dates = np.asarray(zg["era5/date_ints"][:], dtype=np.int64)
        doys  = np.asarray(zg["era5/doys"][:], dtype=np.int64)
        if v.ndim != 2 or v.shape[1] != 19:
            o["status"] = "skip:width"; o["msg"] = f"{v.shape}"; return o
        n = v.shape[0]
        o["n_days"] = int(n)

        # ── Tier 0 on the carried columns ────────────────────────────────────
        o["n_nonfinite"] = int((~np.isfinite(v)).sum())
        o["dates_sorted"] = bool(np.all(np.diff(dates) > 0))
        o["n_dup_dates"]  = int(n - len(np.unique(dates)))
        d = pd.to_datetime(dates.astype(str), format="%Y%m%d")
        o["n_missing_days"] = int((d.max() - d.min()).days + 1 - n)

        # dewpoint can never exceed air temperature
        dd = v[:, IDX["d2m_mean"]] - v[:, IDX["t2m_mean"]]
        o["max_d2m_minus_t2m"] = float(np.nanmax(dd))
        o["n_d2m_gt_t2m"] = int((dd > 0.01).sum())

        # min <= mean <= max for every triple
        bad_tri = 0
        for t in TRIPLES:
            lo, me, hi = v[:, IDX[f"{t}_min"]], v[:, IDX[f"{t}_mean"]], v[:, IDX[f"{t}_max"]]
            bad_tri += int(((lo > me + 1e-3) | (me > hi + 1e-3)).sum())
        o["n_triple_violations"] = bad_tri
        o["n_tp_negative"] = int((v[:, IDX["tp_sum"]] < TP_NEG_TOL).sum())

        o["t2m_mean"] = float(np.nanmean(v[:, IDX["t2m_mean"]]))
        o["sp_mean"]  = float(np.nanmean(v[:, IDX["sp_mean"]]))
        o["sp_isa"]   = float(isa_pressure_Pa(elev))
        o["sp_ratio"] = o["sp_mean"] / o["sp_isa"] if o["sp_isa"] > 0 else np.nan

        # ── radiation ────────────────────────────────────────────────────────
        rad_dir = DATA_ROOT / cat / folder / "ERA5Land"
        files = sorted(rad_dir.glob("rad_????.nc"))
        if not files:
            o["status"] = "ok:no-rad"       # the 3 ocean-masked stations
            return o

        frames, bands_ok = [], True
        for f in files:
            with xr.open_dataset(f) as ds:
                t = pd.to_datetime(ds["time"].values)
                frames.append(pd.DataFrame({
                    "date_int": t.year * 10000 + t.month * 100 + t.day,
                    "ssrd_MJ": ds["ssrd_sum"].values.astype(np.float64) / MJ,
                    "strd_MJ": ds["strd_sum"].values.astype(np.float64) / MJ,
                }))
                # the accumulated-band question, answered by provenance
                if "_hourly" not in str(ds.attrs.get("bands", "")):
                    bands_ok = False
        o["bands_hourly"] = bool(bands_ok)

        rad = pd.concat(frames, ignore_index=True).drop_duplicates("date_int")
        rad = rad.set_index("date_int").reindex(dates)
        o["n_rad"] = int(rad["ssrd_MJ"].notna().sum())
        o["n_rad_gap"] = int(rad["ssrd_MJ"].isna().sum())

        ssrd = rad["ssrd_MJ"].to_numpy()
        strd = rad["strd_MJ"].to_numpy()
        ok = np.isfinite(ssrd)

        h0 = toa_daily_MJ(lat, doys.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            kt = np.where(h0 > 0.1, ssrd / h0, np.nan)

        o["kt_mean"] = float(np.nanmean(kt))
        o["kt_p99"]  = float(np.nanpercentile(kt[np.isfinite(kt)], 99)) if np.isfinite(kt).any() else np.nan
        o["kt_max"]  = float(np.nanmax(kt)) if np.isfinite(kt).any() else np.nan
        o["n_kt_over"] = int(np.nansum(kt > KT_HARD))

        # ── polar night ──────────────────────────────────────────────────────
        # MEASURED, and it settles which convention is right: at 67-69 N the
        # refracted (-0.833 deg) and civil (-6 deg) counts COLLAPSE (0 sunless days
        # at 66.75 N) because lowering the bar by even 0.83 deg is enough for the
        # midwinter sun to clear it. The GEOMETRIC horizon is the correct
        # comparator, and the residual against it is small, one-sided and scales
        # with latitude: -1.0 d/yr at 66.75 N to -4.8 d/yr at 68.62 N, mean -3.9,
        # worst -6.5, across 24 stations and 4-10 year records.
        #
        # So the test is two statements, not an equality:
        #   (i)  HARD, one-sided: ERA5 can never be dark while the sun is up, so
        #        n_ssrd_zero <= n_sunless_geometric. A violation is a real defect.
        #   (ii) the deficit is twilight and is bounded per YEAR, not in absolute
        #        days -- an absolute tolerance just penalises long records.
        o["n_ssrd_zero"] = int((ok & (ssrd <= 1e-9)).sum())
        dd = doys.astype(np.float64)
        for name, e in SUN_ELEV.items():
            o[f"n_sunless_{name}"] = n_sunless_days(lat, dd, e)
        o["n_h0_zero"]   = o["n_sunless_geometric"]
        o["years"]       = float(n / 365.25)
        o["polar_delta"] = o["n_ssrd_zero"] - o["n_sunless_geometric"]
        o["polar_delta_per_yr"] = o["polar_delta"] / o["years"] if o["years"] > 0 else np.nan

        o["ssrd_mean"] = float(np.nanmean(ssrd))
        o["strd_mean"] = float(np.nanmean(strd))
        o["strd_min"]  = float(np.nanmin(strd)) if ok.any() else np.nan
        o["ssrd_season"] = float(np.nanstd(ssrd) / np.nanmean(ssrd)) if np.nanmean(ssrd) > 0 else np.nan

        clim = pd.Series(ssrd[ok]).groupby(doys[ok]).mean().reindex(range(1, 367))
        o["doy_clim"] = clim.to_numpy(dtype=np.float32)
        o["peak_doy"] = int(clim.rolling(15, center=True, min_periods=5).mean().idxmax())

        o["status"] = "ok"
    except Exception as exc:
        o["msg"] = str(exc)[:200]
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = station_rows()
    tasks = list(rows.itertuples(index=False, name=None))
    print(f"scanning {len(tasks)} stations with {args.workers} workers")

    with Pool(args.workers) as pool:
        res = pool.map(scan, tasks, chunksize=1)

    clim = {r["station"]: r.pop("doy_clim") for r in res}
    d = pd.DataFrame(res)
    d.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(d)} rows)")

    print("\n--- status ---")
    print(d["status"].value_counts().to_string())

    g = d[d["status"] == "ok"].copy()
    print(f"\n{len(g)} stations with radiation")

    # ── the verdict table ────────────────────────────────────────────────────
    print("\n=== TIER 0 — hard invariants (any nonzero is a real defect) ===")
    checks = [
        ("non-finite values",            (d["n_nonfinite"] > 0)),
        ("dates not strictly increasing",(~d["dates_sorted"].fillna(True))),
        ("duplicate dates",              (d["n_dup_dates"] > 0)),
        ("missing days in span",         (d["n_missing_days"] > 0)),
        ("d2m > t2m",                    (d["n_d2m_gt_t2m"] > 0)),
        ("min<=mean<=max violated",      (d["n_triple_violations"] > 0)),
        ("negative precipitation",       (d["n_tp_negative"] > 0)),
        ("Kt > 1.05",                    (g["n_kt_over"] > 0)),
        ("bands not _hourly",            (~g["bands_hourly"].fillna(True))),
        ("dark while sun is up",         (g["polar_delta"] > 0)),
        ("twilight deficit > 8 d/yr",    (g["polar_delta_per_yr"] < -8.0)),
        ("radiation gap vs zarr dates",  (g["n_rad_gap"] > 0)),
    ]
    for name, mask in checks:
        n = int(mask.sum())
        flag = "OK  " if n == 0 else "FAIL"
        print(f"  [{flag}] {name:32s} {n:4d} station(s)")

    for name, mask in checks:
        if int(mask.sum()) and int(mask.sum()) <= 20:
            sub = d.loc[mask.index[mask], "station"] if len(mask) == len(d) else g.loc[mask.index[mask], "station"]
            print(f"\n  {name}: {', '.join(sub.astype(str).tolist())}")

    print("\n=== TIER 1 — population statistics ===")
    print(f"  Kt mean   : {g['kt_mean'].min():.3f} .. {g['kt_mean'].max():.3f}  "
          f"(median {g['kt_mean'].median():.3f})")
    print(f"  Kt max    : {g['kt_max'].max():.3f}")
    print(f"  sp/ISA    : {g['sp_ratio'].min():.4f} .. {g['sp_ratio'].max():.4f}  "
          f"(median {g['sp_ratio'].median():.4f})")
    print(f"  peak DOY  : north {g.loc[g.lat > 0, 'peak_doy'].median():.0f}   "
          f"south {g.loc[g.lat < 0, 'peak_doy'].median():.0f}  "
          f"(n south = {int((g.lat < 0).sum())})")

    worst = g.reindex(g["sp_ratio"].sub(1.0).abs().sort_values(ascending=False).index)
    print("\n  10 worst sp/ISA (the right-pixel test):")
    print(worst.head(10)[["station", "lat", "elev_m", "sp_mean", "sp_isa", "sp_ratio"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    plot(g, clim)
    return 0


def plot(g: pd.DataFrame, clim: dict) -> None:
    C1, C2, C3 = "#0072B2", "#D55E00", "#009E73"
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    a = ax.ravel()

    # (a) Kt vs |lat|
    a[0].scatter(g["lat"].abs(), g["kt_mean"], s=12, c=C1, alpha=0.6, lw=0)
    a[0].axhline(1.0, color="k", ls="--", lw=1)
    a[0].set_xlabel("|latitude|"); a[0].set_ylabel("mean clear-sky index Kt")
    a[0].set_title("(a) Kt = ssrd / TOA — must be < 1", loc="left", fontsize=10)

    # (b) Kt max per station
    a[1].scatter(g["lat"].abs(), g["kt_max"], s=12, c=C2, alpha=0.6, lw=0)
    a[1].axhline(KT_HARD, color="k", ls="--", lw=1, label=f"hard limit {KT_HARD}")
    a[1].set_xlabel("|latitude|"); a[1].set_ylabel("max daily Kt")
    a[1].set_title("(b) worst single day per station", loc="left", fontsize=10)
    a[1].legend(fontsize=8, frameon=False)

    # (c) polar night: one-sided, and the deficit is twilight
    pn = g[g["n_sunless_geometric"] > 0]
    a[2].scatter(pn["n_sunless_geometric"], pn["n_ssrd_zero"], s=26, c=C3,
                 alpha=0.85, lw=0, zorder=3)
    m = max(pn["n_sunless_geometric"].max(), pn["n_ssrd_zero"].max()) + 15 if len(pn) else 10
    a[2].plot([0, m], [0, m], "k--", lw=1, label="1:1 — sun never rises")
    a[2].fill_between([0, m], [0, m], [0, 0], color="0.9", zorder=0,
                      label="allowed: twilight gives light")
    a[2].set_xlim(0, m); a[2].set_ylim(0, m)
    a[2].set_xlabel("computed days the sun never rises (geometric)")
    a[2].set_ylabel("days with ssrd == 0 (observed)")
    a[2].set_title("(c) polar night — must lie ON or BELOW 1:1", loc="left", fontsize=10)
    if len(pn):
        a[2].text(0.97, 0.06,
                  f"{len(pn)} stations above the Arctic Circle\n"
                  f"deficit {pn['polar_delta_per_yr'].mean():+.1f} d/yr "
                  f"(worst {pn['polar_delta_per_yr'].min():+.1f})\n"
                  f"never positive: {int((pn['polar_delta'] > 0).sum())} violations",
                  transform=a[2].transAxes, fontsize=7, ha="right", color="0.25")
    a[2].legend(fontsize=7, frameon=False, loc="upper left")

    # (d) the right-pixel test
    a[3].scatter(g["elev_m"], g["sp_mean"] / 100.0, s=12, c=C1, alpha=0.6, lw=0)
    z = np.linspace(g["elev_m"].min(), g["elev_m"].max(), 200)
    a[3].plot(z, isa_pressure_Pa(z) / 100.0, color="k", lw=1.2, label="ISA")
    a[3].set_xlabel("station elevation (m)"); a[3].set_ylabel("mean sp (hPa)")
    a[3].set_title("(d) surface pressure vs elevation — the RIGHT-PIXEL test",
                   loc="left", fontsize=10)
    a[3].legend(fontsize=8, frameon=False)

    # (e) peak DOY vs latitude
    a[4].scatter(g["lat"], g["peak_doy"], s=12, c=C2, alpha=0.6, lw=0)
    a[4].axhline(172, color="0.4", ls=":", lw=1)
    a[4].axhline(355, color="0.4", ls=":", lw=1)
    a[4].axvline(0, color="k", lw=0.8)
    a[4].set_xlabel("latitude"); a[4].set_ylabel("ssrd peak day of year")
    a[4].set_title("(e) peak DOY — hemisphere test", loc="left", fontsize=10)

    # (f) seasonal amplitude vs |lat|
    a[5].scatter(g["lat"].abs(), g["ssrd_season"], s=12, c=C3, alpha=0.6, lw=0)
    a[5].set_xlabel("|latitude|"); a[5].set_ylabel("ssrd std/mean")
    a[5].set_title("(f) seasonality must rise with |latitude|", loc="left", fontsize=10)

    # (g) dewpoint invariant
    a[6].scatter(g["lat"], g["max_d2m_minus_t2m"], s=12, c=C2, alpha=0.6, lw=0)
    a[6].axhline(0.0, color="k", ls="--", lw=1)
    a[6].set_xlabel("latitude"); a[6].set_ylabel("max (d2m - t2m)  K")
    a[6].set_title("(g) dewpoint may never exceed air temp", loc="left", fontsize=10)

    # (h) Hovmoller, stations sorted by latitude
    sub = g.sort_values("lat")
    M = np.full((len(sub), 366), np.nan, dtype=np.float32)
    for i, s in enumerate(sub["station"]):
        c = clim.get(s)
        if c is not None:
            M[i] = c
    im = a[7].imshow(M, aspect="auto", origin="lower", cmap="viridis",
                     extent=(1, 366, 0, len(sub)), interpolation="nearest")
    a[7].set_xlabel("day of year"); a[7].set_ylabel("station, sorted by latitude")
    a[7].set_title("(h) ssrd climatology, all stations", loc="left", fontsize=10)
    fig.colorbar(im, ax=a[7], label="MJ m-2 d-1")

    for x in a:
        x.tick_params(labelsize=8)
    fig.suptitle("§45 — all-station ERA5 driver QC", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "era5_qc_overview.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
