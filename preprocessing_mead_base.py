# ============================================================
# Mead (US-Ne1/Ne2/Ne3) daily soil moisture from AmeriFlux BASE
# ============================================================
# Runbook §69 step 1.  The three Mead sites share soil and weather and
# differ in water management (Ne1 irrigated continuous maize, Ne2
# irrigated maize/soy, Ne3 rainfed maize/soy), which makes them a
# controlled irrigation test.  §44.8 reserved them for validation.
#
# preprocessing_ameriflux.py reads the FLUXNET product (SWC_F_MDS_n +
# FLX15 depths) and cannot read BASE, so this is a BASE reader that
# applies the same rules:
#   - RAW SWC_H_V_R columns only (SWC_PI_F_* are PI gap-filled; we keep
#     measured values, like the ISMN G/D whitelist)
#   - % -> fraction, physical range 0 < theta <= 0.60
#   - daily mean, day masked if < MIN_OBS_PER_DAY hourly values
#   - sensor depth from the AmeriFlux BASE Measurement Height product
#     (the BIF has no VAR_INFO): TDR at 10/25/50/100 cm, Ne3 also 175 cm
#     (below 100 cm, dropped); a depth that changes over time splits the
#     sensor into segments
#   - irrigation log (BIF GRP_DM_WATER) written out per event; some years
#     say "No irrigation over Towers", which is what the sensors see
#   - profiles averaged per depth, depths binned 0-10/10-30/30-100 cm
#
# Two outputs per site, both under NEW roots (never level1/splits/stores):
#   observed/  observed-only daily SM (bins + native depths), sensor
#              counts, tower precipitation (daily sum), LE (daily mean).
#              No gap-fill: the climatological fill would smear
#              irrigation events.
#   level1/    the training-label pipeline (2017-2025 slice, gap
#              analysis, trim, month-day gap-fill), for model scoring.
#
# Precipitation is the tower gauge in BASE (P_PI_F_*); FLUXNET P_F is
# the same gauge with ERA gap-filling, and Ne3 has no FLUXNET release.
#
# The BIF (management log only) is read with the standard library
# (xlsx = zip of XML) because openpyxl is in neither conda env.
#
# Usage:
#   python preprocessing_mead_base.py [--sites US-Ne1 US-Ne2 US-Ne3]
# ============================================================

import argparse
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from utils import (
    longest_available_after_removing_long_gaps,
    trim_to_surface_valid_period_and_keep_well_covered_depths,
    gapfill_by_monthday_mean_with_feb29_fallback,
)

MEAD     = ["US-Ne1", "US-Ne2", "US-Ne3"]
ZIP_DIR    = Path("/projects/prjs1968/raw_soil_moisture/ameriflux_base_mead")
HEIGHT_CSV = ZIP_DIR / "BASE_MeasurementHeight_20260908.csv"   # SWC depths
OUT_ROOT = Path("/gpfs/work3/0/prjs1968/data_mead")

MISSING         = -9999.0
SWC_MAX         = 0.60
MIN_OBS_PER_DAY = 6          # same rule as ISMN / preprocessing_ameriflux.py
MIN_HOURS_FLUX  = 20         # P / LE day valid if >= 20 of 24 hourly values
DEPTH_BINS      = [0.0, 10.0, 30.0, 100.0]
DEPTH_LABELS    = ["0-10", "10-30", "30-100"]
MIN_VALID_DAYS  = 1095
TRAIN_START     = "2017-01-01"
TRAIN_END       = "2025-12-31"

SWC_RAW = re.compile(r"^SWC_\d+_\d+_\d+$")
P_COLS  = ["P_PI_F_1_1_1", "P_PI_F_2_2_1"]
LE_COLS = ["LE_1_1_1", "LE_PI_F_1_1_1"]   # LE_PI_F stops after 2014
TS_COL  = "TS_1_1_1"                       # shallow soil temperature, frozen-soil flag

XNS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# ---------------------------------------------------------------------------
# BIF (xlsx) reader, stdlib only
# ---------------------------------------------------------------------------
def _col_index(ref: str) -> int:
    idx = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def read_xlsx_first_sheet(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as x:
        shared = []
        if "xl/sharedStrings.xml" in x.namelist():
            root = ET.fromstring(x.read("xl/sharedStrings.xml"))
            for si in root.iter(f"{XNS}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{XNS}t")))
        sheet = sorted(n for n in x.namelist()
                       if re.match(r"xl/worksheets/sheet\d+\.xml$", n))[0]
        root = ET.fromstring(x.read(sheet))

    rows = []
    for r in root.iter(f"{XNS}row"):
        vals = {}
        for c in r.findall(f"{XNS}c"):
            t, v = c.get("t"), c.find(f"{XNS}v")
            if t == "s":
                val = shared[int(v.text)] if v is not None else ""
            elif t == "inlineStr":
                val = "".join(tt.text or "" for tt in c.iter(f"{XNS}t"))
            else:
                val = v.text if v is not None else ""
            vals[_col_index(c.get("r"))] = val
        rows.append(vals)

    width  = max(max(r) for r in rows if r) + 1
    matrix = [[r.get(i, "") for i in range(width)] for r in rows]
    return pd.DataFrame(matrix[1:], columns=matrix[0])


def swc_depth_segments(site: str) -> pd.DataFrame:
    """One row per (varname, valid-from date, depth_cm) from the AmeriFlux
    BASE Measurement Height product (the BIF carries no GRP_VAR_INFO)."""
    mh = pd.read_csv(HEIGHT_CSV, dtype=str)
    mh = mh[(mh["Site_ID"] == site) & mh["Variable"].str.match(SWC_RAW.pattern)]
    seg = pd.DataFrame({
        "var":      mh["Variable"].values,
        "start":    pd.to_datetime(mh["Start_Date"].str[:8], format="%Y%m%d",
                                   errors="coerce").values,
        "depth_cm": (mh["Height"].astype(float).abs() * 100.0).round(1).values,
        "model":    mh["Instrument_Model"].values,
    })
    if seg.empty:
        return seg
    seg["start"] = seg["start"].fillna(pd.Timestamp("1900-01-01"))
    return (seg.drop_duplicates(["var", "start"], keep="last")
               .sort_values(["var", "start"]).reset_index(drop=True))


def irrigation_events(bif: pd.DataFrame, site: str) -> pd.DataFrame:
    """GRP_DM_WATER records: date, amount (inch, parsed from comment), and
    whether the comment says the tower area was NOT irrigated."""
    sub = bif[bif["VARIABLE_GROUP"] == "GRP_DM_WATER"]
    recs = []
    for gid, g in sub.groupby("GROUP_ID"):
        f = dict(zip(g["VARIABLE"], g["DATAVALUE"]))
        start = f.get("DM_DATE_START") or f.get("DM_DATE") or ""
        end   = f.get("DM_DATE_END") or start
        com   = str(f.get("DM_COMMENT", ""))
        # ".33 in" must parse as 0.33 (a leading-dot amount read as 33 before)
        amt   = re.search(r"(\d*\.\d+|\d+)\s*(?:in\b|in\.|inch)", com)
        cancelled = bool(re.search(r"cancel", com, re.I))
        recs.append({
            "site": site, "group_id": gid, "type": f.get("DM_WATER", ""),
            "date_start": pd.to_datetime(str(start)[:8], format="%Y%m%d", errors="coerce"),
            "date_end":   pd.to_datetime(str(end)[:8], format="%Y%m%d", errors="coerce"),
            "amount_in": 0.0 if cancelled else (float(amt.group(1)) if amt else np.nan),
            "cancelled": cancelled,
            "not_over_tower": bool(re.search(r"no irrigation over tower", com, re.I)),
            "comment": com,
        })
    return pd.DataFrame(recs).sort_values("date_start") if recs else pd.DataFrame()


# ---------------------------------------------------------------------------
# Hourly -> daily
# ---------------------------------------------------------------------------
def load_site(site: str):
    zpath = ZIP_DIR / f"{site}_BASE-BADM.zip"
    with zipfile.ZipFile(zpath) as zf:
        csv_name = [n for n in zf.namelist() if re.search(r"_BASE_H[HR]_.*\.csv$", n)][0]
        bif_name = [n for n in zf.namelist() if n.endswith(".xlsx")][0]
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh, skiprows=2, na_values=[MISSING, str(int(MISSING))])
        bif = read_xlsx_first_sheet(zf.read(bif_name))
    ts = pd.to_datetime(df["TIMESTAMP_START"].astype("Int64").astype(str),
                        format="%Y%m%d%H%M", errors="coerce")
    df.index = ts
    df = df[df.index.notna()]
    step_h = pd.Series(df.index).diff().median() / pd.Timedelta("1h")
    return df, bif, csv_name, float(step_h)


def sensor_daily(df: pd.DataFrame, segments: pd.DataFrame, report: list) -> tuple:
    """Daily mean per (sensor segment); returns (daily DataFrame, {col: depth_cm})."""
    daily, depth_of = {}, {}
    for var in sorted(c for c in df.columns if SWC_RAW.match(c)):
        s = df[var].astype(float)
        valid = s.dropna()
        if valid.empty:
            report.append(f"    {var:<12} empty")
            continue
        if valid.quantile(0.99) > 1.5:            # percent -> fraction
            s = s / 100.0
        n_out = int(((s <= 0) | (s > SWC_MAX)).sum())
        s = s.where((s > 0) & (s <= SWC_MAX))

        segs = segments[segments["var"] == var]
        if segs.empty:
            report.append(f"    {var:<12} NO BIF DEPTH -> dropped")
            continue
        bounds = list(segs["start"]) + [pd.Timestamp("2100-01-01")]
        for k, (_, row) in enumerate(segs.iterrows()):
            part = s[(s.index >= bounds[k]) & (s.index < bounds[k + 1])]
            if part.dropna().empty:
                continue
            cnt = part.resample("D").count()
            dm  = part.resample("D").mean()
            dm[cnt < MIN_OBS_PER_DAY] = np.nan
            key = f"{var}@{k}"
            daily[key], depth_of[key] = dm, row["depth_cm"]
            report.append(
                f"    {var:<12} seg{k} from {row['start'].date()}  depth {row['depth_cm']:>6.1f} cm"
                f"  valid days {int(dm.notna().sum()):>5}  mean {dm.mean():.3f}"
                f"  out-of-range hrs {n_out}")
    return pd.DataFrame(daily), depth_of


def bin_depths(daily: pd.DataFrame, depth_of: dict) -> tuple:
    """Average profiles per native depth, then bin native depths."""
    native = {}
    for d in sorted(set(depth_of.values())):
        cols = [c for c in daily.columns if depth_of[c] == d]
        native[d] = daily[cols].mean(axis=1)
    native = pd.DataFrame(native)

    binned, n_sens = {}, {}
    for d in native.columns:
        lbl = pd.cut([d], bins=DEPTH_BINS, labels=DEPTH_LABELS,
                     right=True, include_lowest=True)[0]
        if pd.isna(lbl):
            continue
        binned.setdefault(str(lbl), []).append(native[d])
    for lbl in DEPTH_LABELS:
        if lbl in binned:
            n_sens[lbl] = pd.concat(binned[lbl], axis=1).notna().sum(axis=1)
            binned[lbl] = pd.concat(binned[lbl], axis=1).mean(axis=1)
    binned = pd.DataFrame({k: v for k, v in binned.items()})
    return native, binned[[l for l in DEPTH_LABELS if l in binned]], pd.DataFrame(n_sens)


def daily_sum_mean(s: pd.Series, how: str, step_h: float) -> pd.Series:
    """Daily sum or mean; NaN unless >= MIN_HOURS_FLUX hours are present."""
    need = int(np.ceil(MIN_HOURS_FLUX / step_h))
    cnt = s.resample("D").count()
    out = s.resample("D").sum(min_count=1) if how == "sum" else s.resample("D").mean()
    return out.where(cnt >= need)


# ---------------------------------------------------------------------------
# Training-label pipeline (same as preprocessing_ameriflux._run_swc_pipeline)
# ---------------------------------------------------------------------------
def make_swc_dataset(binned: pd.DataFrame) -> xr.Dataset:
    return xr.Dataset(
        {"soil_moisture": xr.DataArray(
            binned.values, dims=["date_time", "depth"],
            attrs={"units": "m3 m-3", "long_name": "Volumetric soil moisture content",
                   "standard_name": "volume_fraction_of_condensed_water_in_soil",
                   "valid_range": [0.0, 1.0]})},
        coords={"date_time": binned.index.values, "depth": list(binned.columns)},
    )


def run_label_pipeline(ds: xr.Dataset):
    surface = str(ds["depth"].values[0])
    longest = longest_available_after_removing_long_gaps(
        ds, depth_dim="depth", time_dim="date_time", var="soil_moisture", max_gap_days=7)
    L, _, _ = longest.get(surface, (0, None, None))
    if L < MIN_VALID_DAYS:
        return None, f"surface run {L} d < {MIN_VALID_DAYS}"
    ds = trim_to_surface_valid_period_and_keep_well_covered_depths(
        ds, longest, surface_depth=surface, var="soil_moisture",
        depth_dim="depth", time_dim="date_time", min_frac=0.95)
    ds = gapfill_by_monthday_mean_with_feb29_fallback(
        ds, var="soil_moisture", time_dim="date_time", flag_name="soil_moisture_qc")
    if bool(ds["soil_moisture"].isnull().any()):
        return None, "NaNs remain after gap-fill"
    ds["soil_moisture_qc"].attrs.update({
        "long_name": "Soil moisture quality control flag",
        "flag_values": "0 1 2", "flag_meanings": "observed gap_filled missing"})
    return ds, f"surface run {L} d"


# ---------------------------------------------------------------------------
def process(site: str) -> str:
    rep = [f"=== {site} ==="]
    df, bif, csv_name, step_h = load_site(site)
    rep.append(f"  BASE {csv_name}  {df.index.min()} .. {df.index.max()}  step {step_h:.2f} h")

    (OUT_ROOT / "bif").mkdir(parents=True, exist_ok=True)
    bif.to_csv(OUT_ROOT / "bif" / f"{site}_bif_long.csv", index=False)
    mgmt = bif[bif["VARIABLE_GROUP"].str.contains(
        "AG|CROP|IRRIG|LAND|DM_|PLANT|HARV|FERT|SITE_DESC", case=False, na=False)]
    mgmt.to_csv(OUT_ROOT / "bif" / f"{site}_bif_management.csv", index=False)

    irr = irrigation_events(bif, site)
    irr.to_csv(OUT_ROOT / "bif" / f"{site}_irrigation_events.csv", index=False)
    if len(irr):
        yr = irr.assign(year=irr["date_start"].dt.year).groupby("year")
        rep.append("  irrigation events per year (n, inches, 'no irrigation over towers'):")
        for y, g in yr:
            rep.append(f"    {y}: {len(g):>2}  {g['amount_in'].sum():5.2f} in  "
                       f"not-over-tower {int(g['not_over_tower'].sum())}")
    else:
        rep.append("  irrigation events: none in BIF")

    seg = swc_depth_segments(site)
    seg.to_csv(OUT_ROOT / "bif" / f"{site}_swc_depths.csv", index=False)
    rep.append(f"  SWC depth records: {len(seg)}  depths {sorted(seg['depth_cm'].unique()) if len(seg) else []}")

    daily, depth_of = sensor_daily(df, seg, rep)
    if daily.empty:
        rep.append("  NO usable SWC")
        return "\n".join(rep)
    native, binned, n_sens = bin_depths(daily, depth_of)

    # tower precipitation (two gauges) + LE
    p = {c: daily_sum_mean(df[c].astype(float), "sum", step_h) for c in P_COLS if c in df}
    p_main = max(p, key=lambda c: p[c].notna().sum()) if p else None
    # LE: PI gap-filled where it exists, else the raw half of the record (12 h minimum)
    le_cols = [c for c in LE_COLS if c in df]
    le_h = (df[le_cols[1]].astype(float).combine_first(df[le_cols[0]].astype(float))
            if len(le_cols) == 2 else df[le_cols[0]].astype(float) if le_cols else None)
    le = (le_h.resample("D").mean().where(le_h.resample("D").count() >= 12 / step_h)
          if le_h is not None else None)
    ts = daily_sum_mean(df[TS_COL].astype(float), "mean", step_h) if TS_COL in df else None
    for c, s in p.items():
        w = s.loc[TRAIN_START:TRAIN_END]
        rep.append(f"  {c}: {int(w.notna().sum())}/{len(w)} valid days 2017-2025, "
                   f"mean annual {w.mean() * 365.25:.0f} mm")

    # ---- observed-only output
    idx = binned.index
    obs = xr.Dataset(coords={"date_time": idx.values})
    obs["soil_moisture"] = (("date_time", "depth"), binned.values)
    obs = obs.assign_coords(depth=list(binned.columns))
    obs["n_depths_in_bin"] = (("date_time", "depth"), n_sens.reindex(idx)[binned.columns].values)
    obs["soil_moisture_native"] = (("date_time", "depth_cm"), native.reindex(idx).values)
    obs = obs.assign_coords(depth_cm=[float(d) for d in native.columns])
    if p_main:
        obs["precip_mm"] = ("date_time", p[p_main].reindex(idx).values)
        obs["precip_mm"].attrs = {"units": "mm d-1", "source": f"tower gauge {p_main}"}
    if le is not None:
        obs["le_wm2"] = ("date_time", le.reindex(idx).values)
        obs["le_wm2"].attrs = {"units": "W m-2", "source": "+".join(LE_COLS)}
    if ts is not None:
        obs["ts_degc"] = ("date_time", ts.reindex(idx).values)
        obs["ts_degc"].attrs = {"units": "degC", "source": TS_COL,
                                "note": "TDR reads liquid water only: SM drops when ts <= 0"}
    obs["soil_moisture"].attrs = {"units": "m3 m-3", "note": "observed only, no gap-fill"}
    obs.attrs = {"site": site, "source": csv_name, "time_zone": "local standard time (BASE)"}
    (OUT_ROOT / "observed").mkdir(parents=True, exist_ok=True)
    obs.to_netcdf(OUT_ROOT / "observed" / f"AmeriFlux_{site}_observed.nc")

    for lbl in binned.columns:
        w = binned[lbl].loc[TRAIN_START:TRAIN_END]
        rep.append(f"  bin {lbl:>6}: {int(w.notna().sum())}/{len(w)} observed days 2017-2025, "
                   f"mean {w.mean():.3f}, p05 {w.quantile(.05):.3f}, p95 {w.quantile(.95):.3f}")

    # ---- training-label pipeline output
    ds = make_swc_dataset(binned.loc[TRAIN_START:TRAIN_END])
    ds, why = run_label_pipeline(ds)
    if ds is None:
        rep.append(f"  level1: SKIPPED ({why})")
    else:
        t = pd.to_datetime(ds["date_time"].values)
        (OUT_ROOT / "level1").mkdir(parents=True, exist_ok=True)
        name = f"AmeriFlux_{site}_{t.min():%Y%m%d}_{t.max():%Y%m%d}.nc"
        ds.to_netcdf(OUT_ROOT / "level1" / name)
        qc = ds["soil_moisture_qc"]
        rep.append(f"  level1: {name}  depths {list(ds['depth'].values)}  ({why}), "
                   f"gap-filled {float((qc == 1).mean()) * 100:.1f} %")
    return "\n".join(rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=MEAD)
    args = ap.parse_args()
    with Pool(len(args.sites)) as pool:
        for txt in pool.map(process, args.sites):
            print(txt, flush=True)


if __name__ == "__main__":
    main()
