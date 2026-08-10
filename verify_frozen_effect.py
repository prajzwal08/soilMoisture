"""§22.12 -- is the SNOTEL penalty a frozen-soil effect?

§22.11 showed SNOTEL stations carry a higher ubRMSE than everything else at every
depth in every split, and that the apparent "Forest is worse" signal disappears once
you condition on SNOTEL membership.  But "SNOTEL" bundles snow, elevation, mountain
terrain and a particular dielectric sensor, so it is a label and not a mechanism.

This splits every station's own record into frozen and unfrozen days using ERA5
skin temperature (skt_min < 0 C) and recomputes ubRMSE on each subset.  Each
station acts as its own control, so elevation, land cover, sensor type and the
absolute level all cancel -- only the seasonal state varies.

Two questions:
    1. within a station, is ubRMSE higher on frozen days than unfrozen ones?
    2. does the SNOTEL-vs-rest gap survive when frozen days are removed?
       If it collapses, the penalty is frozen soil.  If it persists, it is
       something else about those sites and needs a separate explanation.

ERA5 carries no snow depth or soil temperature (see dataset.ERA5_VARS), so
skin temperature is the available freeze proxy; --var t2m_min gives a stricter one.

CPU only.  Needs pyarrow -> run under the terramind env.

    python verify_frozen_effect.py [--workers 32] [--var skt_min] [--min-days 30]
"""
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ZARR_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
CATEGORIES = ["sm_only", "sm_and_flux", "flux_only"]
ERA5_VARS  = [
    "t2m_mean", "t2m_min", "t2m_max", "d2m_mean", "d2m_min", "d2m_max",
    "skt_mean", "skt_min", "skt_max", "u10_mean", "u10_min", "u10_max",
    "v10_mean", "v10_min", "v10_max", "sp_mean", "sp_min", "sp_max", "tp_sum",
]
SM_DEPTHS = ["0-10", "10-30", "30-100"]
SPLITS    = ["oos", "oot", "oost"]
KELVIN    = 273.15


def read_freeze_mask(station: str, var: str):
    """-> DataFrame [station_key, date_int, frozen] or None if no ERA5 for it."""
    idx = ERA5_VARS.index(var)
    for cat in CATEGORIES:
        path = ZARR_ROOT / cat / station
        if not path.exists():
            continue
        try:
            zg = zarr.open(str(path), mode="r")
            if "era5/values" not in zg:
                return None
            v = zg["era5/values"][:, idx].astype(np.float64)
            d = zg["era5/date_ints"][:]
        except Exception:                                        # noqa: BLE001
            return None
        # ERA5 is stored in Kelvin; guard in case a future rebuild stores Celsius
        thresh = KELVIN if np.nanmedian(v) > 100 else 0.0
        return pd.DataFrame({"station_key": station, "date_int": d.astype(np.int64),
                             "frozen": v < thresh})
    return None


def ubrmse(p, o):
    e = p - o
    return float(np.sqrt(np.mean((e - e.mean()) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",   default="eval_output")
    ap.add_argument("--out-dir",  default="eval_output")
    ap.add_argument("--var",      default="skt_min",
                    choices=["skt_min", "skt_mean", "t2m_min", "t2m_mean"])
    ap.add_argument("--min-days", type=int, default=30,
                    help="days required in BOTH subsets for a station-depth to count")
    ap.add_argument("--workers",  type=int, default=32)
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    preds = {}
    for split in SPLITS:
        p = in_dir / f"predictions_{split}.parquet"
        if p.exists():
            preds[split] = pd.read_parquet(p)
    stations = sorted({s for df in preds.values()
                       for s in df["station_key"].unique()})
    print(f"{len(stations)} stations across {len(preds)} splits")

    with Pool(args.workers) as pool:
        masks = pool.map(partial(read_freeze_mask, var=args.var), stations)
    missing = [s for s, m in zip(stations, masks) if m is None]
    mask = pd.concat([m for m in masks if m is not None], ignore_index=True)
    print(f"freeze mask from {args.var}: {mask.station_key.nunique()} stations"
          + (f"  ({len(missing)} without ERA5, dropped)" if missing else ""))

    rows = []
    for split, df in preds.items():
        df = df.copy()
        df["date_int"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d").astype(np.int64)
        df = df.merge(mask, on=["station_key", "date_int"], how="inner")
        for (st, d), g in df.groupby(["station_key", "depth"], observed=True):
            fr, un = g[g.frozen], g[~g.frozen]
            if len(fr) < args.min_days or len(un) < args.min_days:
                continue
            rows.append(dict(
                split=split, station_key=st, depth=d,
                network=st.split("_")[1] if st.startswith("ISMN_") else st.split("_")[0],
                n_frozen=len(fr), n_unfrozen=len(un),
                frac_frozen=len(fr) / len(g),
                ub_frozen=ubrmse(fr["pred"].to_numpy(float), fr["obs"].to_numpy(float)),
                ub_unfrozen=ubrmse(un["pred"].to_numpy(float), un["obs"].to_numpy(float)),
                ub_all=ubrmse(g["pred"].to_numpy(float), g["obs"].to_numpy(float))))
    s = pd.DataFrame(rows)
    s["snotel"] = np.where(s.network == "SNOTEL", "SNOTEL", "other")
    s["delta"]  = s.ub_frozen - s.ub_unfrozen
    s.to_csv(Path(args.out_dir) / "frozen_effect_station_depth.csv", index=False)

    print(f"\n=== 1. within-station: frozen vs unfrozen ubRMSE "
          f"(>= {args.min_days} days in each) ===")
    print(f"{'split':5s} {'depth':7s} {'stn':>4s} {'ub frozen':>10s} {'ub unfroz':>10s} "
          f"{'med delta':>10s} {'frozen worse':>13s} {'med frac froz':>14s}")
    for split in SPLITS:
        for d in SM_DEPTHS:
            g = s[(s.split == split) & (s.depth == d)]
            if g.empty:
                continue
            print(f"{split:5s} {d:7s} {len(g):4d} {g.ub_frozen.median():10.4f} "
                  f"{g.ub_unfrozen.median():10.4f} {g.delta.median():+10.4f} "
                  f"{(g.delta > 0).mean():13.0%} {g.frac_frozen.median():14.2f}")

    try:
        from scipy.stats import wilcoxon
        print("\n  paired Wilcoxon (frozen vs unfrozen, all splits pooled by depth):")
        for d in SM_DEPTHS:
            g = s[s.depth == d]
            if len(g) > 10:
                stat, p = wilcoxon(g.ub_frozen, g.ub_unfrozen)
                print(f"    {d:7s} n={len(g):4d}  p={p:.2e}")
    except ImportError:
        print("  (scipy unavailable -- Wilcoxon skipped)")

    print("\n=== 2. does the SNOTEL gap survive on unfrozen days only? ===")
    print(f"{'split':5s} {'depth':7s} {'group':7s} {'stn':>4s} {'ub all':>8s} "
          f"{'ub unfrozen':>12s}")
    for split in SPLITS:
        for d in SM_DEPTHS:
            for grp in ["SNOTEL", "other"]:
                g = s[(s.split == split) & (s.depth == d) & (s.snotel == grp)]
                if len(g) < 3:
                    continue
                print(f"{split:5s} {d:7s} {grp:7s} {len(g):4d} "
                      f"{g.ub_all.median():8.4f} {g.ub_unfrozen.median():12.4f}")
        g = s[(s.split == split)]
        for d in SM_DEPTHS:
            a = g[(g.depth == d) & (g.snotel == "SNOTEL")]
            b = g[(g.depth == d) & (g.snotel == "other")]
            if len(a) >= 3 and len(b) >= 3:
                print(f"      -> {d:7s} gap all-days "
                      f"{a.ub_all.median() - b.ub_all.median():+.4f}   "
                      f"gap unfrozen "
                      f"{a.ub_unfrozen.median() - b.ub_unfrozen.median():+.4f}")

    print(f"\n=== 3. exposure: median frozen-day fraction by network group ===")
    for grp, g in s.groupby("snotel"):
        print(f"  {grp:7s} n={g.station_key.nunique():4d} "
              f"median frac frozen={g.frac_frozen.median():.2f}")
    print(f"\n→ {Path(args.out_dir) / 'frozen_effect_station_depth.csv'}")


if __name__ == "__main__":
    main()
