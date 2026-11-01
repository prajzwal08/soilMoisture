#!/usr/bin/env python
"""Does S1 resolve soil moisture INSIDE a tile, across every station we have?

The six-probe TxSON tile gave a real but weak answer: at 210 m, d_VV ranked the six the
way their SM anomalies ranked them on 60.9% of 184 dates (2.9 SE above chance), against
nothing at all from DTR. This runs the same test everywhere, so the 60.9% is measured on
many tiles and climates rather than one.

TWO TESTS, and the first is the one with real power:

  A  TEMPORAL, every station with S1 and labels (~900).  r(d_VV, that station's OWN SM
     anomaly) over its own dates.  Does the SAR anomaly track the point's wetness at all?
     This needs no co-located stations and is the precondition for everything else.

  B  SPATIAL, compact multi-station groups.  Per date, across the group's members, does
     d_VV rank them the way their SM anomalies do?  Reported as the fraction of dates
     with the correct sign, since §29.10 forbids reading a single small-n r.

d_VV IS THE §33.12 DOUBLE-CENTRED ANOMALY (text/s1processing.md:562): each cell minus its
own temporal norm minus the whole-tile level that day. BOTH SIDES MUST BE ANOMALIES -- d
has zero temporal mean per cell by construction, so correlating it against absolute SM
compares a deviation to a level and returns nothing whatever the sensor does.

210 m (21x21 at 10 m) is the default block: measured on TxSON as a clear optimum over
70 m and 450 m, and a scale optimum is itself evidence the signal is real, since noise
would not peak in the middle.

GROUP COMPACTNESS MATTERS.  Each station has its OWN 224x224 tile centred on itself, so
the "whole-tile level" removed by the double-centring differs between members. For
members whose tiles overlap heavily that reference is nearly common; for distant members
it is not, and the spatial comparison degrades. Groups are therefore filtered on extent
and the extent is reported, not assumed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
import dataset as _ds                                          # noqa: E402
_ds.ZARR_ROOT = Path("/projects/prjs1968/zarr_tokens")
from dataset import _load_zarr_labels, _open_zarr               # noqa: E402
from census_ecostress import ROOT, STATION_CSV, setup_logging   # noqa: E402

ZARR_TOK = Path("/projects/prjs1968/zarr_tokens")
ZARR_SAT = Path("/projects/prjs1968/satellite_zarr")
HALF = 10                       # 21x21 at 10 m = 210 m
MIN_DATES = 30


def folder_of(r):
    src, net = r["source_network"], r["network"]
    return f"{src}_{net}_{r['station_id']}" if (pd.notna(src) and src != net) \
        else f"{net}_{r['station_id']}"


def category_of(r):
    sm, fl = bool(r["has_soil_moisture"]), bool(r["has_flux"])
    return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")


def one_station(job):
    """-> (station_id, DataFrame[date, d_VV, d_CR, sm_anom]) or None."""
    sid, folder, cat = job
    p = ZARR_SAT / f"{folder}.zarr"
    if not p.exists():
        return None
    try:
        import zarr
        g = zarr.open(str(p), mode="r")
        if "s1_asc" not in g or g["s1_asc"]["data"].shape[0] < MIN_DATES:
            return None
        x = np.asarray(g["s1_asc"]["data"][:], dtype=np.float32)
        x[x == 0] = np.nan
        dates = pd.to_datetime([d.decode() if isinstance(d, bytes) else str(d)
                                for d in g["s1_asc"]["dates"][:]]).normalize()
        d = {}
        for bi, nm in enumerate(("VV", "VH")):
            v = x[:, bi]
            d[nm] = (v - np.nanmean(v, 0, keepdims=True)
                     - np.nanmean(v, (1, 2), keepdims=True) + np.nanmean(v))
        sl = slice(112 - HALF, 112 + HALF + 1)
        dvv = np.nanmean(d["VV"][:, sl, sl], (1, 2))
        dvh = np.nanmean(d["VH"][:, sl, sl], (1, 2))

        zg = _open_zarr(ZARR_TOK / cat / folder, cat)
        out = _load_zarr_labels(zg) if zg is not None else None
        if out is None:
            return None
        sm, depths, times, qc = out
        ser = None
        for i, dep in enumerate(depths):
            dep = dep.decode() if isinstance(dep, bytes) else str(dep)
            if dep != "0-10":
                continue
            k = ~np.isnan(sm[i])
            if qc is not None:
                k &= (qc[i] == 0)
            if k.sum() < MIN_DATES:
                return None
            ser = pd.Series(sm[i][k], index=pd.to_datetime(times[k]).normalize())
        if ser is None:
            return None

        j = pd.DataFrame({"date": dates, "d_VV": dvv, "d_CR": dvh - dvv}).dropna()
        j = j.merge(ser.rename("sm").reset_index().rename(columns={"index": "date"}),
                    on="date", how="inner")
        if len(j) < MIN_DATES:
            return None
        j["sm_anom"] = j["sm"] - j["sm"].mean()
        j["station_id"] = sid
        return sid, j
    except Exception:                                          # noqa: BLE001
        return None


def rv(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--min-group", type=int, default=4)
    ap.add_argument("--max-extent-km", type=float, default=1e9)
    ap.add_argument("--link-km", type=float, default=1.12)
    ap.add_argument("--pair-km", type=float, default=2.24)
    args = ap.parse_args()

    setup_logging("s1_spatial_broad")
    log = logging.getLogger("s1")

    st = pd.read_csv(STATION_CSV)
    st = st[st["has_soil_moisture"].astype(bool)]
    jobs = [(r["station_id"], folder_of(r), category_of(r)) for _, r in st.iterrows()]
    log.info("candidate stations with soil moisture: %d", len(jobs))

    with Pool(args.workers) as pool:
        res = [r for r in pool.map(one_station, jobs, chunksize=4) if r is not None]
    data = {sid: df for sid, df in res}
    log.info("stations with S1 + labels + >= %d matched dates: %d", MIN_DATES, len(data))

    # ---------- A. temporal ----------
    rows, trials = [], []
    for sid, j in data.items():
        rows.append({"station_id": sid, "n": len(j),
                     "r_dVV": rv(j.d_VV.to_numpy(), j.sm_anom.to_numpy()),
                     "r_dCR": rv(j.d_CR.to_numpy(), j.sm_anom.to_numpy())})
    T = pd.DataFrame(rows).dropna(subset=["r_dVV"])
    T = T.join(st.set_index("station_id")[["kg_macro", "igbp_macro", "network"]],
               on="station_id")
    T.to_csv(ROOT / "csvs" / "s1_temporal_r.csv", index=False)

    n = len(T)
    se = np.sqrt(.25 / n)
    log.info("")
    log.info("=== A. TEMPORAL  r(d_VV, own SM anomaly), %d stations, median n=%d dates ===",
             n, int(T.n.median()))
    log.info("  d_VV : median %+.3f   mean %+.3f   %.1f%% POSITIVE  "
             "(chance 50%%, SE %.1f%%)",
             T.r_dVV.median(), T.r_dVV.mean(), 100 * (T.r_dVV > 0).mean(), 100 * se)
    log.info("  d_CR : median %+.3f   %.1f%% positive",
             T.r_dCR.median(), 100 * (T.r_dCR > 0).mean())
    for key in ("kg_macro", "igbp_macro"):
        log.info("  by %s:", key)
        for k, g in T.groupby(key):
            if len(g) >= 20:
                log.info("    %-16s n=%4d  median %+.3f  %.0f%% positive",
                         str(k), len(g), g.r_dVV.median(), 100 * (g.r_dVV > 0).mean())

    # ---------- B. spatial ----------
    # GEOGRAPHIC CLUSTERING, not location_group_id.  That column is near-unique per
    # station -- the six TxSON probes inside one window carry ids 744/742/739/735/736/746
    # -- so grouping on it found 7 groups and missed every real cluster. Here stations are
    # linked when they fall within LINK_KM of each other (tiles are 2.24 km wide, so
    # 1.12 km guarantees the two windows overlap by at least half) and connected
    # components of that graph are the groups.
    LINK_KM = args.link_km
    meta = st.set_index("station_id")
    ids = [s for s in meta.index if s in data]
    la = np.deg2rad(meta.loc[ids, "latitude"].to_numpy())
    lo = np.deg2rad(meta.loc[ids, "longitude"].to_numpy())
    dy = (la[:, None] - la[None, :]) * 6371.0
    dx = (lo[:, None] - lo[None, :]) * 6371.0 * np.cos(la.mean())
    D = np.sqrt(dy ** 2 + dx ** 2)
    adj = D <= LINK_KM

    seen, groups = set(), []
    for i in range(len(ids)):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k); comp.append(k)
            stack.extend(np.where(adj[k])[0].tolist())
        if len(comp) >= args.min_group:
            mem = [ids[k] for k in comp]
            ext = float(D[np.ix_(comp, comp)].max())
            groups.append((f"{meta.loc[mem[0],'network']}_{len(mem)}st", mem, ext))
    log.info("")
    log.info("=== B. SPATIAL  geographic clusters (link <= %.2f km), >= %d members: %d ===",
             LINK_KM, args.min_group, len(groups))
    out = []
    for gid, mem, ext in groups:
        if ext > args.max_extent_km:
            continue
        wide_d = pd.DataFrame({s: data[s].set_index("date")["d_VV"] for s in mem})
        wide_a = pd.DataFrame({s: data[s].set_index("date")["sm_anom"] for s in mem})
        rs = []
        for dt in wide_d.index:
            if dt not in wide_a.index:
                continue
            v = rv(wide_d.loc[dt].to_numpy(float), wide_a.loc[dt].to_numpy(float))
            if np.isfinite(v):
                rs.append(v)
        if len(rs) < 20:
            continue
        rs = np.array(rs)
        out.append({"group": gid, "n_stations": len(mem), "extent_km": ext,
                    "n_dates": len(rs), "mean_r": rs.mean(),
                    "frac_pos": float((rs > 0).mean()),
                    "network": meta.loc[mem[0], "network"]})
    S = pd.DataFrame(out)
    if S.empty:
        log.warning("no group cleared the filters -- spatial test not possible")
        return
    S = S.sort_values("frac_pos", ascending=False)
    S.to_csv(ROOT / "csvs" / "s1_spatial_groups.csv", index=False)
    log.info("  usable groups: %d   stations %d   dates %d",
             len(S), int(S.n_stations.sum()), int(S.n_dates.sum()))
    log.info("  POOLED over groups: mean r %+.3f   mean frac positive %.1f%%",
             S.mean_r.mean(), 100 * S.frac_pos.mean())
    log.info("  groups above chance: %d of %d", int((S.frac_pos > .5).sum()), len(S))
    log.info("")
    log.info("  %-22s %6s %8s %7s %8s %8s", "group", "n_st", "ext_km", "dates",
             "mean_r", "frac+")
    for _, r in S.iterrows():
        log.info("  %-22s %6d %8.2f %7d %+8.3f %7.1f%%", str(r.group)[:22],
                 r.n_stations, r.extent_km, r.n_dates, r.mean_r, 100 * r.frac_pos)



    # ---------- C. the PAIR sign test ----------
    # A group of >=4 is not the only way to test spatial skill, and it is the rarest:
    # only TxSON has one. With a PAIR you cannot compute r, but you can ask the question
    # r was standing in for -- on this date, is the station whose SM anomaly is higher
    # also the one whose d_VV is higher? That is a binomial trial, and every co-located
    # pair in the archive contributes one per shared date.
    log.info("")
    log.info("=== C. PAIR SIGN TEST (link <= %.2f km, one trial per pair per date) ===",
             args.pair_km)
    pairs = [(ids[i], ids[j], float(D[i, j]))
             for i in range(len(ids)) for j in range(i + 1, len(ids))
             if D[i, j] <= args.pair_km]
    log.info("  co-located pairs within %.2f km: %d", args.pair_km, len(pairs))
    if not pairs:
        return
    rows, trials = [], []
    for a, b, dist in pairs:
        ja = data[a].set_index("date")[["d_VV", "sm"]]
        jb = data[b].set_index("date")[["d_VV", "sm"]]
        k = ja.index.intersection(jb.index)
        if len(k) < 20:
            continue
        # RECENTRE ON THE PAIR'S SHARED DATES.  Using each station's own full-record mean
        # leaves a constant offset in the difference whenever the two records cover
        # different dates -- and for pairs under ~0.5 km the true SM difference is near
        # zero, so that offset alone decides the sign. Both series are therefore centred
        # on the SAME date set before differencing.
        sa, sb = ja.loc[k, "sm"], jb.loc[k, "sm"]
        va, vb = ja.loc[k, "d_VV"], jb.loc[k, "d_VV"]
        ds = ((sa - sa.mean()) - (sb - sb.mean())).to_numpy()
        dd = ((va - va.mean()) - (vb - vb.mean())).to_numpy()
        m = np.isfinite(dd) & np.isfinite(ds) & (np.abs(ds) > 1e-6)
        if m.sum() < 20:
            continue
        agree = (np.sign(dd[m]) == np.sign(ds[m]))
        rows.append({"a": a, "b": b, "dist_km": dist, "n": int(m.sum()),
                     "frac_agree": float(agree.mean()),
                     "r_diff": rv(dd[m], ds[m])})
        for q, ad, sd_ in zip(np.abs(ds[m]), agree, ds[m]):
            trials.append((dist, q, bool(ad)))
    P = pd.DataFrame(rows)
    if P.empty:
        log.warning("  no pair cleared the minimum shared dates")
        return
    P.to_csv(ROOT / "csvs" / "s1_pair_sign_test.csv", index=False)
    tot = int(P.n.sum())
    w = float((P.frac_agree * P.n).sum() / tot)
    se = np.sqrt(.25 / tot)
    log.info("  usable pairs %d   total trials %d", len(P), tot)
    log.info("  AGREEMENT %.2f%%   (chance 50%%, SE %.2f%%  -> %.1f SE)",
             100 * w, 100 * se, (w - .5) / se)
    log.info("  per-pair: median %.1f%%   %.0f%% of pairs above chance",
             100 * P.frac_agree.median(), 100 * (P.frac_agree > .5).mean())
    log.info("  mean r of the DIFFERENCES: %+.3f", P.r_diff.mean())
    for lo, hi in ((0, .5), (.5, 1.12), (1.12, 2.24)):
        s = P[(P.dist_km >= lo) & (P.dist_km < hi)]
        if len(s) >= 3:
            ww = float((s.frac_agree * s.n).sum() / s.n.sum())
            log.info("    separation %.2f-%.2f km: %3d pairs, %6d trials, "
                     "agreement %.2f%%", lo, hi, len(s), int(s.n.sum()), 100 * ww)


    # IS THE SIGN TEST BEING DECIDED BY NEAR-ZERO MOISTURE DIFFERENCES?
    # mean r of the differences is positive while sign agreement is below chance. That
    # combination means the large |dSM| trials agree and the small ones do not -- i.e.
    # whenever the true difference is near zero, a bias in dd picks the sign. Stratify on
    # |dSM| and the question answers itself: if agreement climbs with |dSM| the signal is
    # real and the sign test is simply uninformative near zero; if it stays flat, S1 does
    # not resolve within-tile moisture differences.
    TR = pd.DataFrame(trials, columns=["dist_km", "abs_dsm", "agree"])
    qs = TR.abs_dsm.quantile([0, .2, .4, .6, .8, .9, 1.0]).to_numpy()
    log.info("")
    log.info("  agreement stratified by |dSM| (equal-count bins):")
    for lo, hi in zip(qs[:-1], qs[1:]):
        s = TR[(TR.abs_dsm >= lo) & (TR.abs_dsm < hi)]
        if len(s) < 200:
            continue
        a = s.agree.mean(); se2 = np.sqrt(.25 / len(s))
        log.info("    |dSM| %.4f-%.4f  n=%6d  agreement %.2f%%  (%+.1f SE)",
                 lo, hi, len(s), 100 * a, (a - .5) / se2)
    far = TR[TR.dist_km >= .5]
    if len(far) > 500:
        log.info("  and for pairs >= 0.5 km apart only:")
        for lo, hi in zip(qs[:-1], qs[1:]):
            s = far[(far.abs_dsm >= lo) & (far.abs_dsm < hi)]
            if len(s) < 200:
                continue
            a = s.agree.mean(); se2 = np.sqrt(.25 / len(s))
            log.info("    |dSM| %.4f-%.4f  n=%6d  agreement %.2f%%  (%+.1f SE)",
                     lo, hi, len(s), 100 * a, (a - .5) / se2)


if __name__ == "__main__":
    main()
