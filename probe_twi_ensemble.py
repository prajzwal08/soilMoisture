"""
Can a spatially-resolved 30 m TWI be made reproducible? (§32.9.4 follow-up)

Tier 3b showed the between-station contrast dTWI retains only r = 0.195 under 0.2 m
of DEM perturbation, while dHAND retains 0.987. The cause is that `a` is a chaotic
upslope INTEGRAL: routing decisions on hillslopes are near-ties, a flipped decision
propagates and compounds downstream, and `a` is genuinely near-discontinuous in space.

Aggregating TWI over the station footprint would fix the number and destroy the point
of the exercise — a spatially uniform terrain value is the same failure mode as the
FiLM context vector this whole architecture exists to replace. So the resolution stays
at 30 m per cell and the averaging is moved to where the randomness actually lives:
over DEM-ERROR REALISATIONS, not over space.

Perturb, re-derive, repeat, take the per-cell expectation. Every cell keeps its own
value. Cells robustly inside a drainage line keep their high `a`; cells whose high
value came from a coin flip regress toward what their neighbourhood actually supports.
The per-cell spread across realisations is a free uncertainty map, which is the honest
thing to put in the valid_mask channel.

THE TEST: build two INDEPENDENT ensembles (disjoint seeds) and ask whether their mean
dTWI agrees at the station pairs. A single realisation against another single
realisation scores r = 0.195 — that is the number to beat. If two independent
N-member ensembles agree, the ensemble mean is a reproducible field and can carry the
sufficiency gate at full 30 m resolution. If they do not, no amount of averaging over
noise will rescue dTWI and the gate must run on dHAND instead.

Usage:
  conda activate terramind
  python probe_twi_ensemble.py --region-id 10 --region-id 31 --n-members 8 --sigma 0.15
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import terrain_ops as T

TERRAIN_ROOT = Path("/gpfs/work3/0/prjs1968/data/terrain")
STATION_CSV  = Path(__file__).resolve().parent / "csvs" / "station_dem_region.csv"


def derive_twi(dem, res, wd, crs, origin):
    """TWI and HAND only — the two fields the gate would use."""
    cond = T.condition_dem(dem, res, wd, crs=crs, origin=origin)
    acc = T.flow_accum_mfd(cond, res, wd, crs=crs, origin=origin)
    beta = T.horn_slope(dem, res)
    twi, _ = T.twi_from(acc, beta, res)
    flw = T.d8_network(cond, res)
    hand = T.hand_from(flw, cond, T.stream_mask(acc, res))
    return twi, hand


def ensemble(dem, res, wd, crs, origin, n, sigma, seed):
    """Per-cell mean and sd of TWI/HAND over n DEM-error realisations."""
    rng = np.random.default_rng(seed)
    ts, hs = [], []
    for k in range(n):
        pert = (dem + rng.normal(0.0, sigma, dem.shape)).astype(np.float32)
        pert[~np.isfinite(dem)] = np.nan
        t, h = derive_twi(pert, res, wd, crs, origin)
        ts.append(t)
        hs.append(h)
        print(f"      member {k+1}/{n}", flush=True)
    ts, hs = np.stack(ts), np.stack(hs)
    return (np.nanmean(ts, 0), np.nanstd(ts, 0),
            np.nanmean(hs, 0), np.nanstd(hs, 0), ts, hs)


def pair_deltas(field, pts):
    d = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            _, r1, c1 = pts[i]
            _, r2, c2 = pts[j]
            d.append(field[r1, c1] - field[r2, c2])
    return np.array(d)


def r_of(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def probe(rid, n, sigma, seed, source="glo30"):
    p = TERRAIN_ROOT / f"region_{rid:04d}" / f"dem_{source}_30m.tif"
    with rasterio.open(p) as src:
        dem = src.read(1)
        tr, crs = src.transform, src.crs
    res = float(tr.a)
    origin = (tr.c, tr.f)
    h, w = dem.shape

    sta = pd.read_csv(STATION_CSV)
    sta = sta[sta["region_id"] == rid]
    pts = []
    for _, s in sta.iterrows():
        col = int((float(s["laea_x"]) - tr.c) / res)
        row = int((tr.f - float(s["laea_y"])) / res)
        if 0 <= row < h and 0 <= col < w:
            pts.append((s["station_id"], row, col))

    out = {"region_id": rid, "n_stations": len(pts), "n_members": n,
           "sigma": sigma, "source": source}
    wd = T.scratch_dir(f"ens_{rid:04d}_")
    try:
        t0 = time.time()
        print(f"  region {rid}: ensemble A ({n} members, sigma {sigma} m)", flush=True)
        tA, tA_sd, hA, hA_sd, tsA, _ = ensemble(dem, res, wd, crs, origin, n, sigma, seed)
        print(f"  region {rid}: ensemble B", flush=True)
        tB, tB_sd, hB, hB_sd, tsB, _ = ensemble(dem, res, wd, crs, origin, n, sigma,
                                                seed + 10_000)
        out["seconds"] = time.time() - t0

        # single realisation vs single realisation — the number to beat
        dt1a, dt1b = pair_deltas(tsA[0], pts), pair_deltas(tsB[0], pts)
        # ensemble mean vs ensemble mean
        dtA, dtB = pair_deltas(tA, pts), pair_deltas(tB, pts)
        dhA, dhB = pair_deltas(hA, pts), pair_deltas(hB, pts)

        out["r_dtwi_single"] = r_of(dt1a, dt1b)
        out["r_dtwi_ensemble"] = r_of(dtA, dtB)
        out["r_dhand_ensemble"] = r_of(dhA, dhB)
        out["n_pairs"] = int(len(dtA))
        out["sd_dtwi"] = float(np.nanstd(dtA))
        out["med_err_dtwi_single"] = float(np.nanmedian(np.abs(dt1a - dt1b)))
        out["med_err_dtwi_ensemble"] = float(np.nanmedian(np.abs(dtA - dtB)))

        # does the ensemble mean still have spatial structure, or did it smooth away?
        m = np.isfinite(tA) & np.isfinite(tB)
        out["r_twi_field_ensemble"] = r_of(tA[m], tB[m])
        out["r_twi_field_single"] = r_of(tsA[0][m], tsB[0][m])
        out["twi_field_sd_ensemble"] = float(np.nanstd(tA))
        out["twi_field_sd_single"] = float(np.nanstd(tsA[0]))
        # per-cell ensemble spread: the free uncertainty map
        out["twi_member_sd_median"] = float(np.nanmedian(tA_sd))
        out["twi_member_sd_p90"] = float(np.nanpercentile(tA_sd, 90))
        out["hand_member_sd_median"] = float(np.nanmedian(hA_sd))
        # at the stations specifically
        out["twi_member_sd_at_stations"] = float(np.nanmedian(
            [tA_sd[r, c] for _, r, c in pts])) if pts else float("nan")
        out["_dtwi_ens"] = [dtA.tolist(), dtB.tolist()]
        out["_dtwi_single"] = [dt1a.tolist(), dt1b.tolist()]
        out["_dhand_ens"] = [dhA.tolist(), dhB.tolist()]
        return out
    finally:
        T.cleanup(wd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-id", type=int, action="append", required=True)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15,
                    help="DEM-error sd in m used to build the ensemble.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--source", choices=["glo30", "fabdem"], default="glo30")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = Path(f"csvs/twi_ensemble_probe_{args.source}.json")
    print(f"DEM source: {args.source}", flush=True)

    rows = [probe(rid, args.n_members, args.sigma, args.seed, args.source)
            for rid in args.region_id]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, default=float))

    print("\n" + "=" * 86)
    print(f"ensemble of {args.n_members} members, sigma {args.sigma} m — two INDEPENDENT "
          f"ensembles compared")
    print(f"{'reg':>5} {'pairs':>6} {'r(dTWI) single':>15} {'r(dTWI) ens':>12} "
          f"{'r(dHAND) ens':>13} {'field r single':>15} {'field r ens':>12} "
          f"{'per-cell sd':>12}")
    for r in rows:
        print(f"{r['region_id']:>5} {r['n_pairs']:>6} {r['r_dtwi_single']:>15.3f} "
              f"{r['r_dtwi_ensemble']:>12.3f} {r['r_dhand_ensemble']:>13.3f} "
              f"{r['r_twi_field_single']:>15.3f} {r['r_twi_field_ensemble']:>12.3f} "
              f"{r['twi_member_sd_median']:>12.3f}")

    # pooled — the gate's n is across all stations
    for key, label in (("_dtwi_single", "dTWI single realisation"),
                       ("_dtwi_ens", "dTWI ENSEMBLE MEAN"),
                       ("_dhand_ens", "dHAND ensemble mean")):
        a, b = [], []
        for r in rows:
            a += r[key][0]
            b += r[key][1]
        a, b = np.array(a), np.array(b)
        m = np.isfinite(a) & np.isfinite(b)
        print(f"POOLED  {label:<26} n={int(m.sum()):>4}  r={r_of(a,b):>6.3f}  "
              f"med|err|={np.nanmedian(np.abs(a[m]-b[m])):>6.3f}  sd={np.nanstd(a[m]):>6.3f}")

    print("\nField sd tells you whether ensembling smoothed the map away: if "
          "'field sd ens' is\nclose to 'field sd single' the spatial structure "
          "survived, it is only the coin-flip\ncomponent that was removed.")
    for r in rows:
        print(f"  region {r['region_id']}: TWI field sd single {r['twi_field_sd_single']:.3f} "
              f"-> ensemble {r['twi_field_sd_ensemble']:.3f}   "
              f"per-cell sd at stations {r['twi_member_sd_at_stations']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
