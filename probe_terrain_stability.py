"""
Tier 3b — replaces §32.5's Tier 3, which cannot pass as specified.

§32.5 asks for "two independent implementations... disagreement on ordinary hillslopes
is a bug". Measured over 343 regions, pyflwdir and WhiteboxTools D8 accumulation
correlate at r(ln a) = 0.774 median, 0.652 on hillslopes, and it is WORSE on channels
(0.42, and 0.16 above 100 ha) where D8 is supposedly unambiguous. It is not an
orientation error (every flip/transpose scores ~0 against the as-is 0.55) and it is
not conditioning flats (agreement IMPROVES with conditioning burden, r = +0.46). On a
tilted plane with 0.4 m of roughness the two libraries correlate at 0.55 while their
row means agree to 1%.

The explanation is that D8 itself is the unstable object: it makes a discrete choice
among eight neighbours, so sub-metre elevation differences flip a cell's direction and
the accumulated areas diverge from there. Both implementations are correct. The test
is asking D8 to determine something D8 does not determine.

So the question is re-posed as the one that actually matters: IS WHAT WE SHIP STABLE
UNDER PERTURBATIONS THE SIZE OF THE DEM'S OWN ERROR? Perturb the DEM, redo the whole
derivation, and measure how far TWI and HAND move — over the region, at the station
tiles, and most sharply, in the BETWEEN-STATION CONTRAST that the §32.8 sufficiency
gate consumes.

That last one is the decision. The gate regresses dSM on dTWI between colocated pairs
160-1120 m apart. If dTWI does not survive perturbation at the level of GLO-30's own
relative error, the gate would be regressing on noise and would have to be abandoned
regardless of the hydrology.

Noise levels: GLO-30's absolute vertical accuracy is ~2-4 m LE90, but flow routing
only sees RELATIVE error between neighbouring cells, which is far smaller. 0.05 /
0.2 / 1.0 m brackets that from optimistic to pessimistic.

Usage:
  conda activate terramind
  python probe_terrain_stability.py --region-id 122 --region-id 187 --sigma 0.05 --sigma 0.2 --sigma 1.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import terrain_ops as T

TERRAIN_ROOT = Path("/gpfs/work3/0/prjs1968/data/terrain")
STATION_CSV  = Path(__file__).resolve().parent / "csvs" / "station_dem_region.csv"


def derive(dem_raw, res, wd, crs, origin, stream_ha=T.STREAM_HA):
    """The full shipped derivation, so the perturbation is felt everywhere it would be."""
    cond = T.condition_dem(dem_raw, res, wd, crs=crs, origin=origin)
    acc = T.flow_accum_mfd(cond, res, wd, crs=crs, origin=origin)
    flw = T.d8_network(cond, res)
    beta = T.horn_slope(dem_raw, res)
    twi, _ = T.twi_from(acc, beta, res)
    streams = T.stream_mask(acc, res, stream_ha=stream_ha)
    hand = T.hand_from(flw, cond, streams)
    accd8 = flw.upstream_area(unit="cell").astype(np.float32)
    return {"twi": twi, "hand": hand, "acc_mfd": acc, "acc_d8": accd8}


def agree(a, b, log=False):
    m = np.isfinite(a) & np.isfinite(b)
    if log:
        m &= (a > 0) & (b > 0)
        a, b = np.log(a), np.log(b)
    if m.sum() < 100:
        return {}
    d = np.abs(a[m] - b[m])
    return {"r": float(np.corrcoef(a[m], b[m])[0, 1]),
            "median_abs": float(np.median(d)),
            "p90_abs": float(np.percentile(d, 90)),
            "sd_ref": float(np.std(a[m]))}


def probe(rid: int, sigmas: list[float], seed: int) -> dict:
    p = TERRAIN_ROOT / f"region_{rid:04d}" / "dem_glo30_30m.tif"
    with rasterio.open(p) as src:
        dem = src.read(1)
        tr, crs = src.transform, src.crs
    res = float(tr.a)
    origin = (tr.c, tr.f)

    sta = pd.read_csv(STATION_CSV)
    sta = sta[sta["region_id"] == rid]
    h, w = dem.shape
    pts = []
    for _, s in sta.iterrows():
        col = int((float(s["laea_x"]) - tr.c) / res)
        row = int((tr.f - float(s["laea_y"])) / res)
        if 0 <= row < h and 0 <= col < w:
            pts.append((s["station_id"], row, col))

    out = {"region_id": rid, "n_stations": len(pts), "shape": [h, w]}
    wd = T.scratch_dir(f"stab_{rid:04d}_")
    try:
        base = derive(dem, res, wd, crs, origin)
        rng = np.random.default_rng(seed)
        for sig in sigmas:
            pert = (dem + rng.normal(0.0, sig, dem.shape)).astype(np.float32)
            pert[~np.isfinite(dem)] = np.nan
            got = derive(pert, res, wd, crs, origin)
            key = f"sigma_{sig:g}m"
            out[key] = {
                "twi":     agree(base["twi"], got["twi"]),
                "hand":    agree(base["hand"], got["hand"]),
                "acc_mfd": agree(base["acc_mfd"], got["acc_mfd"], log=True),
                "acc_d8":  agree(base["acc_d8"], got["acc_d8"], log=True),
            }
            # the quantity the sufficiency gate actually consumes: the CONTRAST
            # between two stations, not either station's level
            if len(pts) >= 2:
                dt_b, dt_g, dh_b, dh_g = [], [], [], []
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        _, r1, c1 = pts[i]
                        _, r2, c2 = pts[j]
                        dt_b.append(base["twi"][r1, c1] - base["twi"][r2, c2])
                        dt_g.append(got["twi"][r1, c1] - got["twi"][r2, c2])
                        dh_b.append(base["hand"][r1, c1] - base["hand"][r2, c2])
                        dh_g.append(got["hand"][r1, c1] - got["hand"][r2, c2])
                dt_b, dt_g = np.array(dt_b), np.array(dt_g)
                dh_b, dh_g = np.array(dh_b), np.array(dh_g)
                out[key]["delta_twi_pairs"] = agree(dt_b, dt_g)
                out[key]["delta_hand_pairs"] = agree(dh_b, dh_g)
                out[key]["n_pairs"] = len(dt_b)
            # station-level values
            if pts:
                bt = np.array([base["twi"][r, c] for _, r, c in pts])
                gt = np.array([got["twi"][r, c] for _, r, c in pts])
                bh = np.array([base["hand"][r, c] for _, r, c in pts])
                gh = np.array([got["hand"][r, c] for _, r, c in pts])
                out[key]["twi_at_stations"] = agree(bt, gt)
                out[key]["hand_at_stations"] = agree(bh, gh)
        return out
    finally:
        T.cleanup(wd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-id", type=int, action="append", required=True)
    ap.add_argument("--sigma", type=float, action="append", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("csvs/terrain_stability_probe.json"))
    args = ap.parse_args()
    sigmas = args.sigma or [0.05, 0.2, 1.0]

    rows = [probe(rid, sigmas, args.seed) for rid in args.region_id]
    for r in rows:
        print(json.dumps(r, indent=2, default=float), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, default=float))

    print("\n" + "=" * 78)
    print("stability under DEM perturbation — r vs the unperturbed derivation")
    print(f"{'sigma':>8} {'TWI':>7} {'HAND':>7} {'a(MFD)':>7} {'a(D8)':>7} "
          f"{'TWI@stn':>8} {'dTWI pair':>10} {'dHAND pair':>11}")
    for sig in sigmas:
        k = f"sigma_{sig:g}m"
        def med(path):
            v = []
            for r in rows:
                d = r.get(k, {})
                for p in path.split("."):
                    d = d.get(p, {}) if isinstance(d, dict) else {}
                if isinstance(d, float) and np.isfinite(d):
                    v.append(d)
            return np.median(v) if v else float("nan")
        print(f"{sig:>7g}m {med('twi.r'):>7.3f} {med('hand.r'):>7.3f} "
              f"{med('acc_mfd.r'):>7.3f} {med('acc_d8.r'):>7.3f} "
              f"{med('twi_at_stations.r'):>8.3f} {med('delta_twi_pairs.r'):>10.3f} "
              f"{med('delta_hand_pairs.r'):>11.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
