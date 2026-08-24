"""
Settle §32.9.4: WHY do pyflwdir and WhiteboxTools D8 accumulation disagree, and does
it threaten anything we ship?

Tier 3 over 343 regions gave r(ln a) median 0.774 and 0.652 on hillslopes, with a
median log ratio of exactly 0 — the marginal distributions match, individual cells
disagree. That is the signature of tie-breaking, not of a wrong answer, but 'looks
like tie-breaking' is not a result.

The question that actually matters is narrower than the Tier-3 number. TWI's `a` comes
from WhiteboxTools MFD and no D8 field touches it. pyflwdir's D8 feeds exactly three
things: HAND's trace, the basin/region-edge pre-test, and mass conservation. So this
probe asks:

  1. Does the disagreement concentrate on hillslopes and vanish on channels?
     D8 stripes planar hillslopes — that is precisely why §32.4 chose MFD for `a` —
     and two implementations breaking ties differently will stripe them differently.
     On channels the steepest-descent path is unambiguous and both must agree.
  2. Do the two agree on WHERE THE STREAMS ARE? HAND only needs the trace to reach
     the right channel, not to take the same hillslope route to it.
  3. Does HAND itself differ? The end-to-end question, asked by rebuilding the flow
     network from WhiteboxTools' own D8 pointer and recomputing HAND through it.

If (1) says channels agree, (2) says the stream networks coincide, and (3) says HAND
is unchanged, then the Tier-3 number is measuring hillslope tie-breaking in a field
we do not ship, and §32.5's 'disagreement on ordinary hillslopes is a bug' is wrong
for D8 specifically.

Usage:
  conda activate terramind
  python probe_d8_disagreement.py --region-id 122 --region-id 187 --region-id 206
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

import terrain_ops as T

TERRAIN_ROOT = Path("/gpfs/work3/0/prjs1968/data/terrain")
CHANNEL_CELLS = 111          # 10 ha at 30 m — the stream threshold §32.4 uses


def d8_pointer_wbt(dem_cond, res, wd, crs, origin):
    """WhiteboxTools' own D8 pointer, so pyflwdir can trace ITS directions."""
    f_in = wd / "dem_for_ptr.tif"
    f_out = wd / "d8_pointer.tif"
    T.write_wbt_tif(f_in, dem_cond, res, crs=crs, origin=origin)
    T.run_wbt("D8Pointer", f_out, dem=str(f_in), output=str(f_out))
    return T.read_wbt_tif(f_out)


def corr(a, b, m):
    if m.sum() < 100:
        return float("nan")
    return float(np.corrcoef(np.log(a[m]), np.log(b[m]))[0, 1])


def probe(rid: int) -> dict:
    p = TERRAIN_ROOT / f"region_{rid:04d}" / "dem_glo30_30m.tif"
    with rasterio.open(p) as src:
        dem_raw = src.read(1)
        tr, crs = src.transform, src.crs
    res = float(tr.a)
    wd = T.scratch_dir(f"d8probe_{rid:04d}_")
    out = {"region_id": rid, "shape": list(dem_raw.shape)}
    try:
        dem_cond, cst = T.condition_dem(dem_raw, res, wd, crs=crs,
                                        origin=(tr.c, tr.f), return_stats=True)
        out["cond_touched_frac"] = cst["touched_frac"]

        acc_wbt = T.flow_accum_d8_wbt(dem_cond, res, wd, crs=crs, origin=(tr.c, tr.f))
        flw_pfd = T.d8_network(dem_cond, res)
        acc_pfd = flw_pfd.upstream_area(unit="cell").astype(np.float32)

        base = (np.isfinite(acc_wbt) & (acc_wbt > 0) & (acc_pfd > 0)
                & np.isfinite(dem_raw))
        beta = T.horn_slope(dem_raw, res)

        # (1) where does the disagreement live?
        hill = base & (acc_wbt < CHANNEL_CELLS) & (np.tan(beta) > 0.02)
        chan = base & (acc_wbt >= CHANNEL_CELLS)
        big  = base & (acc_wbt >= 10 * CHANNEL_CELLS)
        out["r_all"]      = corr(acc_wbt, acc_pfd, base)
        out["r_hillslope"] = corr(acc_wbt, acc_pfd, hill)
        out["r_channel"]  = corr(acc_wbt, acc_pfd, chan)
        out["r_channel10x"] = corr(acc_wbt, acc_pfd, big)
        out["frac_channel"] = float(chan.sum() / max(base.sum(), 1))

        # (2) do the two agree on WHERE the streams are?
        s_wbt = T.stream_mask(acc_wbt, res)
        s_pfd = T.stream_mask(acc_pfd, res)
        inter = float((s_wbt & s_pfd).sum())
        union = float((s_wbt | s_pfd).sum())
        out["stream_iou"] = inter / union if union else float("nan")
        out["stream_frac_wbt"] = float(s_wbt.mean())
        out["stream_frac_pfd"] = float(s_pfd.mean())

        # (3) the end-to-end question: does HAND change?
        hand_pfd = T.hand_from(flw_pfd, dem_cond, s_pfd)
        try:
            import pyflwdir
            from affine import Affine
            ptr = d8_pointer_wbt(dem_cond, res, wd, crs, (tr.c, tr.f))
            ptr_i = np.nan_to_num(ptr, nan=0.0).astype(np.uint8)
            flw_wbt = pyflwdir.from_array(
                ptr_i, ftype="d8",
                transform=Affine(res, 0.0, 0.0, 0.0, -res, 0.0), latlon=False)
            hand_wbt = T.hand_from(flw_wbt, dem_cond, s_wbt)
            m = np.isfinite(hand_pfd) & np.isfinite(hand_wbt)
            out["hand_r"] = float(np.corrcoef(hand_pfd[m], hand_wbt[m])[0, 1])
            out["hand_median_abs_diff_m"] = float(np.median(np.abs(hand_pfd[m] - hand_wbt[m])))
            out["hand_p90_abs_diff_m"] = float(np.percentile(np.abs(hand_pfd[m] - hand_wbt[m]), 90))
            out["hand_sd_pfd"] = float(np.nanstd(hand_pfd))
            out["hand_sd_wbt"] = float(np.nanstd(hand_wbt))
        except Exception as exc:
            out["hand_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

        # control: does MFD (what we actually ship for `a`) even depend on this?
        acc_mfd = T.flow_accum_mfd(dem_cond, res, wd, crs=crs, origin=(tr.c, tr.f))
        mm = base & np.isfinite(acc_mfd) & (acc_mfd > 0)
        out["r_mfd_vs_d8wbt"] = corr(acc_mfd, acc_wbt, mm)
        out["r_mfd_vs_d8pfd"] = corr(acc_mfd, acc_pfd, mm)
        return out
    finally:
        T.cleanup(wd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-id", type=int, action="append", required=True)
    ap.add_argument("--out", type=Path, default=Path("csvs/d8_disagreement_probe.json"))
    args = ap.parse_args()

    rows = []
    for rid in args.region_id:
        r = probe(rid)
        rows.append(r)
        print(json.dumps(r, indent=2, default=float), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, default=float))

    print("\n" + "=" * 72)
    def med(k):
        v = [r[k] for r in rows if isinstance(r.get(k), float) and np.isfinite(r[k])]
        return np.median(v) if v else float("nan")
    print(f"r(ln a) D8 pyflwdir vs WhiteboxTools   all {med('r_all'):.3f}   "
          f"hillslope {med('r_hillslope'):.3f}   channel {med('r_channel'):.3f}   "
          f"channel>=100ha {med('r_channel10x'):.3f}")
    print(f"stream-network IoU at 10 ha            {med('stream_iou'):.3f}")
    print(f"HAND r {med('hand_r'):.4f}   median |diff| {med('hand_median_abs_diff_m'):.3f} m   "
          f"p90 {med('hand_p90_abs_diff_m'):.3f} m")
    print(f"MFD (shipped `a`) vs D8: wbt {med('r_mfd_vs_d8wbt'):.3f}  "
          f"pyflwdir {med('r_mfd_vs_d8pfd'):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
