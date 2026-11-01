#!/usr/bin/env python
"""Step 0 of the 993-station Landsat ST pull -- QC yield curves, measured, no download.

Reads the GeoTIFFs already on disk from the cancelled job 27015709
(/gpfs/work3/0/prjs1968/data/landsat_st_stations/, 43 stations, ~14.7k scenes) and reports what
each candidate QC threshold actually costs.  The pull pools 30 m -> 100 m in flight, and pooling
is IRREVERSIBLE with respect to the pixel mask -- you cannot recover "mean of pixels with
ST_QA <= 2" from "mean of pixels with ST_QA <= 3".  So the thresholds have to be chosen before
the pull, and this is the last cheap chance to choose them from data instead of from the DFCB.

Three things this file is careful about.

1.  THE SPEC IS NOT EVIDENCE.  The USGS DFCB says only cloud confidence (bits 8-9) populates
    level 2, and that shadow/snow/cirrus use 0/1/3 with 2 reserved.  If that is true, then
    "<= 1" and "<= 2" differ for exactly one of the four confidence fields, and the strictness
    dial is far narrower than it looks.  Measured here against the actual DN histogram.

2.  ST_QA'S SPREAD IS THE WHOLE QUESTION.  Its dominant term is the atmospheric profile, which
    comes from reanalysis on a ~32 km grid and is therefore near-constant across a 2.24 km tile.
    If within-tile spread is small against between-scene spread, ST_QA is a SCENE GATE wearing a
    pixel-level costume, and inverse-variance weighting is the right use of it rather than a hard
    per-pixel cut.  That ratio is reported explicitly.

3.  YIELD IS NOT UNIFORM ACROSS CLIMATE.  ST_QA tracks atmospheric water vapour, so a fixed
    threshold keeps cold dry stations and drops humid ones.  Every yield number is broken out by
    kg_macro, because a threshold that silently re-weights the station set toward one climate is
    a bias, not a filter.

The tifs read here are the OLD 3-band form (lst_kelvin, st_qa_kelvin, qa_pixel_dn).  cdist and
emis are not on disk yet, so their curves come from the Step-1 smoke, not from here.

OUTPUT  csvs/landsat_qc_yield.csv          one row per scene, every statistic, re-aggregatable
        csvs/landsat_qc_yield_summary.txt  the tables, and the two thresholds they set
"""
from __future__ import annotations

import logging
import os
os.environ.pop("PROJ_DATA", None)
import argparse
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling as WarpResampling
from rasterio.warp import reproject

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO      = Path(__file__).resolve().parent
TIF_ROOT  = Path("/gpfs/work3/0/prjs1968/data/landsat_st_stations")
SPLITS    = REPO / "csvs" / "station_splits.csv"
OUT_CSV   = REPO / "csvs" / "landsat_qc_yield.csv"
OUT_TXT   = REPO / "csvs" / "landsat_qc_yield_summary.txt"

TARGET_M  = 2200.0   # 22 x 100 m, the exact partition of §33.12(d)
TARGET_N  = 22
CELL_M    = 100.0

ST_QA_LEVELS = (2.0, 3.0, 5.0)
LST_LO, LST_HI = 250.0, 350.0


# ── QA_PIXEL ────────────────────────────────────────────────────────────────

def _bits(q: np.ndarray, b: int) -> np.ndarray:
    return ((q >> b) & 1).astype(bool)


def _conf(q: np.ndarray, b: int) -> np.ndarray:
    return (q >> b) & 3


def qa_masks(qa_dn: np.ndarray) -> dict:
    """Every candidate QA_PIXEL mask, from the same raw DN, so they are exactly comparable."""
    q = np.nan_to_num(qa_dn, nan=1.0).astype(np.uint16)   # DN 1 == fill bit set

    fill   = _bits(q, 0)
    dilat  = _bits(q, 1)
    cirrus = _bits(q, 2)
    cloud  = _bits(q, 3)
    shadow = _bits(q, 4)
    snow   = _bits(q, 5)
    clear6 = _bits(q, 6)
    water  = _bits(q, 7)

    c_cloud  = _conf(q, 8)
    c_shadow = _conf(q, 10)
    c_snow   = _conf(q, 12)
    c_cirrus = _conf(q, 14)

    # the six rejections everyone agrees on
    base = ~(fill | dilat | cirrus | cloud | shadow | snow)

    conf_strict = (c_cloud <= 1) & (c_shadow <= 1) & (c_cirrus <= 1)
    conf_loose  = (c_cloud <= 2) & (c_shadow <= 1) & (c_cirrus <= 1)

    return {
        # current landsat_clear(): base + water rejected + bit6 required + strict confidence
        "legacy":        base & ~water & clear6 & conf_strict,
        # bit 6 dropped (redundant with bits 1 and 3), water still rejected
        "nobit6":        base & ~water & conf_strict,
        # THE OPERATING POINT: bit 6 dropped, water kept and flagged separately
        "water_kept":    base & conf_strict,
        # the one real strictness dial: allow medium-confidence cloud
        "conf_loose":    base & conf_loose,
        "water":         water,
        "clear6":        clear6,
        "_q":            q,
        "_c":            (c_cloud, c_shadow, c_snow, c_cirrus),
    }


def bit_stats(m: dict) -> dict:
    """Per-bit and per-confidence-level frequencies, to check the DFCB against the data."""
    q = m["_q"]
    n = q.size
    out = {f"bit{b}_frac": round(float(_bits(q, b).mean()), 6) for b in range(8)}
    names = ("ccloud", "cshadow", "csnow", "ccirrus")
    for nm, c in zip(names, m["_c"]):
        for lvl in range(4):
            out[f"{nm}_L{lvl}"] = int((c == lvl).sum())
    out["n_px"] = n
    return out


# ── pooling to the 22x22 @ 100 m grid ───────────────────────────────────────

def pool_frac(mask: np.ndarray, src_transform, src_crs) -> np.ndarray:
    """Area-weighted valid fraction per 100 m cell, on the exact 2200 m / 22 px partition.

    Resampling.average on the boolean mask IS the contributing fraction -- that is what
    §33.12(d) means by masking a cell below a valid-contributing-fraction threshold.
    """
    h, w = mask.shape
    cx = src_transform.c + src_transform.a * w / 2.0
    cy = src_transform.f + src_transform.e * h / 2.0
    half = TARGET_M / 2.0
    dst_transform = from_origin(cx - half, cy + half, CELL_M, CELL_M)
    dst = np.zeros((TARGET_N, TARGET_N), dtype="float32")
    reproject(
        mask.astype("float32"), dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=src_crs,
        resampling=WarpResampling.average,
    )
    return dst


# ── one scene ───────────────────────────────────────────────────────────────

def scene_row(args) -> dict | None:
    station, path = args
    rec = {"station_id": station, "file": Path(path).name,
           "date": Path(path).name[:8], "status": "ok"}
    try:
        with rasterio.open(path) as src:
            a = src.read()
            transform, crs = src.transform, src.crs
    except Exception as exc:
        rec.update(status="unreadable", error=str(exc)[:160])
        return rec

    if a.shape[0] < 3:
        rec.update(status="bad_bands")
        return rec

    lst, st_qa, qa_dn = a[0], a[1], a[2]
    rec["ny"], rec["nx"] = int(lst.shape[0]), int(lst.shape[1])

    m = qa_masks(qa_dn)
    rec.update(bit_stats(m))

    lst_valid = np.isfinite(lst)
    qa_valid  = np.isfinite(st_qa)
    n = lst.size

    # --- the two no-retrieval masks must agree; §29.5 assumed they do, never checked
    rec["lst_valid_frac"]  = round(float(lst_valid.mean()), 6)
    rec["stqa_valid_frac"] = round(float(qa_valid.mean()), 6)
    rec["nodata_agree"]    = round(float((lst_valid == qa_valid).mean()), 6)
    rec["stqa_allfill"]    = int(not qa_valid.any())

    # --- clear fractions, each variant on the same denominator
    for k in ("legacy", "nobit6", "water_kept", "conf_loose"):
        rec[f"clear_{k}"] = round(float((m[k] & lst_valid).sum() / n), 6)
    rec["water_frac"]  = round(float(m["water"].mean()), 6)
    rec["clear6_frac"] = round(float(m["clear6"].mean()), 6)

    # --- ST_QA, over the operating-point clear pixels only
    clear = m["water_kept"] & lst_valid
    rec["n_clear"] = int(clear.sum())
    if clear.any():
        v = st_qa[clear & qa_valid]
        if v.size:
            rec["stqa_median"] = round(float(np.median(v)), 4)
            rec["stqa_p95"]    = round(float(np.percentile(v, 95)), 4)
            # WITHIN-tile spread.  Compared against the spread of these medians
            # BETWEEN scenes in the summary -- that ratio is the whole ST_QA question.
            rec["stqa_std_within"] = round(float(v.std()), 4)
            rec["stqa_lt1_frac"]   = round(float((v < 1.0).mean()), 6)
            for lv in ST_QA_LEVELS:
                rec[f"stqa_le{lv:g}_frac"] = round(float((v <= lv).mean()), 6)
        lv_ = lst[clear]
        rec["lst_median"] = round(float(np.median(lv_)), 3)
        rec["lst_std_within"] = round(float(lv_.std()), 4)
        rec["lst_in_range_frac"] = round(float(((lv_ > LST_LO) & (lv_ < LST_HI)).mean()), 6)

    # --- 100 m cell valid fraction, at the operating point (adds ST_QA <= 3)
    op = clear & qa_valid & (st_qa <= 3.0) & (lst > LST_LO) & (lst < LST_HI)
    rec["op_frac"] = round(float(op.mean()), 6)
    if op.any():
        cf = pool_frac(op, transform, crs)
        for p in (10, 25, 50, 75, 90):
            rec[f"cellfrac_p{p}"] = round(float(np.percentile(cf, p)), 4)
        for t in (0.25, 0.50, 0.75):
            rec[f"cells_ge{int(t*100)}"] = int((cf >= t).sum())
    return rec


# ── main ────────────────────────────────────────────────────────────────────

def setup_logging():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif-root", type=Path, default=TIF_ROOT)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap the number of scenes")
    args = ap.parse_args()
    setup_logging()

    tasks = []
    for d in sorted(args.tif_root.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.tif")):
            tasks.append((d.name, str(f)))
    if args.limit:
        tasks = tasks[: args.limit]
    logging.info("%d scenes across %d stations",
                 len(tasks), len({t[0] for t in tasks}))
    if not tasks:
        raise SystemExit(f"no tifs under {args.tif_root}")

    with Pool(args.workers) as pool:
        rows = [r for r in pool.imap_unordered(scene_row, tasks, chunksize=16) if r]

    df = pd.DataFrame(rows)

    # pandas, never awk: 6 AmeriFlux rows carry quoted commas in station_name
    sp = pd.read_csv(SPLITS)
    df = df.merge(sp[["station_id", "kg_macro", "igbp_macro", "latitude", "longitude"]],
                  on="station_id", how="left")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    logging.info("wrote %s  (%d rows)", OUT_CSV, len(df))

    write_summary(df)


def write_summary(df: pd.DataFrame):
    ok = df[df.status == "ok"].copy()
    L = []
    def p(s=""):
        L.append(s)
        print(s)

    p("=" * 78)
    p("LANDSAT ST -- QC YIELD CURVES  (Step 0, measured, no download)")
    p("=" * 78)
    p(f"scenes read      : {len(df)}")
    p(f"  ok             : {len(ok)}")
    for s, n in df.status.value_counts().items():
        if s != "ok":
            p(f"  {s:<14} : {n}")
    p(f"stations         : {ok.station_id.nunique()}")
    p(f"raster shapes    : {sorted(set(zip(ok.ny, ok.nx)))}")
    p()

    # ---- 1. does the DFCB hold?
    p("-" * 78)
    p("1. CONFIDENCE LEVELS -- does level 2 occur outside cloud confidence?")
    p("-" * 78)
    p("   DFCB says: cloud (8-9) uses 0/1/2/3; shadow, snow, cirrus use 0/1/3 (2 reserved).")
    p("   If true, '<=1' vs '<=2' differs for CLOUD ONLY.")
    p()
    p(f"   {'field':<10} {'L0':>14} {'L1':>14} {'L2':>14} {'L3':>14}")
    for nm in ("ccloud", "cshadow", "csnow", "ccirrus"):
        tot = [int(ok[f"{nm}_L{l}"].sum()) for l in range(4)]
        p(f"   {nm:<10} " + " ".join(f"{v:>14,}" for v in tot))
    l2 = {nm: int(ok[f"{nm}_L2"].sum()) for nm in ("cshadow", "csnow", "ccirrus")}
    if sum(l2.values()) == 0:
        p()
        p("   -> CONFIRMED. Level 2 is unused outside cloud confidence.")
        p("      The strictness dial is cloud confidence alone.")
    else:
        p()
        p(f"   -> DFCB CONTRADICTED: level 2 occurs in {l2}. Revisit the mask.")
    p()

    # ---- 2. clear-fraction yield by mask variant
    p("-" * 78)
    p("2. CLEAR-FRACTION YIELD BY MASK VARIANT  (mean over scenes)")
    p("-" * 78)
    names = {
        "clear_legacy":     "legacy  (bit6 required, water rejected, conf<=low)",
        "clear_nobit6":     "bit6 dropped, water rejected",
        "clear_water_kept": "bit6 dropped, water KEPT  <-- operating point",
        "clear_conf_loose": "bit6 dropped, water kept, cloud conf<=medium",
    }
    base = ok.clear_legacy.mean()
    for k, lab in names.items():
        v = ok[k].mean()
        p(f"   {lab:<52} {v:6.4f}  ({v/base*100 if base else 0:6.1f}% of legacy)")
    p()
    p(f"   mean water fraction per tile : {ok.water_frac.mean():.4f}")
    p(f"   bit6 'Clear' vs derived      : bit6={ok.clear6_frac.mean():.4f}")
    p()

    # ---- 3. ST_QA: scene gate or pixel mask?
    p("-" * 78)
    p("3. ST_QA -- IS IT A PIXEL MASK OR A SCENE GATE?")
    p("-" * 78)
    s = ok.dropna(subset=["stqa_median"])
    if len(s):
        within  = float(s.stqa_std_within.mean())
        between = float(s.stqa_median.std())
        p(f"   median ST_QA over all scenes      : {s.stqa_median.median():.3f} K"
          f"   (§29.13 measured 2.13 K at TxSON)")
        p(f"   mean WITHIN-tile std              : {within:.3f} K")
        p(f"   BETWEEN-scene std of tile medians : {between:.3f} K")
        r = within / between if between else float("nan")
        p(f"   ratio within/between              : {r:.3f}")
        p()
        if r < 0.5:
            p("   -> ST_QA is a SCENE-LEVEL quantity. A per-pixel threshold mostly keeps or")
            p("      drops whole scenes. Inverse-variance weighting is the right use of it;")
            p("      the 3 K cut should be read as a scene gate, not a pixel mask.")
        else:
            p("   -> ST_QA varies substantially WITHIN the tile. A genuine per-pixel mask is")
            p("      justified; revisit the 1/sigma^2-weighting-instead-of-cut recommendation.")
        p()
        p("   yield among clear pixels:")
        for lv in ST_QA_LEVELS:
            c = f"stqa_le{lv:g}_frac"
            p(f"     ST_QA <= {lv:g} K : {s[c].mean():6.4f}")
        p()
        p(f"   ST_QA < 1 K (propagation short-circuit?) : {s.stqa_lt1_frac.mean():.5f}")
        p(f"   all-fill ST_QA scenes                    : {int(ok.stqa_allfill.sum())}")
        p(f"   ST_QA/LST no-retrieval agreement         : {ok.nodata_agree.mean():.5f}")
        if ok.nodata_agree.mean() < 0.999:
            p("     -> the two no-retrieval masks DISAGREE. Carry both, do not assume one.")
    p()

    # ---- 4. climatic bias
    p("-" * 78)
    p("4. YIELD BY KOPPEN MACRO-CLASS -- is the threshold climatically biased?")
    p("-" * 78)
    if "kg_macro" in ok and ok.kg_macro.notna().any():
        cols = ["clear_water_kept"] + [f"stqa_le{lv:g}_frac" for lv in ST_QA_LEVELS]
        g = ok.groupby("kg_macro").agg(
            n_scenes=("file", "count"), n_stations=("station_id", "nunique"),
            **{c: (c, "mean") for c in cols})
        p(g.round(4).to_string())
        sp_ = ok.groupby("kg_macro")["stqa_le3_frac"].mean().dropna()
        if len(sp_) > 1:
            p()
            p(f"   ST_QA<=3 retention spans {sp_.min():.3f} ({sp_.idxmin()}) to "
              f"{sp_.max():.3f} ({sp_.idxmax()})")
            if sp_.max() - sp_.min() > 0.25:
                p("   -> MATERIALLY BIASED. A fixed 3 K cut re-weights the station set by climate.")
            else:
                p("   -> spread is modest; a fixed 3 K cut does not strongly re-weight by climate.")
    p()

    # ---- 5. per-station clear ceiling (§41.6)
    p("-" * 78)
    p("5. PER-STATION CLEAR CEILING  (§41.6: threshold = ceiling - 0.03)")
    p("-" * 78)
    ce = ok.groupby("station_id").clear_water_kept.max().sort_values()
    p(f"   ceiling  min {ce.min():.3f}   p25 {ce.quantile(.25):.3f}   "
      f"median {ce.median():.3f}   p75 {ce.quantile(.75):.3f}   max {ce.max():.3f}")
    p(f"   stations with ceiling < 0.50 (swath-limited, not cloud-limited): "
      f"{int((ce < 0.5).sum())} of {len(ce)}")
    p("   lowest 8:")
    for k, v in ce.head(8).items():
        p(f"     {k:<28} {v:.3f}")
    p()

    # ---- 6. the 100 m cell mask threshold
    p("-" * 78)
    p("6. 100 m CELL VALID-CONTRIBUTING-FRACTION  (§33.12(d) cell mask)")
    p("-" * 78)
    cc = ok.dropna(subset=["cellfrac_p50"])
    if len(cc):
        p("   distribution across the 22x22 cells, at the operating point:")
        for q in (10, 25, 50, 75, 90):
            p(f"     p{q:<3} of cells : {cc[f'cellfrac_p{q}'].mean():.4f}")
        p()
        p("   cells retained per scene (of 484) at each cell-mask threshold:")
        for t in (25, 50, 75):
            v = cc[f"cells_ge{t}"].mean()
            p(f"     >= {t/100:.2f} valid : {v:7.1f}  ({v/484*100:5.1f}%)")
        p()
        p("   -> pick the knee: the threshold that drops partial-coverage edge cells without")
        p("      eating interior ones. 0.50 is the default unless this says otherwise.")
    p()

    p("=" * 78)
    p("THRESHOLDS THIS SETS FOR THE PULL")
    p("=" * 78)
    p("   ST_QA cut         : 3 K  (+ 1/sigma^2 weighting) -- confirm against section 3")
    p("   cell valid-frac   : see section 6")
    p("   CDIST px          : NOT MEASURABLE HERE -- cdist is not on these tifs.")
    p("                       Comes from the Step-1 smoke. Provisional 300 m.")
    p("=" * 78)

    OUT_TXT.write_text("\n".join(L) + "\n")
    logging.info("wrote %s", OUT_TXT)


if __name__ == "__main__":
    main()
