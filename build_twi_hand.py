"""
Derive TWI and HAND per region from the wide DEM, and crop each station's tile (§32.4).

Runs once per region on the buffered mosaic produced by download_wide_dem.py, then
crops each station's 2.24 km tile out of the regional rasters. The region edge is the
only wall and it is 10 km beyond the buffer, so the non-local terms — `a`, integrated
over tens of km, and HAND's D8 trace, which has to reach a stream lying outside the
tile in 25 of 30 sampled cases — are computed before anything is cropped.

Tier-2 checks (§32.5) run inline per region, because they are cheap and because a
region that fails them must not silently produce station tiles:

  HAND >= 0 everywhere            a negative value is a conditioning bug by definition
  HAND == 0 on stream cells       HAND is defined that way
  a >= res everywhere             every cell contributes at least itself
  zero interior sinks             the conditioning guarantee, measured not assumed
  D8 mass conservation            exact, to the integer
  slope-floor fraction            reported per region AND per station tile; a large
                                  fraction means TWI is degenerate there, which is a
                                  different claim from 'it was clipped'
  catchment inside the region     per station, §32.3's self-contained pre-test: trace
                                  the upslope mask and check it does not touch the
                                  region edge. This front-loads the cheap half of the
                                  MERIT gate; it does NOT replace it (§32.6).

Tier 3 (--tier3) recomputes D8 accumulation with WhiteboxTools and correlates ln(a)
against pyflwdir's on the identical conditioned DEM. The codebases share no lineage.
Disagreement on flats and where MFD/D8 genuinely differ is expected; disagreement on
ordinary hillslopes is a bug.

Outputs
  {TERRAIN_ROOT}/region_{id:04d}/terrain_30m.tif
      float32, 5 bands [twi, hand, acc_cells_mfd, slope_rad, valid], LAEA @ 30 m
  {station_dir}/terrain/terrain_tiles.npz
      terrain_lo (3,28,28) @ 80 m, terrain_hi (3,70,70) @ 32 m,
      channels [TWI, HAND, valid_mask]; plus acc_cells_30m for the MERIT gate,
      with its transform and proj4 so §32.6 can aggregate to MERIT's grid
  csvs/terrain_region_log.csv, csvs/terrain_station_log.csv

Environment: terramind (pyflwdir needs numba).

Usage:
  conda activate terramind
  python build_twi_hand.py --region-id 122 --tier3      # single region, cross-checked
  python build_twi_hand.py --max-cells 20000000 --workers 12
  python build_twi_hand.py --min-cells 20000000 --workers 2   # the big ones
"""

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.pop("PROJ_DATA", None)

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

import terrain_ops as T

REPO         = Path(__file__).resolve().parent
REGION_CSV   = REPO / "csvs" / "dem_regions.csv"
STATION_CSV  = REPO / "csvs" / "station_dem_region.csv"
TERRAIN_ROOT = Path("/gpfs/work3/0/prjs1968/data/terrain")
DATA_ROOT    = Path("/gpfs/work3/0/prjs1968/data")
LOG_DIR      = DATA_ROOT / "logs"

# station tile geometry, from §32.4
TILE_M   = 2240.0
HI_PX, HI_RES = 70, 32.0
LO_PX, LO_RES = 28, 80.0

BANDS = ["twi", "hand", "acc_cells_mfd", "slope_rad", "valid"]


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(LOG_DIR / "build_twi_hand.log")],
    )


def resample_tile(src: np.ndarray, src_transform: Affine, src_crs,
                  cx: float, cy: float, n_px: int, res: float,
                  resampling: Resampling) -> np.ndarray:
    """
    Resample one channel from the 30 m regional grid onto an n_px grid at `res`,
    centred on the station. Same CRS, so this is a pure resampling — but it goes
    through reproject rather than array slicing because 30 m does not divide 32 m
    or 80 m, and nearest-neighbour slicing would shift the tile by up to a cell.
    """
    dst_transform = Affine(res, 0.0, cx - n_px * res / 2.0,
                           0.0, -res, cy + n_px * res / 2.0)
    dst = np.full((n_px, n_px), np.nan, dtype=np.float32)
    reproject(source=src, destination=dst,
              src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
              dst_transform=dst_transform, dst_crs=src_crs, dst_nodata=np.nan,
              resampling=resampling)
    return dst


def process_region(rid: int, stations: pd.DataFrame, region: pd.Series,
                   tier3: bool, overwrite: bool, stream_ha: float,
                   breach_dist: int = T.BREACH_DIST_CELLS) -> dict:
    """Derive terrain for one region and write its station tiles."""
    log = logging.getLogger(__name__)
    t_start = time.time()
    timings: dict[str, float] = {}

    def tick(name: str, t0: float) -> float:
        timings[name] = time.time() - t0
        return time.time()
    out_dir  = TERRAIN_ROOT / f"region_{rid:04d}"
    out_path = out_dir / "terrain_30m.tif"
    dem_path = out_dir / "dem_glo30_30m.tif"

    if not dem_path.exists():
        return {"region_id": rid, "status": "fail:no_dem"}
    if out_path.exists() and not overwrite:
        return {"region_id": rid, "status": "skip"}

    with rasterio.open(dem_path) as src:
        dem_raw = src.read(1)
        transform = src.transform
        crs = src.crs
        res = float(transform.a)
        dem_tags = src.tags()

    valid = np.isfinite(dem_raw)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"region_id": rid, "status": "fail:all_nodata"}

    wd = T.scratch_dir(f"twi_r{rid:04d}_")
    rec: dict = {"region_id": rid, "n_stations": len(stations),
                 "height": dem_raw.shape[0], "width": dem_raw.shape[1],
                 "n_valid": n_valid, "res_m": res}
    try:
        # ── conditioning ─────────────────────────────────────────────────────
        t0 = time.time()
        dem_cond, cstats = T.condition_dem(dem_raw, res, wd, crs=crs,
                                           origin=(transform.c, transform.f),
                                           breach_dist=breach_dist,
                                           return_stats=True)
        rec.update({f"cond_{k}": v for k, v in cstats.items()})
        t0 = tick("condition", t0)

        # ── accumulation: MFD for a, D8 for the HAND trace ───────────────────
        acc_mfd = T.flow_accum_mfd(dem_cond, res, wd, crs=crs,
                                   origin=(transform.c, transform.f))
        t0 = tick("acc_mfd", t0)
        flw = T.d8_network(dem_cond, res)
        t0 = tick("d8_network", t0)

        # ── slope from the RAW DEM, not the conditioned one ──────────────────
        beta = T.horn_slope(dem_raw, res)
        twi, floored = T.twi_from(acc_mfd, beta, res)
        streams = T.stream_mask(acc_mfd, res, stream_ha=stream_ha)
        t0 = tick("slope_twi", t0)
        # HAND on the CONDITIONED surface — see hand_from's docstring for why
        # §32.4's 'elevtn=dem_raw' and 'HAND >= 0' cannot both hold. The raw-surface
        # version is still computed so the size of that discrepancy stays measured
        # per region rather than being quietly designed away.
        hand = T.hand_from(flw, dem_cond, streams)
        hand_raw = T.hand_from(flw, dem_raw, streams)
        rec["hand_raw_min"] = float(np.nanmin(hand_raw)) if np.isfinite(hand_raw).any() else np.nan
        rec["hand_raw_neg_frac"] = float(np.nanmean(hand_raw < -1e-4))
        t0 = tick("hand", t0)

        rec["slope_floor_frac"] = floored
        rec["stream_frac"] = float(streams.mean())
        rec["stream_ha"] = stream_ha

        # ── Tier 2 ───────────────────────────────────────────────────────────
        a = T.sca_from_cells(acc_mfd, res)
        hand_min = float(np.nanmin(hand)) if np.isfinite(hand).any() else np.nan
        got, want = T.mass_conservation_d8(flw, n_valid)
        rec.update({
            "t2_hand_min": hand_min,
            "t2_hand_neg_frac": float(np.nanmean(hand < -1e-4)),
            "t2_hand_max_on_streams": float(np.nanmax(np.abs(hand[streams])))
                                       if streams.any() else 0.0,
            "t2_a_min_m": float(np.nanmin(a)) if np.isfinite(a).any() else np.nan,
            "t2_sinks_final": cstats["sinks_final"],
            "t2_flats_final": cstats["flats_final"],
            "t2_mass_outlet_sum": got,
            "t2_mass_n_valid": want,
            "t2_mass_ok": bool(abs(got - want) < 0.5),
            "twi_mean": float(np.nanmean(twi)), "twi_sd": float(np.nanstd(twi)),
            "hand_mean": float(np.nanmean(hand)), "hand_sd": float(np.nanstd(hand)),
        })

        # ── Tier 3: independent implementation on the identical conditioned DEM ──
        if tier3:
            acc_d8_wbt = T.flow_accum_d8_wbt(dem_cond, res, wd, crs=crs,
                                             origin=(transform.c, transform.f))
            acc_d8_pfd = flw.upstream_area(unit="cell").astype(np.float32)
            m = valid & np.isfinite(acc_d8_wbt) & (acc_d8_pfd > 0) & (acc_d8_wbt > 0)
            if m.sum() > 100:
                lw = np.log(acc_d8_wbt[m]); lp = np.log(acc_d8_pfd[m])
                rec["t3_r_ln_acc_d8"] = float(np.corrcoef(lw, lp)[0, 1])
                rec["t3_median_ln_ratio"] = float(np.median(lw - lp))
                # hillslopes only: agreement on channels is easy, on flats it is not
                hill = m & (acc_mfd < 111) & (np.tan(beta) > 0.02)
                if hill.sum() > 100:
                    rec["t3_r_ln_acc_hillslope"] = float(np.corrcoef(
                        np.log(acc_d8_wbt[hill]), np.log(acc_d8_pfd[hill]))[0, 1])

        # ── regional raster ──────────────────────────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        stack = np.stack([twi, hand, acc_mfd, beta,
                          valid.astype(np.float32)], axis=0).astype(np.float32)
        tmp = out_path.with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", driver="GTiff",
                           height=stack.shape[1], width=stack.shape[2],
                           count=len(BANDS), dtype="float32", crs=crs,
                           transform=transform, nodata=np.nan, compress="deflate",
                           predictor=3, tiled=True, blockxsize=512, blockysize=512,
                           BIGTIFF="IF_SAFER") as dst:
            dst.write(stack)
            for i, b in enumerate(BANDS, start=1):
                dst.set_band_description(i, b)
            dst.update_tags(
                laea_proj4=dem_tags.get("laea_proj4", ""),
                res_m=f"{res:g}", stream_ha=f"{stream_ha:g}",
                mfd_exponent=f"{T.MFD_EXPONENT:g}",
                breach_dist_cells=str(T.BREACH_DIST_CELLS),
                tan_slope_floor=f"{T.TAN_SLOPE_FLOOR:g}",
                slope_floor_frac=f"{floored:.6f}",
                slope_source="raw DEM (conditioning flattens valleys)",
                dem_source=dem_tags.get("source", ""),
                dem_nan_frac=dem_tags.get("nan_frac", ""),
                note="a = acc_cells * res, metres. HAND from a D8 trace, MFD for a.",
            )
        tmp.rename(out_path)

        # ── per-station tiles + the catchment-inside-region pre-test ─────────
        srec = []
        h, w = dem_raw.shape
        for _, s in stations.iterrows():
            cx, cy = float(s["laea_x"]), float(s["laea_y"])
            col = int((cx - transform.c) / res)
            row = int((transform.f - cy) / res)
            if not (0 <= row < h and 0 <= col < w):
                srec.append({"station_id": s["station_id"], "region_id": rid,
                             "status": "fail:outside_region"})
                continue

            # trace the station's upslope mask; if it reaches the region edge the
            # catchment was truncated and `a` is an underestimate
            idx = np.array([row * w + col], dtype=np.int64)
            basin = flw.basins(idxs=idx).reshape(h, w) > 0
            touches_edge = bool(basin[0, :].any() or basin[-1, :].any()
                                or basin[:, 0].any() or basin[:, -1].any())

            # 32 m from 30 m is barely a resample, so bilinear; 80 m from 30 m is a
            # 7x area reduction, so average — nearest would alias the valley network.
            # valid is a fraction, so average at both scales.
            chans_hi, chans_lo = [], []
            for arr, hi_rs in ((twi, Resampling.bilinear), (hand, Resampling.bilinear),
                               (valid.astype(np.float32), Resampling.average)):
                chans_hi.append(resample_tile(arr, transform, crs, cx, cy,
                                              HI_PX, HI_RES, hi_rs))
                chans_lo.append(resample_tile(arr, transform, crs, cx, cy,
                                              LO_PX, LO_RES, Resampling.average))
            terrain_hi = np.stack(chans_hi).astype(np.float32)
            terrain_lo = np.stack(chans_lo).astype(np.float32)

            # native-30 m accumulation over the footprint, for the MERIT gate:
            # §32.6 compares over the 2.24 km footprint, never at the station point
            half = int(np.ceil(TILE_M / 2 / res)) + 1
            r0, r1 = max(row - half, 0), min(row + half + 1, h)
            c0, c1 = max(col - half, 0), min(col + half + 1, w)
            acc_fp = acc_mfd[r0:r1, c0:c1].astype(np.float32)
            fp_transform = rasterio.windows.transform(
                rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), transform)

            tile_beta = beta[r0:r1, c0:c1]
            out_st = Path(s["station_dir"]) / "terrain" if "station_dir" in s \
                else DATA_ROOT / "terrain_tiles" / str(s["station_id"])
            out_st.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_st / "terrain_tiles.npz",
                terrain_lo=terrain_lo, terrain_hi=terrain_hi,
                acc_cells_30m=acc_fp,
                acc_transform=np.array(fp_transform.to_gdal(), dtype=np.float64),
                proj4=np.array(dem_tags.get("laea_proj4", ""), dtype=object),
                channels=np.array(["TWI", "HAND", "valid_mask"], dtype=object),
                station_xy=np.array([cx, cy], dtype=np.float64),
                res_m=np.float64(res), stream_ha=np.float64(stream_ha),
            )

            srec.append({
                "station_id": s["station_id"], "region_id": rid, "status": "done",
                "catchment_touches_region_edge": touches_edge,
                "basin_cells": int(basin.sum()),
                "acc_at_station_cells": float(acc_mfd[row, col]),
                "a_at_station_m": float(acc_mfd[row, col] * res),
                "twi_at_station": float(twi[row, col]),
                "hand_at_station": float(hand[row, col]),
                "slope_at_station": float(beta[row, col]),
                "tile_slope_floor_frac": float(np.nanmean(
                    np.tan(tile_beta) < T.TAN_SLOPE_FLOOR)),
                "tile_twi_sd": float(np.nanstd(terrain_hi[0])),
                "tile_hand_sd": float(np.nanstd(terrain_hi[1])),
                "tile_valid_frac": float(np.nanmean(terrain_hi[2])),
            })

        tick("stations", t0)
        rec["status"] = "done"
        rec["n_edge_truncated"] = sum(
            1 for r in srec if r.get("catchment_touches_region_edge"))
        rec["seconds"] = time.time() - t_start
        rec.update({f"t_{k}": round(v, 2) for k, v in timings.items()})
        log.info(f"  region {rid:04d}: done in {rec['seconds']:.0f}s  "
                 f"{len(stations)} station(s), slope_floor {floored:.2%}, "
                 f"streams {rec['stream_frac']:.2%}, HAND min {hand_min:.3f}, mass "
                 f"{'OK' if rec['t2_mass_ok'] else 'FAIL'}, "
                 f"edge-truncated {rec['n_edge_truncated']}/{len(stations)}  "
                 + " ".join(f"{k}={v:.0f}s" for k, v in timings.items()))
        return {**rec, "_stations": srec}
    except Exception as exc:
        logging.getLogger(__name__).error(
            f"  region {rid:04d}: {type(exc).__name__}: {str(exc)[:300]}")
        return {"region_id": rid, "status": f"error:{type(exc).__name__}:{str(exc)[:200]}"}
    finally:
        T.cleanup(wd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive TWI/HAND per region (§32.4)")
    ap.add_argument("--region-csv", type=Path, default=REGION_CSV)
    ap.add_argument("--station-csv", type=Path, default=STATION_CSV)
    ap.add_argument("--terrain-root", type=Path, default=None)
    ap.add_argument("--region-id", type=int, action="append", default=None)
    ap.add_argument("--min-cells", type=float, default=None,
                    help="Only regions with at least this many cells (big ones).")
    ap.add_argument("--max-cells", type=float, default=None,
                    help="Only regions below this many cells (the bulk).")
    ap.add_argument("--max-stations", type=int, default=None,
                    help="Pilot mode: smallest set of regions covering N stations.")
    ap.add_argument("--stream-ha", type=float, default=T.STREAM_HA)
    ap.add_argument("--breach-dist", type=int, default=T.BREACH_DIST_CELLS,
                    help="Max least-cost breach search distance in cells. Cost is "
                         "~O(dist^2) per pit; canopy dams are 1-3 cells wide.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tier3", action="store_true",
                    help="Also run the WhiteboxTools-vs-pyflwdir D8 cross-check.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--region-log", type=Path,
                    default=REPO / "csvs" / "terrain_region_log.csv")
    ap.add_argument("--station-log", type=Path,
                    default=REPO / "csvs" / "terrain_station_log.csv")
    args = ap.parse_args()

    if args.terrain_root is not None:
        global TERRAIN_ROOT
        TERRAIN_ROOT = args.terrain_root

    setup_logging()
    log = logging.getLogger(__name__)

    reg = pd.read_csv(args.region_csv)
    sta = pd.read_csv(args.station_csv)

    # station_dir, same folder convention as the rest of the pipeline
    splits = pd.read_csv(REPO / "csvs" / "station_splits.csv")

    def _folder(r):
        if r["source_network"] != r["network"]:
            return f"{r['source_network']}_{r['network']}_{r['station_id']}"
        return f"{r['network']}_{r['station_id']}"

    splits["sid"] = splits.apply(_folder, axis=1)

    def _dir(r):
        has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        has_fl = str(r.get("has_flux", "False")).lower() == "true"
        cat = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")
        return str(DATA_ROOT / cat / r["sid"])

    splits["station_dir"] = splits.apply(_dir, axis=1)
    sta = sta.merge(splits[["sid", "station_dir"]], left_on="station_id",
                    right_on="sid", how="left").drop(columns=["sid"])

    if args.region_id:
        reg = reg[reg["region_id"].isin(args.region_id)]
    if args.min_cells:
        reg = reg[reg["n_cells"] >= args.min_cells]
    if args.max_cells:
        reg = reg[reg["n_cells"] < args.max_cells]
    if args.max_stations:
        reg = reg.sort_values("n_stations")
        reg = reg[reg["n_stations"].cumsum() <= args.max_stations]

    reg = reg.sort_values("n_cells", ascending=False).reset_index(drop=True)
    log.info(f"{len(reg)} regions, {int(reg['n_stations'].sum())} stations, "
             f"{reg['n_cells'].sum()/1e9:.3f}e9 cells, workers={args.workers}, "
             f"tier3={args.tier3}, stream_ha={args.stream_ha:g}")

    region_recs, station_recs = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for _, row in reg.iterrows():
            rid = int(row["region_id"])
            futs[pool.submit(process_region, rid,
                             sta[sta["region_id"] == rid].copy(), row,
                             args.tier3, args.overwrite, args.stream_ha,
                             args.breach_dist)] = rid

        for i, fut in enumerate(as_completed(futs), 1):
            rid = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                log.error(f"  region {rid:04d} worker died: {type(exc).__name__}: {exc}")
                region_recs.append({"region_id": rid, "status": "worker_died"})
                continue
            station_recs.extend(rec.pop("_stations", []))
            region_recs.append(rec)
            if i % 20 == 0:
                ok = sum(1 for r in region_recs if r.get("status") == "done")
                log.info(f"[{i}/{len(reg)}] done={ok}")

    rdf = pd.DataFrame(region_recs)
    sdf = pd.DataFrame(station_recs)
    for path, df in ((args.region_log, rdf), (args.station_log, sdf)):
        if len(df) == 0:
            continue
        key = "region_id" if "region_id" in df and path is args.region_log else "station_id"
        if path.exists() and not args.overwrite:
            old = pd.read_csv(path)
            if key in old:
                df = pd.concat([old[~old[key].isin(df[key])], df], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    # ── Tier-2 summary: a failure here must be loud, not a row in a csv ──────
    done = rdf[rdf["status"] == "done"] if "status" in rdf else rdf
    log.info(f"regions done {len(done)}/{len(reg)}")
    if len(done):
        bad_mass = done[~done["t2_mass_ok"].astype(bool)]
        bad_hand = done[done["t2_hand_neg_frac"] > 0]
        bad_sink = done[done["t2_sinks_final"] > 0]
        bad_a    = done[done["t2_a_min_m"] < done["res_m"] - 1e-3]
        log.info(f"TIER 2  mass-conservation failures  {len(bad_mass)}")
        log.info(f"TIER 2  regions with HAND < 0       {len(bad_hand)}")
        log.info(f"TIER 2  regions with residual sinks {len(bad_sink)}")
        log.info(f"TIER 2  regions with a < res        {len(bad_a)}")
        log.info(f"        slope-floor fraction: median {done['slope_floor_frac'].median():.4%}"
                 f"  max {done['slope_floor_frac'].max():.4%}")
        for name, sub in (("mass", bad_mass), ("hand<0", bad_hand),
                          ("sinks", bad_sink), ("a<res", bad_a)):
            if len(sub):
                log.error(f"  {name}: regions {sorted(sub['region_id'].tolist())[:20]}")
        if "t3_r_ln_acc_d8" in done:
            t3 = done["t3_r_ln_acc_d8"].dropna()
            if len(t3):
                log.info(f"TIER 3  r(ln a) pyflwdir vs WhiteboxTools D8: "
                         f"median {t3.median():.4f}  min {t3.min():.4f}  n={len(t3)}")
                h3 = done.get("t3_r_ln_acc_hillslope", pd.Series(dtype=float)).dropna()
                if len(h3):
                    log.info(f"        hillslopes only: median {h3.median():.4f}  "
                             f"min {h3.min():.4f}")
    if len(sdf):
        trunc = sdf[sdf.get("catchment_touches_region_edge", False) == True]
        log.info(f"stations {len(sdf)}, catchment touching the region edge: "
                 f"{len(trunc)} ({len(trunc)/max(len(sdf),1):.1%}) — these need MERIT "
                 f"fallback per §32.6, the pre-test does not replace the gate")
    log.info(f"logs: {args.region_log}  {args.station_log}")


if __name__ == "__main__":
    main()
