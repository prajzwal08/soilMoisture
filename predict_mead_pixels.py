# ============================================================
# Mead Ne1 + Ne2 in ONE tile: every-day predictions at both pixels (runbook §69.3)
# ============================================================
# Ne1 (41.1651, -96.4766) and Ne2 (41.1649, -96.4701) are ~545 m apart, so each
# station's 2.24 km tile also contains the other one.  The frozen §66 model emits a
# (3, 112, 112) SM map at 20 m; only the centre pixel (56, 56) was ever supervised
# (§56: the off-centre map is largely a fixed decoder pattern).  This reads, from the
# SAME forward pass, the centre pixel and the other station's pixel:
#
#   tile Ne1:  Ne1 @ centre,  Ne2 @ ~(57, 83)
#   tile Ne2:  Ne2 @ centre,  Ne1 @ ~(55, 29)
#
# so the question "does the map separate an irrigated field from its neighbour, given
# identical weather input" has a direct answer.  ERA5: both stations should sit in the
# same ERA5-Land cell (centre 41.2 N, 96.5 W); checked here from era5/values18.
#
# Inference only, no labels needed (dataset predict_all_days).  flux_only stations are
# admitted explicitly (the checkpoint's category_filter drops them).  Writes ONLY under
# /gpfs/work3/0/prjs1968/data_mead/pred/.
#
# Usage (GPU):
#   python predict_mead_pixels.py [--years 2017-2024] [--save-maps]
# ============================================================

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader

from dataset import SoilMoistureDataset, SM_DEPTHS, ZARR_ROOT
from model import SoilMoistureModel
from train import CudaPrefetcher
from ckpt_utils import load_checkpoint
from eval_predict import ERA5_STATS, SPLITS_CSV, worker_init_fn, _make_key, _git_sha

CKPT     = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/"
                "nolst_L3_wu200_20261005/best.pt")            # §66 frozen baseline
OUT_DIR  = Path("/gpfs/work3/0/prjs1968/data_mead/pred")
STATIONS = {"AmeriFlux_US-Ne1": (41.1651, -96.4766),
            "AmeriFlux_US-Ne2": (41.1649, -96.4701)}
PIX_M    = 20.0                                                  # map pixel size (m)
C        = SoilMoistureModel.STATION_ROW                         # 56


def offset_px(lat0, lon0, lat1, lon1):
    """(row, col) of point 1 in the 112x112 map of a north-up UTM tile centred on point 0."""
    try:
        from pyproj import Transformer
        zone = int((lon0 + 180) // 6) + 1
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:326{zone:02d}", always_xy=True)
        x0, y0 = tr.transform(lon0, lat0)
        x1, y1 = tr.transform(lon1, lat1)
        dx, dy = x1 - x0, y1 - y0
    except ImportError:
        dx = (lon1 - lon0) * 111_320 * np.cos(np.radians(lat0))
        dy = (lat1 - lat0) * 110_574
    return C + int(round(-dy / PIX_M)), C + int(round(dx / PIX_M)), float(np.hypot(dx, dy))


def era5_check():
    """Max |difference| of era5/values18 between the two stations on shared days."""
    v = {}
    for k in STATIONS:
        g = zarr.open_group(str(ZARR_ROOT / "flux_only" / k / "era5"), mode="r")
        v[k] = pd.DataFrame(np.asarray(g["values18"]), index=np.asarray(g["date_ints"]))
    a, b = v["AmeriFlux_US-Ne1"], v["AmeriFlux_US-Ne2"]
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).abs()
    return int(len(common)), float(np.nanmax(d.values)), [float(x) for x in np.nanmax(d.values, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2017-2024")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-maps", action="store_true",
                    help="also save the full (days, 3, 112, 112) fp16 maps per tile")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))
    years = list(range(lo, hi + 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- pixel table
    pix = {}
    for tile, (la0, lo0) in STATIONS.items():
        pix[tile] = {tile: (C, C, 0.0)}
        for other, (la1, lo1) in STATIONS.items():
            if other != tile:
                pix[tile][other] = offset_px(la0, lo0, la1, lo1)
    for t, d in pix.items():
        print(f"tile {t}: " + ", ".join(f"{s[-3:]} @ ({r},{c}) {m:.0f} m" for s, (r, c, m) in d.items()))

    n_common, dmax, dcol = era5_check()
    print(f"ERA5 values18 Ne1 vs Ne2: {n_common} shared days, max |diff| {dmax:.3g} "
          f"({'IDENTICAL -> same ERA5 cell' if dmax == 0 else 'DIFFERENT'})")

    # ---- dataset: the two Mead rows only, flux_only admitted
    sdf = pd.read_csv(SPLITS_CSV)
    sub = sdf[sdf.apply(_make_key, axis=1).isin(STATIONS)]
    assert len(sub) == 2, f"expected 2 Mead rows, got {len(sub)}"
    tmp = Path(tempfile.mkdtemp(dir=OUT_DIR)) / "_mead_rows.csv"
    sub.to_csv(tmp, index=False)

    model, cfg, epoch = load_checkpoint(CKPT, torch.device("cpu"))
    device = torch.device("cuda")
    model = model.to(device).eval()

    ds = SoilMoistureDataset(
        splits_csv=str(tmp), era5_stats_path=str(ERA5_STATS), years=years,
        category_filter=["sm_only", "sm_and_flux", "flux_only"],
        split_filter=sorted(sub["split"].unique()), training=False,
        predict_all_days=True,
    )
    print(f"{len(ds):,} samples, stations {sorted({s['station_key'] for s in ds.samples})}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        worker_init_fn=worker_init_fn,
                        prefetch_factor=2 if args.num_workers else None)

    rows, maps = [], {t: [] for t in STATIONS}
    map_keys = {t: [] for t in STATIONS}
    with torch.no_grad():
        for batch in CudaPrefetcher(loader, device):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sm = model(batch)["sm"].float()                       # (B, 3, 112, 112)
            sm = sm.cpu().numpy()
            yrs = np.asarray(batch["year"].cpu() if torch.is_tensor(batch["year"]) else batch["year"])
            doys = np.asarray(batch["doy"].cpu() if torch.is_tensor(batch["doy"]) else batch["doy"])
            for i, tile in enumerate(batch["station_key"]):
                date = pd.Timestamp(int(yrs[i]), 1, 1) + pd.Timedelta(days=int(doys[i]) - 1)
                for st, (r, c, m) in pix[tile].items():
                    for k, dep in enumerate(SM_DEPTHS):
                        rows.append((tile, st, r, c, date, dep, float(sm[i, k, r, c])))
                if args.save_maps:
                    maps[tile].append(sm[i].astype(np.float16))
                    map_keys[tile].append(date)

    df = pd.DataFrame(rows, columns=["tile", "station", "row", "col", "date", "depth", "pred"])
    out = OUT_DIR / f"mead_pixels_{args.years}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(df):,} rows)")
    if args.save_maps:
        for t in STATIONS:
            np.savez_compressed(OUT_DIR / f"maps_{t}_{args.years}.npz",
                                sm=np.stack(maps[t]), dates=np.array(map_keys[t], "datetime64[D]"))
    (OUT_DIR / f"mead_pixels_{args.years}.json").write_text(json.dumps({
        "checkpoint": str(CKPT), "epoch": int(epoch), "git_sha": _git_sha(),
        "years": years, "pixels": {t: {s: list(v) for s, v in d.items()} for t, d in pix.items()},
        "era5_shared_days": n_common, "era5_max_abs_diff": dmax, "era5_max_abs_diff_per_col": dcol,
        "zarr_root": str(ZARR_ROOT),
        "env": {k: os.environ.get(k) for k in ("ZARR_ROOT", "RAW_ZARR_ROOT", "S48_CACHE_ROOT")},
    }, indent=1))

    # ---- quick numbers: does the map separate the two fields?
    p = df.pivot_table(index=["tile", "date", "depth"], columns="station", values="pred")
    p.columns = [c[-3:] for c in p.columns]
    p["d"] = p["Ne1"] - p["Ne2"]
    print("\nNe1 - Ne2 predicted, per tile and depth: mean, std, corr(Ne1, Ne2)")
    for (tile, dep), g in p.groupby(level=["tile", "depth"]):
        print(f"  tile {tile[-3:]}  {dep:>6}: mean {g['d'].mean():+.4f}  std {g['d'].std():.4f}  "
              f"corr {g['Ne1'].corr(g['Ne2']):.4f}")


if __name__ == "__main__":
    main()
