#!/usr/bin/env python
"""§37.10 -- look at the DTR field before testing it.

Five columns per date, sharing one colour scale DOWN each column so the rows are
comparable to each other -- which is the whole point, because the question G0 asks is
whether the DTR pattern is the SAME every date:

    S2 RGB   |  Day LST  |  Night LST  |  DTR = day - night  |  DTR anomaly

The anomaly column (DTR minus that date's own scene mean) is the one to read.  §29
measured Landsat daytime LST at +0.967 spatial coherence against its own annual mean --
the tile was heterogeneous, 5-9 K, but the pattern never moved, so it tracked topography
and land cover rather than soil moisture.  DTR is supposed to escape that by cancelling
the static emissivity and albedo structure and leaving thermal inertia.  If the anomaly
panels are visually identical across dates, DTR has the same disease and G0 fails.

REGISTRATION, stated rather than assumed.  The S2 patch is 224x224 at 10 m from
bounds_utm in the station's UTM zone; the LST patch is 32x32 at 70 m on the ECOSTRESS
MGRS grid.  Both are 2240 m centred on the station, but they are cut from different
grids, so they are co-located to within about half an LST pixel -- not resampled onto a
common grid here.  Good enough to see what is where; not good enough to difference.

The S2 scene is the one nearest in time with the LOWEST mean blue reflectance among the
candidates, because cloud is bright in B02.  The actual day offset is printed on every
panel -- a 12-day-old RGB is a context image, not a measurement of that day.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DATA_ROOT   = Path("/gpfs/work3/0/prjs1968/data")
ZARR_ROOT   = Path("/projects/prjs1968/satellite_zarr")
OUT_DIR     = Path("/gpfs/work3/0/prjs1968/soilMoisture/fig/dtr_txson")
S2_RGB_IDX  = (3, 2, 1)        # B04, B03, B02 in the stored band order
S2_BLUE_IDX = 1                # B02


def load_bundle(folder: str):
    for cat in ("sm_only", "sm_and_flux", "flux_only"):
        hits = sorted((DATA_ROOT / cat / folder / "ECOSTRESS").glob(f"{folder}_dtr_*.npz"))
        if hits:
            return np.load(hits[-1], allow_pickle=False), hits[-1]
    raise SystemExit(f"no DTR bundle for {folder} -- run consolidate_dtr.py first")


def s2_rgb(folder: str, want: str, max_days: int = 20):
    """-> (rgb float[224,224,3] in [0,1], s2_date str, offset_days int) or None.

    Nearest in time, then least-blue among the ties -- cloud is bright in B02.
    """
    import zarr
    p = ZARR_ROOT / f"{folder}.zarr"
    if not p.exists():
        return None
    g = zarr.open(str(p), mode="r")
    if "s2" not in g:
        return None
    dates = np.array([d.decode() if isinstance(d, bytes) else str(d)
                      for d in g["s2"]["dates"][:]])
    wd = pd.Timestamp(str(want))
    off = np.array([(pd.Timestamp(str(d)) - wd).days for d in dates])
    cand = np.where(np.abs(off) <= max_days)[0]
    if cand.size == 0:
        return None
    # least-blue wins, so a clear scene 9 days away beats a cloudy one 1 day away
    blues = [float(np.mean(g["s2"]["data"][int(i), S2_BLUE_IDX])) for i in cand]
    i = int(cand[int(np.argmin(blues))])

    cube = np.asarray(g["s2"]["data"][i], dtype=np.float32)      # [12,224,224]
    rgb = np.stack([cube[b] for b in S2_RGB_IDX], -1) / 10000.0
    rgb[~np.isfinite(rgb)] = 0.0
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((rgb - lo) / (hi - lo), 0, 1), dates[i], int(off[i])


def pick_dates(z, n: int):
    """The n best-covered aligned pairs, spread over distinct months where possible."""
    ok = (z["grid_aligned"] == 1) & (z["n_valid_px"] > 0)
    idx = np.where(ok)[0]
    if idx.size == 0:
        raise SystemExit("no aligned pair with a valid pixel in this bundle")
    order = idx[np.argsort(-z["n_valid_px"][idx])]
    days = [str(z["day_utc"][i].decode() if isinstance(z["day_utc"][i], bytes)
                else z["day_utc"][i])[:10] for i in order]
    chosen, seen = [], set()
    for i, d in zip(order, days):
        ym = d[:7]
        if ym in seen:
            continue
        seen.add(ym)
        chosen.append(i)
        if len(chosen) == n:
            break
    for i in order:                    # top up if there were not enough months
        if len(chosen) == n:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="", help="e.g. ISMN_TxSON_CR1000-1; blank = best TxSON")
    ap.add_argument("--network", default="TxSON")
    ap.add_argument("--n-dates", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    folder = args.folder
    if not folder:
        bundles = pd.read_csv("/gpfs/work3/0/prjs1968/soilMoisture/csvs/"
                              f"ecostress_dtr_bundles.{args.network}.csv")
        bundles = bundles[bundles.folder.isin([p.stem for p in ZARR_ROOT.glob("*.zarr")])]
        if bundles.empty:
            raise SystemExit(f"no {args.network} bundle also has an S2 zarr")
        folder = bundles.sort_values("n_pairs_usable", ascending=False).iloc[0]["folder"]
        print(f"auto-picked {folder} "
              f"({int(bundles.n_pairs_usable.max())} usable pairs)")

    z, path = load_bundle(folder)
    print(f"bundle {path}")
    rows = pick_dates(z, args.n_dates)

    day = z["day_lst_k"][rows]
    nig = z["night_lst_k"][rows]
    dtr = z["dtr_k"][rows]
    val = z["valid"][rows].astype(bool)
    day = np.where(val, day, np.nan)
    nig = np.where(val, nig, np.nan)

    # anomaly: each date minus ITS OWN scene mean, so only the spatial pattern is left
    ano = np.stack([d - np.nanmean(d) for d in dtr])

    def lim(a, p=2):
        f = a[np.isfinite(a)]
        return (np.percentile(f, p), np.percentile(f, 100 - p)) if f.size else (0, 1)

    dl, nl, tl = lim(day), lim(nig), lim(dtr)
    am = np.nanmax(np.abs([np.nanpercentile(ano, 2), np.nanpercentile(ano, 98)]))

    n = len(rows)
    fig = plt.figure(figsize=(14.5, 2.9 * n + 1.5))
    gs = GridSpec(n + 1, 5, figure=fig, height_ratios=[1] * n + [0.09],
                  hspace=0.16, wspace=0.07)

    cols = [
        ("Sentinel-2 RGB",       None, None,   None),
        ("Day LST (K)",          "magma",  dl, "seq"),
        ("Night LST (K)",        "magma",  nl, "seq"),
        ("DTR = day - night (K)", "viridis", tl, "seq"),
        ("DTR anomaly (K)",      "RdBu_r", (-am, am), "div"),
    ]
    ims = [None] * 5

    for r, i in enumerate(rows):
        du = str(z["day_utc"][i].decode() if isinstance(z["day_utc"][i], bytes)
                 else z["day_utc"][i])[:10]
        nu = str(z["night_utc"][i].decode() if isinstance(z["night_utc"][i], bytes)
                 else z["night_utc"][i])[:10]

        ax = fig.add_subplot(gs[r, 0])
        got = s2_rgb(folder, du)
        if got is None:
            ax.text(.5, .5, "no S2 within 20 d", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#666")
            ax.set_facecolor("#f0f0f0")
        else:
            rgb, sd, off = got
            ax.imshow(rgb)
            ax.text(.03, .04, f"S2 {sd}  ({off:+d} d)", transform=ax.transAxes,
                    fontsize=7.5, color="w", va="bottom",
                    bbox=dict(fc="black", alpha=.55, pad=1.6, lw=0))
        ax.set_ylabel(f"{du}\nnight {nu}\ndt {z['dt_hours'][i]:.1f} h",
                      fontsize=8.5, rotation=0, ha="right", va="center", labelpad=8)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(cols[0][0], fontsize=10, pad=6)

        for c, (title, cmap, (v0, v1), _k) in enumerate(cols[1:], start=1):
            arr = [None, day, nig, dtr, ano][c][r]
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor("#e8e8e8")          # NaN shows as grey, not as a colour
            ims[c] = ax.imshow(arr, cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10, pad=6)
            if c in (1, 2, 3):
                f = arr[np.isfinite(arr)]
                if f.size:
                    ax.text(.03, .04, f"mean {f.mean():.1f}", transform=ax.transAxes,
                            fontsize=7.5, color="w", va="bottom",
                            bbox=dict(fc="black", alpha=.55, pad=1.6, lw=0))
            ax.text(.97, .04, f"{int(np.isfinite(arr).sum())}/1024 px",
                    transform=ax.transAxes, fontsize=7, color="w", ha="right",
                    va="bottom", bbox=dict(fc="black", alpha=.45, pad=1.4, lw=0))

    for c in (1, 2, 3, 4):
        cax = fig.add_subplot(gs[n, c])
        fig.colorbar(ims[c], cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=7.5, length=2)

    lat, lon = float(z["latitude"]), float(z["longitude"])
    fig.suptitle(f"{folder}   ({lat:.4f}, {lon:.4f})   "
                 f"2.24 km x 2.24 km   ECOSTRESS L2T LSTE v002 at 70 m\n"
                 f"colour scales are SHARED DOWN EACH COLUMN -- rows are comparable; "
                 f"grey = no valid LST (cloud, QC, or off-swath)",
                 fontsize=10.5, y=0.995)

    out = Path(args.out) if args.out else OUT_DIR / f"dtr_panels_{folder}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    # The number the eye cannot give you: is the anomaly pattern the SAME every date?
    flat = ano.reshape(len(rows), -1)
    m = np.isfinite(flat).all(0)
    if m.sum() > 30:
        C = np.corrcoef(flat[:, m])
        iu = np.triu_indices(len(rows), 1)
        print(f"\nDTR-anomaly spatial correlation between dates "
              f"({int(m.sum())} common pixels):")
        print(f"  mean r = {C[iu].mean():+.3f}   min {C[iu].min():+.3f}   "
              f"max {C[iu].max():+.3f}")
        print("  (for scale: 29 measured daytime Landsat LST at +0.967 -- "
              "a static field)")
    else:
        print(f"\nonly {int(m.sum())} pixels valid in ALL {len(rows)} dates -- "
              "too few for the cross-date correlation")


if __name__ == "__main__":
    main()
