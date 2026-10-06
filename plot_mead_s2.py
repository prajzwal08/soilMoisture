# ============================================================
# Mead: Sentinel-2 scenes over Ne1 and Ne2 (runbook §69.5)
# ============================================================
# Both fields sit in Ne1's 2.24 km tile (Ne2 546 m east of the centre), so one tile
# shows the irrigated-vs-irrigated pair side by side.  Imagery = the raw store the
# model reads (dataset.py RAW_ROOT), 10 m, DN = reflectance x 10000.
# Figures per window:
#   mead_s2_rgb_<tag>   true colour (B04/B03/B02), one panel per scene
#   mead_s2_ndvi_<tag>  NDVI, same scenes
#   mead_s2_ndvi_ts_<tag>  NDVI in a 50 m box at each tower + logged Ne2 irrigation
# Panel label: date and clear fraction of the 640 m box around the centre (cloud mask).
#
# Usage:
#   python plot_mead_s2.py --window 2019-07-15 2019-08-01
#   python plot_mead_s2.py --window 2019-01-01 2019-12-31
# ============================================================

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

import plot_style_bw
from plot_mead_pred import ROOT, FIG_DIR, acquisitions

RAW     = Path("/gpfs/scratch1/shared/pkhanal/satellite_zarr/AmeriFlux_US-Ne1.zarr")
TOWERS  = {"US-Ne1": (41.1651, -96.4766), "US-Ne2": (41.1649, -96.4701)}
MARK    = {"US-Ne1": ("o", plot_style_bw.BLUE), "US-Ne2": ("^", plot_style_bw.GREEN)}
B       = {"B02": 1, "B03": 2, "B04": 3, "B08": 7}
BOX     = 2                                   # +-2 px = 50 m box for the NDVI series


def tower_px(attrs):
    """(row, col) of each tower in the 224 x 224 tile from its UTM bounds."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{attrs['epsg']}", always_xy=True)
    x0, _, _, y1 = attrs["bounds_utm"]
    px = attrs["pixel_size_m"]
    out = {}
    for s, (lat, lon) in TOWERS.items():
        x, y = tr.transform(lon, lat)
        out[s] = ((y1 - y) / px - 0.5, (x - x0) / px - 0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", nargs=2, default=["2019-07-15", "2019-08-01"])
    ap.add_argument("--max-cloud", type=float, default=1.0,
                    help="skip scenes whose clear fraction is below 1 - this (default: keep all)")
    args = ap.parse_args()
    lo, hi = (pd.Timestamp(w) for w in args.window)
    tag = f"{lo:%Y%m%d}_{hi:%Y%m%d}"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(RAW), mode="r")
    g = root["s2"]
    dates = pd.to_datetime([d.decode() if isinstance(d, bytes) else str(d) for d in g["dates"][:]],
                           format="%Y%m%d")
    idx = np.where((dates >= lo) & (dates <= hi))[0]
    _, clear = acquisitions("AmeriFlux_US-Ne1")
    clear = clear[~clear.index.duplicated()]
    keep = [i for i in idx if not (clear.get(dates[i], np.nan) < 1 - args.max_cloud)]
    print(f"{tag}: {len(idx)} S2 scenes, {len(keep)} plotted")
    if not keep:
        return
    px = tower_px(dict(root.attrs))
    print("tower pixels (row, col):", {k: tuple(round(v, 1) for v in rc) for k, rc in px.items()})

    rgb, ndvi, rows = [], [], []
    for i in keep:
        a = np.asarray(g["data"][i]).astype(np.float32)
        valid = (a[B["B04"]] > 0)
        r = np.clip(np.stack([a[B["B04"]], a[B["B03"]], a[B["B02"]]], -1) / 3000.0, 0, 1) ** (1 / 1.4)
        r[~valid] = 1.0
        nir, red = a[B["B08"]], a[B["B04"]]
        nd = np.where(valid, (nir - red) / np.maximum(nir + red, 1), np.nan)
        rgb.append(r)
        ndvi.append(nd)
        row = dict(date=dates[i].date(), clear=clear.get(dates[i], np.nan),
                   valid_frac=float(valid.mean()))
        for s, (rr, cc) in px.items():
            ri, ci = int(round(rr)), int(round(cc))
            row[f"ndvi_{s[-3:]}"] = float(np.nanmean(nd[ri - BOX:ri + BOX + 1, ci - BOX:ci + BOX + 1]))
        rows.append(row)
    tab = pd.DataFrame(rows)
    out = ROOT / "pred" / f"s2_ndvi_towers_{tag}.csv"
    tab.to_csv(out, index=False)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"table: {out}")

    plot_style_bw.apply(globals(), "paper")
    n = len(keep)
    nc = min(n, 6 if n <= 12 else 8)
    nr = int(np.ceil(n / nc))
    for kind, imgs in (("rgb", rgb), ("ndvi", ndvi)):
        fig, axes = plt.subplots(nr, nc, figsize=(2.0 * nc, 2.15 * nr + 0.6), squeeze=False)
        for k, ax in enumerate(axes.flat):
            ax.set_axis_off()
            if k >= n:
                continue
            if kind == "rgb":
                ax.imshow(imgs[k], interpolation="nearest")
            else:
                im = ax.imshow(imgs[k], cmap="RdYlGn", vmin=0, vmax=0.9, interpolation="nearest")
            for s, (rr, cc) in px.items():
                mk, c = MARK[s]
                ax.plot(cc, rr, mk, ms=6, mfc=c, mec="white", mew=1.0)
            cl = tab["clear"].iloc[k]
            ax.text(0.03, 0.97, f"{tab['date'].iloc[k]:%d %b %Y}"
                    + ("" if np.isnan(cl) else f"  {cl:.0%} clear"),
                    transform=ax.transAxes, va="top", fontsize="small",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
        h = [plt.Line2D([], [], marker=MARK[s][0], ls="", ms=7, mfc=MARK[s][1], mec="black",
                        label=f"{s[-3:]} tower") for s in TOWERS]
        fig.legend(handles=h, loc="lower center", ncol=2, frameon=False)
        rect = [0, 0.6 / (2.15 * nr + 0.6), 1, 1]
        if kind == "ndvi":
            fig.tight_layout(rect=[0, rect[1], 0.92, 1])
            cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
            fig.colorbar(im, cax=cax, label="NDVI")
        else:
            fig.tight_layout(rect=rect)
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"mead_s2_{kind}_{tag}.{ext}", dpi=300)
        plt.close(fig)
        print(f"figure: {FIG_DIR}/mead_s2_{kind}_{tag}.png")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    for s in TOWERS:
        mk, c = MARK[s]
        t = tab.set_index(pd.to_datetime(tab["date"]))
        ok = t["clear"].fillna(0) >= 0.99
        ax.plot(t.index, t[f"ndvi_{s[-3:]}"], "-", color=c, lw=1.0, zorder=2)
        ax.plot(t.index[ok], t.loc[ok, f"ndvi_{s[-3:]}"], mk, ms=5, color=c, mec="black", mew=0.4,
                label=f"{s[-3:]} (clear scene)", zorder=3)
        ax.plot(t.index[~ok], t.loc[~ok, f"ndvi_{s[-3:]}"], mk, ms=5, mfc="white", mec=c,
                label=f"{s[-3:]} (cloud / no mask)", zorder=3)
    f = ROOT / "bif" / "US-Ne2_irrigation_events.csv"
    e = pd.read_csv(f, parse_dates=["date_start", "date_end"])
    e = e[(e["date_start"] <= hi) & (e["date_end"] >= lo) & ~e["cancelled"].fillna(False).astype(bool)]
    for r in e.itertuples():
        ax.axvspan(r.date_start, r.date_end + pd.Timedelta("1D"), color="0.88", lw=0, zorder=0)
    ax.plot([], [], "s", ms=8, color="0.88", label="Ne2 irrigation logged")
    ax.set_xlim(lo, hi + pd.Timedelta("1D"))
    ax.set_ylabel(f"NDVI ({2 * BOX + 1}x{2 * BOX + 1} px box)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_s2_ndvi_ts_{tag}.{ext}", dpi=300)
    print(f"figure: {FIG_DIR}/mead_s2_ndvi_ts_{tag}.png")


if __name__ == "__main__":
    main()
