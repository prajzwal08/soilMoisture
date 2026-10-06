#!/usr/bin/env python
"""Publication figure for ONE tile holding several stations: RGB + per-station time series.

(a) the whole 2.24 km tile in Sentinel-2 true colour, with a box around the stations
(b) a zoom on that box, every station marked (stations closer than --merge-m share a marker)
(c...) one panel per station x depth: observed (black dots, QC=0 days only) vs the model's
    prediction (Python-blue line, every day, eval_predict.py --keep-unobserved), with the
    station metrics above each panel. One shared y range across all panels.

The spatial (predicted map) part is deliberately NOT here yet (user 2026-10-06).

Predictions are each station's OWN-tile readout (its centre pixel), from the frozen baseline's
all-day parquets -- no GPU. Metrics are computed on observed days only and printed next to the
eval_output per_station_*.csv values as a check.

Paper style (plot_style_bw.RC_PAPER): Times bold, no titles, opaque label boxes, 600 dpi + PDF.

Usage
-----
    python plot_tile_paper.py --tile-station ISMN_FMI_SOD101 \
        --stations ISMN_FMI_SOD101 ISMN_FMI_SOD102 ISMN_FMI_SOD103 \
                   ISMN_FMI_SOD071 ISMN_FMI_SOD072 ISMN_FMI_SOD073 \
        --name fmi_sodankyla
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pyproj import Transformer

import plot_style_bw
from eval_metrics import SM_DEPTHS, _make_key, metrics_from_arrays
from visualize_embeddings import _dstr, cloud_frac_by_date

REPO      = Path(__file__).resolve().parent
SAT_ZARR  = Path("/projects/prjs1968/satellite_zarr")
TOK_ZARR  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SPLITS    = REPO / "csvs" / "station_splits.csv"
RUN       = "baseline_selected_20261005"
DEPTH_LABELS = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
PRED_COLOR, OBS_COLOR, OOST_SHADE = "#3776AB", "black", plot_style_bw.RED
GAP_DAYS  = 15                      # break the line across prediction gaps longer than this
OOST_FROM = pd.Timestamp("2023-01-01")


# ---------------------------------------------------------------------------
def station_pixels(raw, meta: pd.DataFrame) -> pd.DataFrame:
    """lat/lon -> (row, col) on the tile's 224 x 10 m grid; pixel centres at integers."""
    a = raw.attrs
    xmin, ymin, xmax, ymax = a["bounds_utm"]
    res = float(a["pixel_size_m"])
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(a['epsg'])}", always_xy=True)
    x, y = tr.transform(meta["longitude"].to_numpy(), meta["latitude"].to_numpy())
    out = meta.copy()
    out["col"] = (x - xmin) / res - 0.5
    out["row"] = (ymax - y) / res - 0.5
    out["x"], out["y"] = x, y
    return out


def pick_rgb_scene(raw, tok, want: str | None) -> tuple[int, str, float]:
    """Least-cloudy June-August scene with full coverage; ties -> closest to mid-July."""
    dates = _dstr(raw["s2/dates"][:])
    if want:
        idx = int(np.where(dates == want)[0][0])
        return idx, want, float("nan")
    frac = cloud_frac_by_date(tok)
    best = None
    for i, d in enumerate(dates):
        if len(d) != 8 or d[4:6] not in ("06", "07", "08") or d not in frac:
            continue
        key = (round(frac[d], 3), abs(int(d[4:8]) - 715))
        if best is None or key < best[0]:
            best = (key, i, d)
    if best is None:
        raise SystemExit("no summer S2 scene with a cloud mask")
    _, i, d = best
    img = raw["s2/data"][i]
    if (img == 0).all(axis=0).mean() > 0.01:
        print(f"  WARNING {d}: {(img == 0).all(axis=0).mean():.1%} nodata pixels")
    return i, d, float(frac[d])


def s2_rgb(raw, idx: int) -> np.ndarray:
    """True colour (B04, B03, B02), 2-98 % stretch per channel; band order asserted."""
    bands = list(raw["s2"].attrs["bands"])
    img = raw["s2/data"][idx].astype(np.float32)
    out = np.zeros(img.shape[1:] + (3,), np.float32)
    for k, b in enumerate(["B04", "B03", "B02"]):
        c = img[bands.index(b)]
        p2, p98 = np.percentile(c, (2, 98))
        out[..., k] = np.clip((c - p2) / max(p98 - p2, 1e-6), 0, 1)
    return out


def merge_close(st: pd.DataFrame, merge_m: float) -> list[pd.DataFrame]:
    """Greedy single-link grouping of stations closer than merge_m (they share one marker)."""
    left, groups = list(st.index), []
    while left:
        g = [left.pop(0)]
        changed = True
        while changed:
            changed = False
            for j in list(left):
                if any(np.hypot(st.at[j, "x"] - st.at[i, "x"], st.at[j, "y"] - st.at[i, "y"])
                       < merge_m for i in g):
                    g.append(j); left.remove(j); changed = True
        groups.append(st.loc[g])
    return groups


def group_label(g: pd.DataFrame) -> str:
    names = list(g["short"])
    if len(names) == 1:
        return names[0]
    pre = names[0].rstrip("0123456789")
    if all(n.startswith(pre) for n in names):
        nums = sorted(n[len(pre):] for n in names)
        return f"{pre}{nums[0]}–{nums[-1]}" if len(nums) > 2 else f"{pre}{nums[0]}/{nums[1]}"
    return "/".join(names)


def scale_bar(ax, x0, y0, metres, res, color="white"):
    w = metres / res
    ax.add_patch(mpatches.Rectangle((x0, y0), w, max(w * 0.06, 0.6), fc=color, ec="black",
                                    lw=0.8, zorder=8))
    ax.text(x0 + w / 2, y0 - max(w * 0.08, 0.8), f"{metres:g} m", ha="center", va="bottom",
            color="black", zorder=9, bbox=dict(fc="white", ec="black", lw=0.6, pad=1.5))


def ylim_all(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
    pad = 0.06 * max(hi - lo, 0.02)
    return min(0.0, lo) - pad * 0.6, hi + pad * 2


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-station", required=True, help="station whose tile is drawn")
    ap.add_argument("--stations", nargs="+", required=True)
    ap.add_argument("--name", required=True, help="output file stem")
    ap.add_argument("--pred-dir", default=f"eval_output/{RUN}/allday")
    ap.add_argument("--metrics-dir", default=f"eval_output/{RUN}")
    ap.add_argument("--out-dir", default=f"figures/eval/{RUN}_paper/tiles")
    ap.add_argument("--rgb-date", default=None, help="YYYYMMDD; default = least-cloudy summer scene")
    ap.add_argument("--merge-m", type=float, default=20.0)
    ap.add_argument("--category", default="sm_only")
    ap.add_argument("--unbiased", action="store_true",
                    help="shift each station x depth prediction by its observed-day mean bias")
    a = ap.parse_args()
    plt.rcParams.update(plot_style_bw.RC_PAPER)

    # ── stations + geometry ────────────────────────────────────────────────
    sp = pd.read_csv(SPLITS)                      # pandas only: quoted commas (AmeriFlux names)
    sp["key"] = sp.apply(_make_key, axis=1)
    meta = sp.set_index("key").loc[a.stations].reset_index()
    meta["short"] = [str(r.station_name) if r.source_network == "ISMN" else str(r.station_id)
                     for r in meta.itertuples()]     # AmeriFlux names are long: use the site id
    raw = zarr.open_group(str(SAT_ZARR / f"{a.tile_station}.zarr"), mode="r")
    tok = zarr.open_consolidated(str(TOK_ZARR / a.category / a.tile_station))
    st = station_pixels(raw, meta)
    res = float(raw.attrs["pixel_size_m"]); npx = raw["s2/data"].shape[-1]
    print("station pixels (tile centre should be ~111.5):")
    print(st[["key", "split", "latitude", "longitude", "row", "col"]].to_string(index=False))
    if not ((st[["row", "col"]] >= 0) & (st[["row", "col"]] < npx)).all().all():
        raise SystemExit("a station falls outside the tile")

    # ── predictions (every day) + observed ─────────────────────────────────
    frames = []
    for s in ("oos", "oot", "oost"):
        f = Path(a.pred_dir) / f"predictions_{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            frames.append(d[d["station_key"].isin(a.stations)])
    df = pd.concat(frames, ignore_index=True)
    df["depth"] = df["depth"].astype(str)
    df = df.drop_duplicates(["station_key", "depth", "date"], keep="first")
    df["pred"] = df["pred"].clip(0.0, 1.0)                 # EVAL_CLIP=0,1, as in the eval
    missing = sorted(set(a.stations) - set(df["station_key"]))
    if missing:
        raise SystemExit(f"no predictions for {missing}")

    # panels: stations grouped by marker group (one column per group), depths in order
    groups = merge_close(st, a.merge_m)
    cols = []
    for g in groups:
        panels = []
        for key in g["key"]:
            for dep in SM_DEPTHS:
                q = df[(df["station_key"] == key) & (df["depth"] == dep)]
                if q["obs"].notna().any():
                    q = q.sort_values("date").copy()
                    o = q["obs"].notna()
                    q.attrs["offset"] = float((q.loc[o, "pred"] - q.loc[o, "obs"]).mean())
                    if a.unbiased:     # shift the whole line by the observed-day mean bias
                        q["pred"] = q["pred"] - q.attrs["offset"]
                    panels.append((key, dep, q))
        cols.append((g, panels))
    if len(cols) > 2:            # many separate stations (e.g. TxSON): two balanced columns
        flat = [p for _, ps in cols for p in ps]
        h = (len(flat) + 1) // 2
        cols = [(None, flat[:h]), (None, flat[h:])]
    nrow = max(len(p) for _, p in cols)
    ncol = len(cols)

    # ── metrics check against the eval CSVs ────────────────────────────────
    ref = pd.concat([pd.read_csv(f) for f in Path(a.metrics_dir).glob("per_station_*.csv")],
                    ignore_index=True)
    ref = ref[ref["eval_split"] == "oos"].set_index("station_key")
    print("\nstation x depth: ubRMSE here vs per_station_oos.csv  (n here / n csv)")

    # ── figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.0 * ncol + 0.6, 5.1 + 1.75 * nrow))
    # rows: images | legend strip | time series -- the legend gets its own space
    outer = GridSpec(3, 1, figure=fig, height_ratios=[4.6, 0.5, 1.75 * nrow], hspace=0.04,
                     left=0.07, right=0.985, top=0.975, bottom=0.04)

    idx, date, cf = pick_rgb_scene(raw, tok, a.rgb_date)
    print(f"\nRGB scene {date} (cloud fraction {cf:.3f})")
    rgb = s2_rgb(raw, idx)
    date_s = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    # zoom window: station bounding box + margin, at least 400 m, kept inside the tile.
    # Stations spread over most of the tile (TxSON) -> no zoom, ONE panel with the markers.
    cx, cy = st["col"].mean(), st["row"].mean()
    half = max(0.5 * max(np.ptp(st["col"]), np.ptp(st["row"])) + 12, 200 / res)
    zoom = 2 * half <= 0.55 * npx
    if zoom:
        x0 = float(np.clip(cx - half, -0.5, npx - 0.5 - 2 * half))
        y0 = float(np.clip(cy - half, -0.5, npx - 0.5 - 2 * half))
        box = (x0, y0, 2 * half)
    else:
        box = (-0.5, -0.5, float(npx))

    def draw_markers(ax, box):
        """Marker per group; label pushed radially away from the stations' centroid."""
        for g in groups:
            gx, gy = g["col"].mean(), g["row"].mean()
            ax.plot(gx, gy, marker="o", ms=11, mfc=plot_style_bw.RED, mec="white", mew=2.0,
                    zorder=7)
            v = np.array([gx - cx, gy - cy])
            u = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([0.0, 1.0])
            d = 0.075 * box[2]
            ax.annotate(group_label(g), (gx, gy), xytext=(gx + u[0] * d, gy + u[1] * d),
                        ha="left" if u[0] > 0.35 else "right" if u[0] < -0.35 else "center",
                        va="top" if u[1] > 0.35 else "bottom" if u[1] < -0.35 else "center",
                        zorder=8, bbox=dict(fc="white", ec="black", lw=0.8, pad=2.5))

    if zoom:
        top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.06)
        axA = fig.add_subplot(top[0, 0])
        axA.imshow(rgb, interpolation="nearest")
        axA.add_patch(mpatches.Rectangle((box[0], box[1]), box[2], box[2], fill=False,
                                         ec="white", lw=2.0, zorder=5))
        axA.add_patch(mpatches.Rectangle((box[0], box[1]), box[2], box[2], fill=False,
                                         ec="black", lw=0.8, ls="--", zorder=6))
        scale_bar(axA, 10, npx - 14, 500, res)
        axA.set_xticks([]); axA.set_yticks([])
        axA.text(0.0, 1.01, "(a)", transform=axA.transAxes, va="bottom", ha="left")

        axB = fig.add_subplot(top[0, 1])
        axB.imshow(rgb, interpolation="nearest")
        axB.set_xlim(box[0], box[0] + box[2]); axB.set_ylim(box[1] + box[2], box[1])
        draw_markers(axB, box)
        scale_bar(axB, box[0] + 0.06 * box[2], box[1] + 0.9 * box[2], 100, res)
        axB.set_xticks([]); axB.set_yticks([])
        for s in axB.spines.values():
            s.set_linestyle("--")
        axB.text(0.0, 1.01, f"(b)   Sentinel-2 true colour, {date_s}",
                 transform=axB.transAxes, va="bottom", ha="left")
        first_letter = "c"
    else:
        top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.0)
        axA = fig.add_subplot(top[0, 1])
        axA.imshow(rgb, interpolation="nearest")
        draw_markers(axA, box)
        scale_bar(axA, npx - 10 - 500 / res, npx - 14, 500, res)    # bottom-right: clear of labels
        axA.set_xlim(-0.5, npx - 0.5); axA.set_ylim(npx - 0.5, -0.5)
        axA.set_xticks([]); axA.set_yticks([])
        axA.text(0.0, 1.01, f"(a)   Sentinel-2 true colour, {date_s}",
                 transform=axA.transAxes, va="bottom", ha="left")
        first_letter = "b"

    allv = np.concatenate([np.r_[p[2]["pred"].to_numpy(float), p[2]["obs"].to_numpy(float)]
                           for _, ps in cols for p in ps])
    ylo, yhi = ylim_all(allv)
    ts = GridSpecFromSubplotSpec(nrow, ncol, subplot_spec=outer[2], hspace=0.55, wspace=0.12)
    letter = ord(first_letter)
    first_ax, shaded = None, False
    for j, (g, panels) in enumerate(cols):
        share = None
        for i, (key, dep, q) in enumerate(panels):
            ax = fig.add_subplot(ts[i, j], sharex=share)
            share = share or ax
            first_ax = first_ax or ax
            dates = q["date"].to_numpy()
            pred = q["pred"].to_numpy(np.float64).copy()
            gap = np.diff(dates).astype("timedelta64[D]").astype(int) > GAP_DAYS
            pred[np.append(gap, False)] = np.nan
            o = q[q["obs"].notna()]
            if o["date"].max() >= OOST_FROM:
                ax.axvspan(OOST_FROM, q["date"].max(), color=OOST_SHADE, alpha=0.12, lw=0, zorder=0)
                ax.axvline(OOST_FROM, color=OOST_SHADE, lw=0.9, ls="--", zorder=1)
                shaded = True
            ax.plot(o["date"], o["obs"], ".", ms=2.2, color=OBS_COLOR, label="observed", zorder=2)
            ax.plot(dates, pred, "-", lw=1.1, color=PRED_COLOR, label="predicted (bias removed)" if a.unbiased else "predicted", zorder=3)
            ax.set_ylim(ylo, yhi)
            ax.margins(x=0.01)
            m = metrics_from_arrays(o["pred"].to_numpy(np.float64), o["obs"].to_numpy(np.float64))
            btxt = (f"bias removed {q.attrs['offset']:+.3f}" if a.unbiased
                    else f"bias {m['bias']:+.3f}")
            ax.text(1.0, 1.02, f"ubRMSE {m['ubRMSE']:.3f}  r {m['R']:.2f}  {btxt}"
                               f"  n {m['n']}", transform=ax.transAxes, ha="right", va="bottom")
            short = key.split("_")[-1]
            ax.text(0.0, 1.02, f"({chr(letter)}) {short}, {DEPTH_LABELS[dep]}",
                    transform=ax.transAxes, ha="left", va="bottom")
            letter += 1
            loc = mdates.AutoDateLocator(maxticks=7)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            if i < len(panels) - 1:
                ax.tick_params(labelbottom=False)
            c = "_".join(dep.split("-")); r = ref.loc[key] if key in ref.index else None
            rv = r[f"ubRMSE_{c}"] if r is not None else float("nan")
            rn = r[f"n_{c}"] if r is not None else float("nan")
            print(f"  {key:18s} {dep:7s} {m['ubRMSE']:.4f} vs {rv:.4f}   ({m['n']} / {rn:.0f})")

    handles, labels = first_ax.get_legend_handles_labels()
    if shaded:
        handles.append(mpatches.Patch(facecolor=OOST_SHADE, alpha=0.12, edgecolor=OOST_SHADE,
                                      ls="--"))
        labels.append("2023–2025 (OOST)")
    axL = fig.add_subplot(outer[1]); axL.axis("off")
    axL.legend(handles, labels, loc="center", ncol=len(labels), frameon=False, markerscale=5)
    fig.supylabel("Soil moisture (m$^3$/m$^3$)", x=0.005, y=outer[2].get_position(fig).y1 / 2,
                  fontsize=plt.rcParams["axes.labelsize"], fontweight="bold")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{a.name}.{ext}", dpi=600)
    print(f"\nwrote {out}/{a.name}.{{png,pdf}}")


if __name__ == "__main__":
    main()
