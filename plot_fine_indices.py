"""Visual check of the §57/§61 fine-path NDVI and NDMI (user request 2026-10-05).

For a few stations, the clearest summer S2 scene in the raw store is shown as
    RGB (10 m) | index direct from raw DN (fp64, 20 m) | index as the model sees it
    (fp16 fine tensor -> model.fine_to_indices, current floor) | |error| with the OLD floor (10 DN)
    | |error| with the current floor
for NDVI (row 1) and NDMI (row 2). The direct reference uses no floor, so the error panels also
show which pixels the floor zeroes (water, deep shadow).

    python plot_fine_indices.py [--stations A B ...] [--out-dir figures/fine_indices]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

import model as M
from dataset import RAW_ROOT, _load_fine_stats, fine_s2_scene
from model import FINE_CH, fine_to_indices

DEFAULT_STATIONS = ["ISMN_REMEDHUS_Canizal", "AmeriFlux_US-Ne1", "ICOS_FI-Hyy", "ICOS_SE-Deg"]
RAW_B = {"B02": 1, "B03": 2, "B04": 3, "B08": 7, "B8A": 8, "B11": 10}   # 12-band raw order
OFF = M.S2_BOA_OFFSET


def open_store(p):
    try:
        return zarr.open_consolidated(str(p), mode="r")
    except KeyError:
        return zarr.open_group(str(p), mode="r")


def pick_scene(rg):
    """Clearest Jun-Aug scene: all 12 bands non-zero almost everywhere, fewest bright-blue pixels."""
    dates = [str(bytes(d).decode() if isinstance(d, (bytes, np.bytes_)) else d)[:8]
             for d in rg["s2/dates"][:]]
    best = None
    for i, d in enumerate(dates):
        if d[4:6] not in ("06", "07", "08"):
            continue
        x = np.asarray(rg["s2/data"][i], dtype=np.float32)
        valid = (x != 0).all(axis=0).mean()
        if valid < 0.99:
            continue
        cloud = ((x[RAW_B["B02"]] - OFF) > 1500).mean()        # blue reflectance > 0.15
        if best is None or cloud < best[0]:
            best = (cloud, i, d, x)
    return best


def pool(a, m):
    """Masked 2x2 mean, 224 -> 112 (as dataset._pool2)."""
    mf = m.astype(np.float64)
    s = (a * mf).reshape(112, 2, 112, 2).sum(axis=(1, 3))
    n = mf.reshape(112, 2, 112, 2).sum(axis=(1, 3))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def direct_nd(x, a, b):
    """Reference index at 20 m from raw DN in fp64, no floor (NaN where a+b == 0)."""
    m = (x != 0).all(axis=0)
    pa, pb = pool(x[RAW_B[a]].astype(np.float64) - OFF, m), pool(x[RAW_B[b]].astype(np.float64) - OFF, m)
    den = pa + pb
    return np.where(den != 0, (pa - pb) / np.where(den != 0, den, 1), np.nan).clip(-1, 1), den


def model_path(x, fs, floor):
    """fp16 fine tensor -> fine_to_indices with a given denominator floor."""
    sc = fine_s2_scene(x, np.zeros((224, 224), np.uint8), fs)
    f = np.zeros((FINE_CH, 112, 112), np.float32)
    f[0:11] = sc
    old = M._IDX_DEN_FLOOR
    M._IDX_DEN_FLOOR = floor
    try:
        out = fine_to_indices(torch.from_numpy(f.astype(np.float16)).float()[None])[0].numpy()
    finally:
        M._IDX_DEN_FLOOR = old
    return out[0], out[1]                                      # NDVI, NDMI


def rgb(x):
    c = np.stack([x[RAW_B[b]] for b in ("B04", "B03", "B02")], -1).astype(np.float32) - OFF
    return np.clip(c / 3000.0, 0, 1) ** (1 / 1.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    ap.add_argument("--out-dir", default="figures/fine_indices")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fs = _load_fine_stats()
    cur = M._IDX_DEN_FLOOR
    plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "savefig.bbox": "tight"})

    print(f"current floor {cur:g} DN, old floor 10 DN")
    print(f"{'station':28s} {'date':8s} idx    p99|d|old  frac>.02old  p99|d|new  frac>.02new  zeroed_new")
    for st in args.stations:
        p = RAW_ROOT / f"{st}.zarr"
        if not p.exists():
            print(f"{st}: no raw store, skipped")
            continue
        rg = open_store(p)
        if "s2/data" not in rg:
            print(f"{st}: no s2, skipped")
            continue
        sel = pick_scene(rg)
        if sel is None:
            print(f"{st}: no clear Jun-Aug scene, skipped")
            continue
        cloud, i, date, x = sel
        ndvi_d, den_v = direct_nd(x, "B08", "B04")
        ndmi_d, den_m = direct_nd(x, "B8A", "B11")
        ndvi_old, ndmi_old = model_path(x, fs, 10.0)
        ndvi_new, ndmi_new = model_path(x, fs, cur)

        fig, ax = plt.subplots(2, 5, figsize=(16, 6.6))
        for r, (name, ref, old, new, den, cmap) in enumerate((
                ("NDVI", ndvi_d, ndvi_old, ndvi_new, den_v, "RdYlGn"),
                ("NDMI", ndmi_d, ndmi_old, ndmi_new, den_m, "BrBG"))):
            ok = np.isfinite(ref)
            e_old, e_new = np.abs(old - ref), np.abs(new - ref)
            stats = []
            for e in (e_old, e_new):
                v = e[ok]
                stats += [np.percentile(v, 99), (v > 0.02).mean()]
            zeroed = ((np.abs(den) <= cur) & ok).mean()
            print(f"{st:28s} {date} {name}  {stats[0]:9.3e}  {stats[1]:11.2e}  {stats[2]:9.3e}  "
                  f"{stats[3]:11.2e}  {zeroed:10.2e}")
            ax[r, 0].imshow(rgb(x))
            ax[r, 0].set_title(f"RGB 10 m  {date}" if r == 0 else "RGB 10 m")
            for c, (img, ttl) in enumerate(((ref, f"{name} direct (raw DN, fp64)"),
                                            (new, f"{name} model path (fp16, floor {cur:g})")), 1):
                im = ax[r, c].imshow(img, cmap=cmap, vmin=-1, vmax=1)
                ax[r, c].set_title(ttl)
                fig.colorbar(im, ax=ax[r, c], fraction=0.046, pad=0.02)
            for c, (e, ttl) in enumerate(((e_old, "|error| old floor 10 DN"),
                                          (e_new, f"|error| floor {cur:g} DN")), 3):
                im = ax[r, c].imshow(np.where(ok, e, np.nan), cmap="magma", vmin=0, vmax=0.1)
                ax[r, c].set_title(f"{ttl}\np99 {np.percentile(e[ok], 99):.3f}, "
                                   f">0.02: {(e[ok] > 0.02).mean():.1%}")
                fig.colorbar(im, ax=ax[r, c], fraction=0.046, pad=0.02)
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{st}   scene {date}   (bright-blue fraction {cloud:.1%})", y=1.0)
        fp = out / f"{st}_{date}.png"
        fig.savefig(fp)
        plt.close(fig)
        print(f"  wrote {fp}")


if __name__ == "__main__":
    main()
