"""
TerraMind embedding diagnostic (Tier-0).  See text/training_runbook.md §15.

Builds, per station, a figure whose rows are modalities (S2 × 4 climate-aware seasons,
S1 ascending, S1 descending if present, DEM, LULC) and whose columns are:
    raw scene  →  per-token L2 norm (14×14)  →  PCA→RGB (14×14)
plus per-panel metrics (off-diagonal cosine, PCA top-3 variance ratio, neighbour
autocorrelation).  The figure answers whether frozen TerraMind L12 tokens carry
scene-tracking, land-cover-discriminative structure (Tier 1) or wash out (Tier 2).

Data facts (verified 2026-07-01):
  Token store : /gpfs/scratch1/shared/pkhanal/zarr/{category}/{station}/   (open_consolidated)
      s2/{l3,l6,l9,l12} (N,196,768) fp16 + s2/dates (N,) U8
      s1_asc/{...} + s1_asc/dates + s1_asc/token_mask (M,14,14) bool ; s1_desc/* optional
      cm/masks (N,224,224) uint8  (0=clear, nonzero=cloud/shadow) + cm/dates (index-aligned to s2)
      dem, lulc (196,768) fp16 static + dem_token_mask, lulc_token_mask (14,14) bool
  Raw store   : /projects/prjs1968/satellite_zarr/<token_dir_name>.zarr
      s2/data (T,12,224,224) int16 + s2/dates ; bands B01..B12 → RGB = idx 3,2,1 (B04/B03/B02)
      s1_asc/data (T,2,224,224) fp16 (VV,VH) ; dem/data (1,224,224) ; lulc/data (Y,224,224) uint8
      root attrs: latitude, longitude, epsg, pixel_size_m=10, patch_size_px=224

Decisions (locked): climate-aware seasons (hemisphere flip; wet/dry labels for Köppen A/B);
held-out stations only (split ∈ {val, oos}); cloud-free = mean(cm != 0) < 0.08 whole-patch.

Usage:
    python visualize_embeddings.py --dry-run                 # Phase 1 selection only
    python visualize_embeddings.py                           # select 4 + render
    python visualize_embeddings.py --stations A,B,C,D        # explicit stations
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr")
RAW_ROOT  = Path("/projects/prjs1968/satellite_zarr")
SPLITS    = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
OUT_DIR   = Path("/gpfs/work3/0/prjs1968/soilMoisture/embed_viz_output")

CATEGORIES   = ("sm_only", "sm_and_flux", "flux_only")
CLOUD_THRESH = 0.08          # whole-patch fraction of nonzero cm pixels
RGB_IDX      = (3, 2, 1)     # B04, B03, B02 in the 12-band raw S2 stack
HELD_OUT     = ("val", "oos")
LAYERS       = ("l3", "l6", "l9", "l12")   # per-layer PCA→RGB sweep (depth degradation)


# ── station <-> path helpers ────────────────────────────────────────────────

def category_of(row) -> str:
    if row["has_soil_moisture"] and row["has_flux"]:
        return "sm_and_flux"
    if row["has_soil_moisture"]:
        return "sm_only"
    return "flux_only"


def station_dir_name(row) -> str:
    if str(row["source_network"]) == "ISMN":
        return f"ISMN_{row['network']}_{row['station_name']}"
    return f"{row['source_network']}_{row['station_id']}"


def build_disk_index() -> dict:
    """name -> category for every store on disk."""
    idx = {}
    for cat in CATEGORIES:
        d = ZARR_ROOT / cat
        if d.is_dir():
            for n in os.listdir(d):
                idx[n] = cat
    return idx


def open_tokens(name: str, cat: str):
    return zarr.open_consolidated(str(ZARR_ROOT / cat / name), mode="r")


def open_raw(name: str):
    p = RAW_ROOT / f"{name}.zarr"
    if not p.exists():
        return None
    try:
        return zarr.open_consolidated(str(p), mode="r")
    except Exception:
        return zarr.open_group(str(p), mode="r")


def _dstr(a) -> np.ndarray:
    """Decode a dates array (bytes or str) to a python str ndarray."""
    a = np.asarray(a)
    if a.dtype.kind == "S":
        return np.array([x.decode() for x in a])
    return a.astype(str)


# ── climate-aware seasons ───────────────────────────────────────────────────

_TEMPERATE = ["Winter", "Spring", "Summer", "Autumn"]
_TROPICAL  = ["Dry(cool)", "Dry(warm)", "Wet(monsoon)", "Wet→Dry"]
_Q = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


def season_slot(month: int, lat: float, kg: str):
    """Return (slot 0..3, label). Southern hemisphere flips by 6 months; Köppen A/B
    (tropical/arid) get wet/dry labels instead of temperate-season names."""
    m = ((month + 5) % 12) + 1 if lat < 0 else month     # SH → shift 6 months
    q = _Q[m]
    labels = _TROPICAL if str(kg) in ("A", "B") else _TEMPERATE
    return q, labels[q]


# ── per-station analysis ────────────────────────────────────────────────────

def cloud_frac_by_date(g) -> dict:
    """date-string -> mean(cm != 0). Reads the whole cm/masks array once — it is a single
    (128,224,224) chunk, so per-index access would re-decompress the full 6.4 MB chunk on
    every acquisition. cm/dates is not guaranteed positionally aligned to s2/dates, so we
    key by date string and look up per S2 acquisition."""
    masks = g["cm/masks"][:]                      # (N,224,224) one decompress
    fr = (masks != 0).reshape(masks.shape[0], -1).mean(axis=1).astype(np.float64)
    return dict(zip(_dstr(g["cm/dates"][:]), fr))


def choose_seasonal_s2(g, lat: float, kg: str):
    """Pick the least-cloudy cloud-free S2 acquisition per season slot.
    Returns dict slot -> (idx, date, cloud_frac, label) for slots that qualify."""
    s2_dates = _dstr(g["s2/dates"][:])
    cm_frac  = cloud_frac_by_date(g)

    chosen = {}
    for i, d in enumerate(s2_dates):
        if len(d) != 8 or not d.isdigit():
            continue
        f = cm_frac.get(d)                         # None if no cloud mask for this date
        if f is None or f >= CLOUD_THRESH:
            continue
        slot, label = season_slot(int(d[4:6]), lat, kg)
        if slot not in chosen or f < chosen[slot][2]:
            chosen[slot] = (i, d, f, label)
    return chosen


def analyse_station(name: str, cat: str, row) -> dict | None:
    """Phase-1 feasibility: openable store, 4 cloud-free seasons, raw present."""
    try:
        g = open_tokens(name, cat)
    except Exception as e:
        return None
    if "s2" not in g or "cm" not in g:
        return None
    lat = float(row["latitude"])
    chosen = choose_seasonal_s2(g, lat, row["kg_macro"])
    return {
        "name": name, "cat": cat, "lat": lat, "lon": float(row["longitude"]),
        "igbp": row["igbp_macro"], "kg": row["kg_macro"],
        "koppen": row["koppen_geiger"], "split": row["split"],
        "seasons": chosen, "n_seasons": len(chosen),
        "has_desc": "s1_desc" in g, "has_raw": open_raw(name) is not None,
    }


# ── Phase 1: selection ──────────────────────────────────────────────────────

def select_stations(df: pd.DataFrame, disk: dict, n: int = 4) -> list:
    """Pick n stations with 4 cloud-free seasons, contrasting IGBP×Köppen, raw present."""
    df = df[df["split"].isin(HELD_OUT)].copy()
    feas = []
    for _, row in df.iterrows():
        name = station_dir_name(row)
        if name not in disk:
            continue
        info = analyse_station(name, disk[name], row)
        if info and info["n_seasons"] == 4 and info["has_raw"]:
            feas.append(info)
    print(f"[phase1] feasible (4 seasons + raw, held-out): {len(feas)}")

    # one per contrasting IGBP bucket, diversify Köppen, prefer s1_desc
    order = ["Forest", "Grass-Crop", "Shrub-Savanna", "Other"]
    picked, used_kg = [], set()
    for bucket in order:
        cands = [s for s in feas if s["igbp"] == bucket and s["name"] not in
                 {p["name"] for p in picked}]
        cands.sort(key=lambda s: (s["kg"] in used_kg, not s["has_desc"]))
        if cands:
            picked.append(cands[0])
            used_kg.add(cands[0]["kg"])
        if len(picked) == n:
            break
    # backfill if a bucket was empty
    if len(picked) < n:
        rest = [s for s in feas if s["name"] not in {p["name"] for p in picked}]
        rest.sort(key=lambda s: (s["kg"] in used_kg, not s["has_desc"]))
        picked.extend(rest[: n - len(picked)])
    return picked


def print_selection(picked: list):
    print("\n=== Phase 1 selection ===")
    for s in picked:
        print(f"\n{s['name']}  [{s['split']}]  IGBP={s['igbp']}  "
              f"Koppen={s['koppen']}({s['kg']})  lat={s['lat']:.2f} lon={s['lon']:.2f}  "
              f"s1_desc={'Y' if s['has_desc'] else 'N'}")
        for slot in sorted(s["seasons"]):
            i, d, f, label = s["seasons"][slot]
            print(f"    slot{slot} {label:14s} date={d} cloud={f*100:4.1f}%  (s2 idx {i})")


# ── token maps + metrics ────────────────────────────────────────────────────

def token_norm_map(tok: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """tok (196,768) → (14,14) L2 norm; invalid tokens set to NaN."""
    norm = np.linalg.norm(tok.astype(np.float32), axis=-1).reshape(14, 14)
    if valid is not None:
        norm = np.where(valid.reshape(14, 14), norm, np.nan)
    return norm


def pca_rgb(tok: np.ndarray, valid: np.ndarray | None = None):
    """tok (196,768) → (14,14,3) in [0,1] via PCA(3); returns (rgb, var_ratio[3])."""
    x = tok.astype(np.float32)
    m = np.ones(196, bool) if valid is None else valid.astype(bool)
    xm = x[m]
    xc = xm - xm.mean(0, keepdims=True)
    # SVD-based PCA
    U, S, Vt = np.linalg.svd(xc, full_matrices=False)
    comps = Vt[:3]                                  # (3,768)
    proj_all = (x - xm.mean(0, keepdims=True)) @ comps.T   # (196,3)
    var = (S ** 2)
    var_ratio = (var[:3] / var.sum()) if var.sum() > 0 else np.zeros(3)
    rgb = proj_all.reshape(14, 14, 3)
    # normalise each channel on valid tokens
    vm = m.reshape(14, 14)
    for c in range(3):
        ch = rgb[..., c][vm]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    rgb[~vm] = 0.5                                   # grey out padded tokens
    return rgb, var_ratio


def panel_metrics(tok: np.ndarray, valid: np.ndarray | None = None) -> dict:
    x = tok.astype(np.float32)
    m = np.ones(196, bool) if valid is None else valid.astype(bool)
    xv = x[m]
    xn = xv / (np.linalg.norm(xv, axis=1, keepdims=True) + 1e-8)
    cos = xn @ xn.T
    iu = np.triu_indices(cos.shape[0], k=1)
    off = float(cos[iu].mean()) if iu[0].size else float("nan")
    # neighbour autocorrelation on the norm map
    nm = token_norm_map(tok, valid)
    fin = np.isfinite(nm)
    ac = float("nan")
    if fin.sum() > 4:
        a, b = nm[:, :-1], nm[:, 1:]
        pair = np.isfinite(a) & np.isfinite(b)
        if pair.sum() > 4:
            ac = float(np.corrcoef(a[pair], b[pair])[0, 1])
    return {"offdiag_cos": off, "neighbor_ac": ac}


# ── raw scene rendering ─────────────────────────────────────────────────────

def raw_s2_rgb(raw, date: str) -> np.ndarray | None:
    dates = _dstr(raw["s2/dates"][:])
    hit = np.where(dates == date)[0]
    if hit.size == 0:
        return None
    img = raw["s2/data"][int(hit[0])][list(RGB_IDX)].astype(np.float32)  # (3,224,224)
    img = np.transpose(img, (1, 2, 0)) / 3000.0                          # DN → ~reflectance
    return np.clip(img, 0, 1)


def raw_s1_gray(raw, orbit: str, date: str) -> np.ndarray | None:
    key = f"s1_{orbit}"
    if key not in raw or "dates" not in raw[key]:
        return None
    dates = _dstr(raw[f"{key}/dates"][:])
    hit = np.where(dates == date)[0]
    if hit.size == 0:
        return None
    vv = raw[f"{key}/data"][int(hit[0])][0].astype(np.float32)           # VV
    lo, hi = np.percentile(vv, 2), np.percentile(vv, 98)
    return np.clip((vv - lo) / (hi - lo + 1e-8), 0, 1)


# ── Phase 3: figure ─────────────────────────────────────────────────────────

def render_station(info: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = info["name"]
    g = open_tokens(name, info["cat"])
    raw = open_raw(name)

    # assemble rows: (row_label, raw_img, {layer -> (196,768) | None}, valid(196,)|None)
    # Columns are a per-layer PCA→RGB sweep L3→L6→L9→L12 to show at what depth spatial
    # scene structure collapses (Tier-0). Raw scene comes from satellite_zarr (real imagery).
    rows = []
    s1a_dates = _dstr(g["s1_asc/dates"][:]) if "s1_asc" in g else np.array([])
    s1d_dates = _dstr(g["s1_desc/dates"][:]) if "s1_desc" in g else np.array([])

    def _layers(group_key, idx):
        return {l: (g[f"{group_key}/{l}"][idx] if f"{group_key}/{l}" in g else None)
                for l in LAYERS}

    for slot in sorted(info["seasons"]):
        i, d, f, label = info["seasons"][slot]
        rgb = raw_s2_rgb(raw, d) if raw is not None else None
        rows.append((f"S2 {label}\n{d} ({f*100:.0f}% cld)", rgb, _layers("s2", i), None))

    # one S1 asc + one desc, nearest to the summer/wet S2 date if available
    ref_date = info["seasons"].get(2, info["seasons"][sorted(info["seasons"])[0]])[1]
    for orbit, odates in (("asc", s1a_dates), ("desc", s1d_dates)):
        gk = f"s1_{orbit}"
        if gk not in g or odates.size == 0:
            continue
        j = int(np.argmin(np.abs(odates.astype(int) - int(ref_date))))
        tm = g[f"{gk}/token_mask"][j].reshape(-1) if "token_mask" in g[gk] else None
        gray = raw_s1_gray(raw, orbit, odates[j]) if raw is not None else None
        img = None if gray is None else np.stack([gray] * 3, -1)
        rows.append((f"S1 {orbit}\n{odates[j]}", img, _layers(gk, j), tm))

    # static DEM / LULC — L12 only
    for mod in ("dem", "lulc"):
        if mod not in g:
            continue
        tmk = f"{mod}_token_mask"
        tm = g[tmk][:].reshape(-1) if tmk in g else None
        img = None
        if raw is not None and mod in raw:
            arr = raw[f"{mod}/data"][:]
            plane = (arr[0] if mod == "dem" else arr[-1]).astype(np.float32)
            if mod == "dem":
                lo, hi = np.nanpercentile(plane, 2), np.nanpercentile(plane, 98)
                img = np.clip((plane - lo) / (hi - lo + 1e-8), 0, 1)
            else:
                img = plane            # categorical LULC classes → discrete cmap in plot loop
        layers = {l: (g[mod][:] if l == "l12" else None) for l in LAYERS}
        rows.append((mod.upper(), img, layers, tm))

    cols = ["raw scene"] + [l.upper() for l in LAYERS]      # raw, L3, L6, L9, L12
    ncol = len(cols)
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncol, figsize=(2.6 * ncol, 2.6 * nrows), squeeze=False)
    fig.suptitle(f"{name}   IGBP={info['igbp']}  Koppen={info['koppen']}  [{info['split']}]"
                 f"   —  per-layer PCA→RGB sweep", fontsize=11)
    for c, title in enumerate(cols):
        axes[0, c].set_title(title, fontsize=9)

    for r, (label, img, layers, valid) in enumerate(rows):
        ax0 = axes[r, 0]
        ax0.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=40)
        if img is None:
            ax0.text(0.5, 0.5, "raw n/a", ha="center", va="center", fontsize=8)
        elif img.ndim == 2:
            ax0.imshow(img, cmap=("tab20" if label.startswith("LULC") else "gray"))
        else:
            ax0.imshow(img)
        for c, layer in enumerate(LAYERS, start=1):
            ax = axes[r, c]
            tok = layers.get(layer)
            if tok is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8, color="grey")
            else:
                rgb, vr = pca_rgb(tok, valid)
                ax.imshow(rgb)
                met = panel_metrics(tok, valid)
                ax.set_xlabel(f"cos={met['offdiag_cos']:.2f} ac={met['neighbor_ac']:.2f}\n"
                              f"pca={vr[0]:.2f}/{vr[1]:.2f}/{vr[2]:.2f}", fontsize=5.5)
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"embed_{name}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[render] wrote {out}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stations", type=int, default=4)
    ap.add_argument("--stations", type=str, default=None,
                    help="Comma-separated explicit station dir names (skip auto-selection)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Phase 1 selection only — print picks, do not render")
    args = ap.parse_args()

    df = pd.read_csv(SPLITS).dropna(subset=["source_network", "network", "station_id"])
    disk = build_disk_index()
    print(f"stores on disk: {len(disk)}")

    if args.stations:
        want = set(args.stations.split(","))
        df["dir"] = df.apply(station_dir_name, axis=1)
        picked = []
        for _, row in df[df["dir"].isin(want)].iterrows():
            name = row["dir"]
            if name in disk:
                info = analyse_station(name, disk[name], row)
                if info:
                    info["has_raw"] = open_raw(name) is not None
                    picked.append(info)
    else:
        picked = select_stations(df, disk, args.n_stations)

    print_selection(picked)
    if args.dry_run:
        print("\n[dry-run] selection complete; no figures rendered.")
        return
    for info in picked:
        if info["n_seasons"] < 4:
            print(f"[skip] {info['name']} only {info['n_seasons']} seasons")
            continue
        render_station(info)


if __name__ == "__main__":
    main()
