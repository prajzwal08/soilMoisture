"""
analyze_nan_filtering.py
========================
Apply TerraMesh two-tier NaN filtering to all raw satellite tiles and report
how many images / tokens survive per modality.

Tier-1 (tile level):  > 50 % NaN pixels across all bands → discard whole tile
Tier-2 (token level): > 1 %  NaN pixels inside a 16×16 patch → invalid token

Modalities analysed:
  S2L2A  — 12-band, one TIF per date   (e.g. 20170105.tif)
  S1RTC  — 2-band,  one TIF per date   (e.g. 20170103_ASC.tif)
  DEM    — 1-band,  single file        (dem.tif)

For cross-modality, the DEM NaN map is the same for every date at a station.
A "combined" analysis is therefore:
  invalid_pixel(date) = nan(S2L2A, date) | nan(S1RTC, nearest) | nan(DEM)

The cross-modality section uses S2L2A dates as the reference and looks up the
station-level DEM NaN map (S1RTC cross-modality would require date alignment so
is reported independently).

Output
------
  data/logs/nan_filtering_analysis.png  — three summary panels
  data/logs/nan_filtering_analysis.svg
  Console summary table

Usage
-----
  python analyze_nan_filtering.py                     # all stations
  python analyze_nan_filtering.py --workers 8
  python analyze_nan_filtering.py --n-stations 10    # quick smoke-test
"""

from __future__ import annotations

import argparse
import os
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── GDAL performance env vars ─────────────────────────────────────────────────
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE",                    "TRUE")
os.environ.setdefault("GDAL_MAX_RAW_BLOCK_CACHE_SIZE","200000000")

# ── paths ─────────────────────────────────────────────────────────────────────
SCRATCH_DIR = Path("/gpfs/scratch1/shared/pkhanal/satellite")
LOGS_DIR    = Path("/gpfs/work3/0/prjs1968/data/logs")

TILE_NAN_THRESHOLD  = 0.50   # >50 % → discard tile
PATCH_NAN_THRESHOLD = 0.01   # >1  % → invalid token

MODALITIES = ["S2L2A", "S1RTC", "DEM"]


# ── low-level helpers ─────────────────────────────────────────────────────────

def _nan_map(tif_path: Path, zero_is_nodata: bool = False) -> np.ndarray | None:
    """
    Return a (H, W) bool array: True = pixel invalid.
    zero_is_nodata=True (S2L2A): pixel invalid when ALL bands == 0 (fill value).
    Otherwise: pixel invalid when ANY band is NaN.
    Returns None on read error.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(tif_path) as src:
                arr = src.read().astype(np.float32)   # (C, H, W)
                nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        invalid = np.isnan(arr).any(axis=0)           # (H, W) bool
        if zero_is_nodata:
            invalid |= (arr == 0.0).all(axis=0)       # fill pixel: all bands zero
        return invalid
    except Exception:
        return None


def _token_stats(nan_px: np.ndarray) -> tuple[int, int]:
    """
    Given a (224, 224) bool NaN map, return (n_valid_tokens, n_total_tokens).
    A token (16×16 patch) is valid if its NaN fraction < PATCH_NAN_THRESHOLD.
    """
    nm = nan_px[:224, :224]
    if nm.shape[0] < 224 or nm.shape[1] < 224:
        pad = np.ones((224, 224), dtype=bool)
        pad[:nm.shape[0], :nm.shape[1]] = nm
        nm = pad
    patch_frac = nm.reshape(14, 16, 14, 16).mean(axis=(1, 3))  # (14, 14)
    valid = (patch_frac < PATCH_NAN_THRESHOLD).sum()
    return int(valid), 196


# ── per-station worker ────────────────────────────────────────────────────────

def _analyze_station(station_dir: Path) -> dict | None:
    """
    Returns a dict with per-modality tile/token stats for one station.
    Keys:
      S2L2A_tiles, S2L2A_t1_pass, S2L2A_tokens_total, S2L2A_tokens_valid
      S1RTC_*  (same pattern)
      DEM_tiles, DEM_t1_pass, DEM_tokens_total, DEM_tokens_valid
      combined_tokens_total, combined_tokens_valid   (S2L2A masked by DEM)
      station
    """
    result: dict = {"station": station_dir.name}

    # ── DEM (single static file) ──────────────────────────────────────────────
    dem_nan: np.ndarray | None = None
    dem_dir = station_dir / "DEM"
    if (dem_dir / "dem.tif").exists():
        dem_nan = _nan_map(dem_dir / "dem.tif")

    result["DEM"] = _analyze_dem(dem_nan)

    # ── S2L2A (one TIF per date) ──────────────────────────────────────────────
    s2_dir = station_dir / "S2L2A"
    s2_months: list[int]       = []
    s2_tiles                   = 0
    s2_t1_pass                 = 0
    s2_tok_total               = 0
    s2_tok_valid               = 0
    comb_tok_total             = 0
    comb_tok_valid             = 0

    if s2_dir.exists():
        for tif in sorted(s2_dir.glob("*.tif")):
            s2_tiles += 1
            try:
                month = int(tif.stem[4:6])
            except (ValueError, IndexError):
                month = 0
            nm = _nan_map(tif, zero_is_nodata=True)
            if nm is None:
                continue
            if nm.mean() > TILE_NAN_THRESHOLD:
                continue
            s2_t1_pass += 1
            s2_months.append(month)
            v, t = _token_stats(nm)
            s2_tok_total += t
            s2_tok_valid += v

            # combined: S2L2A NaN  OR  DEM NaN
            if dem_nan is not None:
                comb_nm = nm | dem_nan[:nm.shape[0], :nm.shape[1]]
                if comb_nm.mean() <= TILE_NAN_THRESHOLD:
                    cv, ct = _token_stats(comb_nm)
                    comb_tok_total += ct
                    comb_tok_valid += cv
            else:
                comb_tok_total += t
                comb_tok_valid += v

    result["S2L2A"] = (s2_tiles, s2_t1_pass, s2_tok_total, s2_tok_valid)
    result["S2L2A_months"] = s2_months
    result["combined"] = (comb_tok_total, comb_tok_valid)

    # ── S1RTC (one TIF per date) ──────────────────────────────────────────────
    s1_dir = station_dir / "S1RTC"
    s1_tiles   = 0
    s1_t1_pass = 0
    s1_tok_total = 0
    s1_tok_valid = 0

    if s1_dir.exists():
        for tif in sorted(s1_dir.glob("*.tif")):
            s1_tiles += 1
            nm = _nan_map(tif)
            if nm is None:
                continue
            if nm.mean() > TILE_NAN_THRESHOLD:
                continue
            s1_t1_pass += 1
            v, t = _token_stats(nm)
            s1_tok_total += t
            s1_tok_valid += v

    result["S1RTC"] = (s1_tiles, s1_t1_pass, s1_tok_total, s1_tok_valid)

    return result


def _analyze_dem(dem_nan: np.ndarray | None) -> tuple[int, int, int, int]:
    """Return (tiles, t1_pass, tok_total, tok_valid) for the station DEM."""
    if dem_nan is None:
        return (0, 0, 0, 0)
    t1_pass   = 1 if dem_nan.mean() <= TILE_NAN_THRESHOLD else 0
    if t1_pass:
        tok_valid, tok_total = _token_stats(dem_nan)
    else:
        tok_valid, tok_total = 0, 0
    return (1, t1_pass, tok_total, tok_valid)


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    agg: dict = {}
    for mod in MODALITIES:
        tiles = t1 = tok_t = tok_v = 0
        for r in results:
            if mod not in r:
                continue
            vals = r[mod]
            if mod == "DEM":
                tiles_r, t1_r, tok_t_r, tok_v_r = vals
            else:
                tiles_r, t1_r, tok_t_r, tok_v_r = vals
            tiles += tiles_r
            t1    += t1_r
            tok_t += tok_t_r
            tok_v += tok_v_r
        agg[mod] = {"tiles": tiles, "t1_pass": t1,
                    "tok_total": tok_t, "tok_valid": tok_v}

    # combined
    ctt = cvv = 0
    for r in results:
        ct, cv = r.get("combined", (0, 0))
        ctt += ct
        cvv += cv
    agg["combined"] = {"tok_total": ctt, "tok_valid": cvv}

    # monthly breakdown for S2L2A (only from t1-passing tiles)
    by_month: dict[int, list[int]] = defaultdict(list)
    for r in results:
        for m in r.get("S2L2A_months", []):
            by_month[m].append(1)
    agg["S2L2A_by_month"] = {m: len(v) for m, v in by_month.items()}

    return agg


# ── printing ──────────────────────────────────────────────────────────────────

MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]


def print_summary(agg: dict, n_stations: int):
    print(f"\n{'='*65}")
    print("NaN FILTERING ANALYSIS  (TerraMesh two-tier filter)")
    print(f"{'='*65}")
    print(f"  Stations analysed : {n_stations:,}")
    print()

    hdr = f"  {'Modality':<10}  {'Tiles':>8}  {'T1 pass':>9}  {'T1 rate':>8}  "
    hdr += f"{'Tokens':>10}  {'Valid tok':>10}  {'Tok rate':>8}"
    print(hdr)
    print(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")

    for mod in MODALITIES:
        d = agg[mod]
        t1r  = d["t1_pass"] / d["tiles"] if d["tiles"] else 0
        tokr = d["tok_valid"] / d["tok_total"] if d["tok_total"] else 0
        print(f"  {mod:<10}  {d['tiles']:>8,}  {d['t1_pass']:>9,}  "
              f"{t1r:>7.1%}  {d['tok_total']:>10,}  "
              f"{d['tok_valid']:>10,}  {tokr:>7.1%}")

    # combined row
    c = agg["combined"]
    if c["tok_total"]:
        cr = c["tok_valid"] / c["tok_total"]
        print(f"\n  {'S2+DEM':<10}  {'':>8}  {'':>9}  {'':>8}  "
              f"{c['tok_total']:>10,}  {c['tok_valid']:>10,}  {cr:>7.1%}")
        print(f"  (S2L2A tokens further masked by DEM NaN)")

    print(f"\n{'='*65}\n")


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_summary(agg: dict, n_stations: int, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"NaN filtering analysis — {n_stations:,} stations", fontsize=11)

    # ── Panel 1: tile survival at tier-1 per modality ────────────────────────
    ax = axes[0]
    mods  = [m for m in MODALITIES if agg[m]["tiles"] > 0]
    rates = [agg[m]["t1_pass"] / agg[m]["tiles"] * 100 for m in mods]
    colors = ["#1e88e5", "#fb8c00", "#43a047"][:len(mods)]
    bars = ax.bar(mods, rates, color=colors, edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% tiles surviving tier-1 (≤50% NaN)")
    ax.set_title("Tier-1: tile survival by modality")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    # ── Panel 2: token survival at tier-2 per modality ───────────────────────
    ax = axes[1]
    mods2  = [m for m in MODALITIES if agg[m]["tok_total"] > 0]
    trates = [agg[m]["tok_valid"] / agg[m]["tok_total"] * 100 for m in mods2]

    # also add combined if it has data
    if agg["combined"]["tok_total"] > 0:
        mods2.append("S2+DEM")
        trates.append(agg["combined"]["tok_valid"] / agg["combined"]["tok_total"] * 100)

    colors2 = ["#1e88e5", "#fb8c00", "#43a047", "#e53935"][:len(mods2)]
    bars2 = ax.bar(mods2, trates, color=colors2, edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars2, trates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% tokens valid (≤1% NaN per 16×16 patch)")
    ax.set_title("Tier-2: token survival (of tier-1 tiles)")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    # ── Panel 3: S2L2A monthly tile count (tier-1 passing) ───────────────────
    ax = axes[2]
    by_month = agg["S2L2A_by_month"]
    months_present = sorted(by_month)
    counts = [by_month[m] for m in months_present]
    x = np.arange(len(months_present))
    ax.bar(x, counts, color="#1e88e5", edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([MONTH_ABBR[m - 1] for m in months_present], fontsize=8)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Tiles passing tier-1")
    ax.set_title("S2L2A: monthly distribution of valid tiles")

    plt.tight_layout()
    stem = out_dir / "nan_filtering_analysis"
    fig.savefig(stem.with_suffix(".png"), dpi=130, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {stem}.png / .svg")


# ── per-station table ─────────────────────────────────────────────────────────

def print_station_table(results: list[dict], top_n: int = 20):
    """Print the top_n worst stations (most tiles discarded at tier-1)."""
    rows = []
    for r in results:
        s2 = r.get("S2L2A", (0, 0, 0, 0))
        s1 = r.get("S1RTC", (0, 0, 0, 0))
        dem = r.get("DEM",  (0, 0, 0, 0))
        rows.append((r["station"], s2, s1, dem))

    # Sort by S2L2A t1_pass fraction (ascending = worst first)
    rows.sort(key=lambda x: x[1][1] / max(x[1][0], 1))

    print(f"  {'Station':<45} {'S2 tiles':>8} {'S2 pass':>8}"
          f" {'S1 pass':>8} {'DEM ok':>6}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for stn, s2, s1, dem in rows[:top_n]:
        s2_rate = s2[1] / max(s2[0], 1)
        s1_rate = s1[1] / max(s1[0], 1)
        dem_ok  = "yes" if dem[1] == dem[0] and dem[0] > 0 else ("no " if dem[0] > 0 else "n/a")
        print(f"  {stn:<45} {s2[0]:>8,} {s2_rate:>7.0%}"
              f" {s1_rate:>7.0%}  {dem_ok:>5}")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply TerraMesh NaN filters and report tile/token survival")
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR)
    parser.add_argument("--out-dir",     type=Path, default=LOGS_DIR)
    parser.add_argument("--workers",     type=int,  default=4,
                        help="Parallel worker processes")
    parser.add_argument("--n-stations",  type=int,  default=None,
                        help="Limit to first N stations (for testing)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stations = sorted(args.scratch_dir.iterdir())
    stations = [s for s in stations if s.is_dir()]
    if args.n_stations:
        stations = stations[:args.n_stations]

    print(f"Analysing {len(stations)} station(s) with {args.workers} worker(s)…")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_analyze_station, s): s.name for s in stations}
        done = 0
        for fut in as_completed(futures):
            done += 1
            name = futures[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as exc:
                print(f"  [error] {name}: {exc}")
            if done % 50 == 0 or done == len(stations):
                print(f"  {done}/{len(stations)} stations processed…")

    if not results:
        print("No results. Check scratch path.")
        return

    agg = aggregate(results)
    print_summary(agg, len(results))
    print_station_table(results)
    plot_summary(agg, len(results), args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
