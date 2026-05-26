"""
Satellite patch visualisation — 4 geographically diverse stations.

Columns:
  1. S2 RGB  (summer 2018)
  2. S2 RGB  (summer, other year)
  3. S1 false-colour composite  (R=VV, G=VH, B=VV-VH) — closest date to col-1
  4. DEM

S1 false-colour legend (approximate):
  Orange/red  -> high VV, low VH  -> smooth/urban surface
  Green       -> high VH          -> vegetation / volume scattering
  Dark        -> low both         -> open water / shadow

Usage:
  python plot_satellite_patches.py

Saves to:  plots/satellite_patches.png  (300 dpi)
"""

import re, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import scienceplots   # noqa
from pathlib import Path

plt.style.use(["science", "no-latex"])

# ── Paths ────────────────────────────────────────────────────────────────────
MPC_DIR   = Path("/home/khanalp/data/satellite")
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Station catalogue  (name -> country, land-cover type) ────────────────────
STATIONS = {
    "COSMOS-UK_BickleyHall":  ("United Kingdom",  "Temperate grassland"),
    "HOAL_Hoal-12":           ("Austria",          "Mixed cropland"),
    "NGARI_SQ06":             ("China (Tibet)",    "Alpine meadow / steppe"),
    "REMEDHUS_LasBrozas":     ("Spain",            "Mediterranean dryland cropland"),
}

# S2L1C band indices (0-based) stored in MPC files
# Band order: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12
BLUE, GREEN, RED = 1, 2, 3

# ── I/O helpers ──────────────────────────────────────────────────────────────
def read_band(path: Path, idx: int) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(idx + 1).astype(np.float32)


def summer_files(folder: Path, year: str | None = None) -> list[Path]:
    """Return S2 GeoTIFFs from June/July/August, optionally filtered by year."""
    files = []
    for f in sorted(folder.glob("*.tif")):
        m = re.search(r"(\d{4})(\d{2})\d{2}", f.name)
        if m and m.group(2) in ("06", "07", "08"):
            if year is None or m.group(1) == year:
                files.append(f)
    return files


def clearest(files: list[Path]) -> Path:
    """
    Return the least-cloudy, most-complete S2 file.
    Rejects images with >30% NaN (tile-boundary gaps) or >50% saturated pixels
    (cloud). Among remaining, picks the one with the lowest mean reflectance
    (dark land is cloud-free; cloud is bright).
    """
    scored = []
    for f in files:
        with rasterio.open(f) as src:
            data = src.read([RED + 1, GREEN + 1, BLUE + 1]).astype(np.float32)
        if float(np.mean(np.isnan(data))) > 0.30:
            continue
        if float(np.nanmean(data > 8500)) > 0.50:
            continue
        scored.append((float(np.nanmean(data)), f))
    if not scored:
        # fallback: pick image with fewest NaNs
        for f in files:
            with rasterio.open(f) as src:
                data = src.read([RED + 1, GREEN + 1, BLUE + 1]).astype(np.float32)
            scored.append((float(np.mean(np.isnan(data))), f))
    return min(scored, key=lambda x: x[0])[1]


def closest_s1(s1_dir: Path, target_date: str) -> Path | None:
    """
    Return the S1 ascending-orbit file closest in acquisition date to
    target_date (YYYYMMDD). Falls back to any orbit if no ASC files exist.
    """
    candidates = list(s1_dir.glob("*_ASC.tif")) or list(s1_dir.glob("*.tif"))
    if not candidates:
        return None
    td = int(target_date)
    return min(candidates,
               key=lambda f: abs(int(re.search(r"\d{8}", f.name).group()) - td))


# ── Visualisation helpers ─────────────────────────────────────────────────────
def pstretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """
    Percentile-clip stretch to [0, 1]. NaN pixels are set to 1.0 (white).
    Applied per channel so each band uses its own dynamic range.
    """
    vlo, vhi = np.nanpercentile(arr, lo), np.nanpercentile(arr, hi)
    if vhi == vlo:
        return np.ones_like(arr)
    out = np.clip((arr - vlo) / (vhi - vlo), 0.0, 1.0)
    return np.where(np.isnan(arr), 1.0, out)


def rgb_image(path: Path) -> np.ndarray:
    """Natural-colour RGB (B04-B03-B02) with per-channel percentile stretch."""
    return np.stack([
        pstretch(read_band(path, RED)),
        pstretch(read_band(path, GREEN)),
        pstretch(read_band(path, BLUE)),
    ], axis=-1)


def s1_false_color(path: Path) -> np.ndarray:
    """
    Sentinel-1 false-colour composite (per-channel percentile stretch):
      R = VV  (dB)
      G = VH  (dB)
      B = VV - VH  (dB cross-polarisation ratio)

    Colour interpretation:
      Orange/red   -> high VV, low VH     -> smooth or urban surfaces
      Green/yellow -> high VH relative VV -> vegetation / volume scattering
      Dark         -> low VV and VH       -> open water or radar shadow
    """
    with rasterio.open(path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
    return np.stack([pstretch(vv), pstretch(vh), pstretch(vv - vh)], axis=-1)


def read_dem(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def date_fmt(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


# ── Gather per-station data ───────────────────────────────────────────────────
rows = []
for station, (country, lc) in STATIONS.items():
    sd  = MPC_DIR / station
    s2d = sd / "S2L1C"
    s1d = sd / "S1RTC"
    dem = sd / "DEM" / "dem.tif"

    files18 = summer_files(s2d, "2018")
    if not files18:
        print(f"  {station}: no summer 2018 S2 -- skipping")
        continue
    f18 = clearest(files18)

    other = [f for f in summer_files(s2d) if "2018" not in f.name]
    if not other:
        print(f"  {station}: no other-year summer S2 -- skipping")
        continue
    f_other = clearest(other)

    date18 = re.search(r"\d{8}", f18.name).group()
    s1f    = closest_s1(s1d, date18)
    if s1f is None or not dem.exists():
        print(f"  {station}: missing S1 or DEM -- skipping")
        continue

    meta  = json.loads((sd / "metadata.json").read_text())
    s1_date  = re.search(r"\d{8}", s1f.name).group()
    d_other  = re.search(r"\d{8}", f_other.name).group()
    gap      = abs(int(date18) - int(s1_date))

    rows.append(dict(
        station=station, country=country, lc=lc,
        lat=meta["latitude"], lon=meta["longitude"],
        f18=f18, f_other=f_other, s1f=s1f, dem=dem,
        date18=date18, date_other=d_other, s1_date=s1_date, gap=gap,
    ))
    gap_str = f"{gap//10000}y {(gap % 10000)//100}m" if gap > 100 else "same day"
    print(f"  {station}: S2={date18}  S2-other={d_other}  S1={s1_date} ({gap_str})")

# ── Figure ────────────────────────────────────────────────────────────────────
N_ROWS, N_COLS = len(rows), 4

fig, axes = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(N_COLS * 2.8, N_ROWS * 3.1),
    gridspec_kw={"hspace": 0.45, "wspace": 0.06},
)

for col, title in enumerate([
    "S2 RGB\n(summer 2018)",
    "S2 RGB\n(summer, other year)",
    "S1 false colour\nR=VV  G=VH  B=VV-VH",
    "DEM  [m]",
]):
    axes[0, col].set_title(title, fontsize=8, pad=5)


def stamp(ax, txt: str) -> None:
    ax.text(0.03, 0.03, txt, transform=ax.transAxes,
            fontsize=5.5, color="white", va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", fc="black", alpha=0.55, lw=0))


for ri, d in enumerate(rows):
    axs = axes[ri]

    # Row label
    axs[0].set_ylabel(
        f"{d['country']}\n{d['lc']}\n({d['lat']:.1f}N, {d['lon']:.1f}E)",
        fontsize=6.5, labelpad=5, linespacing=1.5,
    )

    # Col 0 — S2 RGB 2018
    axs[0].imshow(rgb_image(d["f18"]), interpolation="nearest")
    stamp(axs[0], date_fmt(d["date18"]))

    # Col 1 — S2 RGB other year
    axs[1].imshow(rgb_image(d["f_other"]), interpolation="nearest")
    stamp(axs[1], date_fmt(d["date_other"]))

    # Col 2 — S1 false colour
    axs[2].imshow(s1_false_color(d["s1f"]), interpolation="nearest")
    gap_str = (f"{d['gap']//10000}y {(d['gap'] % 10000)//100}m from S2"
               if d["gap"] > 100 else "same day as S2")
    stamp(axs[2], f"{date_fmt(d['s1_date'])}  ({gap_str})")

    # Col 3 — DEM
    im = axs[3].imshow(read_dem(d["dem"]), cmap="terrain", interpolation="nearest")
    cb = fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("m", fontsize=5.5)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)

# S1 colour legend
fig.legend(
    handles=[
        mpatches.Patch(color=(0.9, 0.5, 0.1),
                       label="Orange/red: smooth / urban (high VV)"),
        mpatches.Patch(color=(0.3, 0.7, 0.2),
                       label="Green: vegetation (high VH)"),
        mpatches.Patch(color=(0.1, 0.1, 0.1),
                       label="Dark: open water (low VV & VH)"),
    ],
    loc="lower center", ncol=3, fontsize=6, frameon=True,
    bbox_to_anchor=(0.5, -0.01), handlelength=1.2, handleheight=0.9,
)

fig.suptitle(
    "Satellite patches across 4 stations\n"
    "S2 natural colour  |  S1 false colour (R=VV  G=VH  B=VV-VH)  |  DEM",
    fontsize=9, y=1.01,
)

out = PLOTS_DIR / "satellite_patches.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved -> {out}  (300 dpi)")
plt.close()
