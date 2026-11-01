#!/usr/bin/env python
"""
§36.22 -- visualise the ECOSTRESS census filter chain.

Reads the three merged census CSVs and draws, for a handful of contrasting stations,
what every filter in `census_ecostress.py` actually does to every overpass.

Reads only CSVs.  No network, no EDL, no rasterio, no torch.

NEVER run this on a login node.  Always:
    sbatch slurm/ecostress_census_analyze.sh

Two facts the figures exist to make visible, because neither is legible from the CSVs:

  1. The chain FORKS.  Stage 5 splits day from night; they run stages 6-11 independently
     and rejoin only at stage 12.  `pair_station` is handed `inwin` (phase-filtered) and
     never consults `passed_qc` -- `quality` is attached AFTERWARDS as a label.  So
     "1717 passed QC -> 216 pairs" is not a subtraction; those numbers sit on different
     branches.  Any single-column funnel drawing of this census is wrong.

  2. Pair yield is capped by ORBITAL DATE COINCIDENCE, not by cloud.  At BodieHills only
     184 of 1083 solar dates carry both a day and a night pass.  715 of 931 day passes
     find no partner -- and `census_ecostress.py:625` drops them with a bare `continue`,
     recording neither a row nor a counter, so that number can only be derived.

Runbook §36.22.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

ROOT     = Path("/gpfs/work3/0/prjs1968/soilMoisture")
CENSUS   = ROOT / "census_ecostress.py"
GRAN_CSV = ROOT / "csvs" / "ecostress_census_granules.csv"
PAIR_CSV = ROOT / "csvs" / "ecostress_census_pairs.csv"
LOG_CSV  = ROOT / "csvs" / "ecostress_census_log.csv"

DEFAULT_STATIONS = ["BodieHills", "Rothamsted", "PSA2Tiergarten", "Banizoumbou"]

# ============================================================
# PALETTE
# ============================================================
# Validated categorical slots.  Only TWO hues carry identity (day / night); everything
# rejected is neutral, so the scatter panels stay inside the all-pairs CVD gate.
C_DAY       = "#eb6834"   # slot 2, orange
C_NIGHT     = "#2a78d6"   # slot 1, blue
C_CROSS     = "#9a9890"   # neutral -- the crossover kill band
C_REJECT    = "#e34948"   # status: critical
C_OK        = "#1baf7a"   # slot 3, aqua -- survivors
C_DERIVED   = "#4a3aa7"   # slot 7, violet -- quantities we had to derive
INK         = "#0b0b0b"
INK2        = "#52514e"
INK3        = "#8a8880"
GRID        = "#e4e3de"
SURFACE     = "#fcfcfb"
# Ordinal ramp for the funnel, blue, no lighter than step 250 on a light surface.
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]

PHASE_COLOR = {"day": C_DAY, "night": C_NIGHT, "crossover": C_CROSS, "off": INK3}


FORMATS = ("png", "pdf")


def style(dpi: int = 300):
    """Publication styling: scienceplots `science` + `no-latex`, then our overrides.

    `no-latex` is mandatory, not a fallback -- there is no LaTeX install to render
    against, and the `science` style enables usetex by default.
    """
    try:
        import scienceplots                                   # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception as exc:                                  # noqa: BLE001
        print(f"  (scienceplots unavailable -- {exc}; falling back to default style)")
        plt.style.use("default")
    plt.rcParams.update({
        # a serif stack that actually carries the glyphs used here (−, ≥, →, ·, —)
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Nimbus Roman", "Times New Roman"],
        "mathtext.fontset":  "dejavuserif",
        "axes.unicode_minus": True,
        "figure.facecolor":  SURFACE,
        "axes.facecolor":    SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor":    INK3,
        "axes.linewidth":    0.7,
        "axes.labelcolor":   INK2,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   INK,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.color":        GRID,
        "grid.linewidth":    0.5,
        "xtick.color":       INK2,
        "ytick.color":       INK2,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "font.size":         9,
        "legend.frameon":    False,
        "legend.fontsize":   8,
        "figure.dpi":        dpi,
        "savefig.dpi":       dpi,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype":      42,      # embed TrueType, so text stays selectable
        "ps.fonttype":       42,
    })


def save(fig, outdir: Path, name: str):
    """Write every configured format, then close.  PDF is the one that goes in a thesis."""
    for ext in FORMATS:
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)


# ============================================================
# CONSTANTS, READ FROM census_ecostress.py -- NEVER RETYPED
# ============================================================

_SAFE_CALLS = {"int": int, "round": round, "abs": abs, "float": float, "len": len}
_SAFE_BINOPS = {
    ast.Add: lambda a, b: a + b,     ast.Sub:  lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,    ast.Div:  lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}


def _resolve(node: ast.AST, env: dict):
    """Evaluate a constant expression against constants already parsed.

    Needed because not every constant is a literal -- e.g. :95 defines
        N_PX_EXPECTED = int(round(TILE_M / PIXEL_M)) ** 2
    Resolving it rather than retyping `1024` is what keeps this file honest if the
    station window or the ECOSTRESS grid spacing ever changes.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        pass
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"unresolved name {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_resolve(node.left, env),
                                           _resolve(node.right, env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_resolve(node.operand, env)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _SAFE_CALLS and not node.keywords):
        return _SAFE_CALLS[node.func.id](*[_resolve(a, env) for a in node.args])
    raise ValueError(f"unsupported expression node {type(node).__name__}")


def load_constants() -> dict:
    """AST-parse the census module for its module-level constants.

    Parsed rather than imported on purpose: no side effects, and no dependency on
    `requests`/`rasterio` being present in whatever env runs the plots.  The values
    therefore cannot drift from the code that produced the CSVs.
    """
    tree = ast.parse(CENSUS.read_text())
    out: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                try:
                    out[tgt.id] = _resolve(node.value, out)
                except (ValueError, SyntaxError, TypeError, ZeroDivisionError):
                    pass
            elif isinstance(tgt, ast.Tuple):          # DAY_LO, DAY_HI = 0.5, 3.5
                try:
                    vals = _resolve(node.value, out)
                except (ValueError, SyntaxError, TypeError, ZeroDivisionError):
                    continue
                for el, v in zip(tgt.elts, vals):
                    if isinstance(el, ast.Name):
                        out[el.id] = v
    required = ["CROSSOVER_LO", "CROSSOVER_HI", "CLEAR_FRAC_MIN", "WINDOW_FRAC_MIN",
                "N_PX_EXPECTED", "LAT_LIMIT", "LAT_NOMINAL", "THERMAL_PEAK_LAG_H",
                "WELL_PHASED_DAY_H", "WELL_PHASED_NIGHT_TST", "MISSION_START",
                "CONCEPT_ID", "TILE_M", "PIXEL_M"]
    missing = [k for k in required if k not in out]
    if missing:
        sys.exit(f"FATAL: could not parse constants from {CENSUS}: {missing}")
    return out


# ============================================================
# SOLAR GEOMETRY -- OVERLAY CURVES ONLY
# ============================================================
# Every filter value plotted comes from the CSV, i.e. from the census's own geometry.
# These helpers draw the *context* curves (sunrise/sunset, the crossover contours) and
# are never used to decide whether a granule passed anything.

def declination_deg(doy: np.ndarray) -> np.ndarray:
    """Solar declination, Cooper's approximation.  Good to ~0.5 deg; context only."""
    return 23.44 * np.sin(np.radians(360.0 / 365.0 * (doy - 81.0)))


def tst_at_elevation(doy: np.ndarray, lat: float, elev_deg: float):
    """The two true-solar-times each day at which the sun sits at `elev_deg`.

    Returns (morning_tst, evening_tst); NaN on days the elevation is never reached.
    """
    dec = np.radians(declination_deg(doy))
    phi = np.radians(lat)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosH = (np.sin(np.radians(elev_deg)) - np.sin(phi) * np.sin(dec)) / (
            np.cos(phi) * np.cos(dec))
    cosH = np.where(np.abs(cosH) <= 1.0, cosH, np.nan)
    H = np.degrees(np.arccos(cosH))          # hour angle, degrees
    return 12.0 - H / 15.0, 12.0 + H / 15.0


def max_elevation(doy: np.ndarray, lat: float) -> np.ndarray:
    """Solar elevation at local solar noon."""
    return 90.0 - np.abs(lat - declination_deg(doy))


# ============================================================
# DATA
# ============================================================

def load_data(stations: list[str]):
    gran = pd.read_csv(GRAN_CSV, low_memory=False)
    pair = pd.read_csv(PAIR_CSV, low_memory=False)
    logd = pd.read_csv(LOG_CSV, low_memory=False)

    missing = [s for s in stations if s not in set(logd["station_id"])]
    if missing:
        sys.exit(f"FATAL: stations absent from the census log: {missing}")

    gran["utc_ts"] = pd.to_datetime(gran["utc"], errors="coerce", utc=True)
    gran["solar_date"] = pd.to_datetime(gran["solar_date_str"], errors="coerce")
    for c in ("tst", "hours_from_solar_noon", "solar_elev", "clear_frac", "valid_frac",
              "window_frac", "vza_mean_abs", "vza_max_abs", "frac_mand00", "frac_mand01",
              "frac_lstacc_ge2", "frac_water", "frac_cloud"):
        gran[c] = pd.to_numeric(gran[c], errors="coerce")
    for c in ("read_ok", "passed_qc", "clipped", "n_px"):
        gran[c] = pd.to_numeric(gran[c], errors="coerce")

    pair["day_ts"] = pd.to_datetime(pair["day_utc"], errors="coerce", utc=True)
    for c in ("dt_hours", "day_tst", "night_tst", "day_elev", "night_elev", "elev_drop",
              "day_clear", "night_clear"):
        pair[c] = pd.to_numeric(pair[c], errors="coerce")
    for c in ("well_phased", "quality"):
        pair[c] = pd.to_numeric(pair[c], errors="coerce").fillna(0).astype(int)

    return gran, pair, logd


def funnel(sid: str, gran: pd.DataFrame, pair: pd.DataFrame, logd: pd.DataFrame) -> dict:
    """Per-station stage counts, reconciled against the census's own log row.

    Every count that CAN be recomputed from the granule/pair CSVs IS recomputed and
    asserted equal to the log.  A silently wrong waterfall is worse than no waterfall.
    """
    g = gran[gran.station_id == sid]
    p = pair[pair.station_id == sid]
    L = logd[logd.station_id == sid].iloc[0]

    n_cross = int((g.phase == "crossover").sum())
    n_day   = int((g.phase == "day").sum())
    n_night = int((g.phase == "night").sum())
    f = {
        "station_id":   sid,
        "lat":          float(g.lat.iloc[0]) if len(g) else float("nan"),
        "lon":          float(g.lon.iloc[0]) if len(g) else float("nan"),
        "network":      str(g.network.iloc[0]) if len(g) else "",
        "elevation_m":  float(g.elevation_m.iloc[0]) if len(g) else float("nan"),
        "kg_macro":     str(g.kg_macro.iloc[0]) if len(g) else "",
        "n_hits":       int(L.n_hits),
        "n_dupe_reproc": int(L.n_dupe_reproc),
        "n_dupe_orbit": int(L.n_dupe_orbit),
        "n_overpasses": int(L.n_overpasses),
        "n_crossover":  n_cross,
        "n_inwindow":   int(L.n_inwindow),
        "n_day":        n_day,
        "n_night":      n_night,
        "n_read_ok":    int(L.n_read_ok),
        "n_passed":     int(L.n_passed),
        "n_pairs":      int(len(p)),
        "n_pairs_quality":     int(L.n_pairs_quality),
        "n_pairs_well_phased": int(L.n_pairs_well_phased),
    }
    # day/night lanes, independently
    for lane in ("day", "night"):
        sub = g[g.phase == lane]
        f[f"n_{lane}_read_ok"] = int((sub.read_ok == 1).sum())
        f[f"n_{lane}_passed"]  = int((sub.passed_qc == 1).sum())
    # derived: the pairing stage records nothing about what it dropped
    f["n_day_unpaired"]   = f["n_day"] - f["n_pairs"]
    f["n_night_unpaired"] = f["n_night"] - f["n_pairs"]

    checks = [
        ("hits - reproc - orbit == overpasses",
         f["n_hits"] - f["n_dupe_reproc"] - f["n_dupe_orbit"], f["n_overpasses"]),
        ("rows in granule CSV == overpasses", len(g), f["n_overpasses"]),
        ("overpasses - crossover == inwindow",
         f["n_overpasses"] - n_cross, f["n_inwindow"]),
        ("day + night == inwindow", n_day + n_night, f["n_inwindow"]),
        ("count(read_ok==1) == n_read_ok", int((g.read_ok == 1).sum()), f["n_read_ok"]),
        ("count(passed_qc==1) == n_passed",
         int((g.passed_qc == 1).sum()), f["n_passed"]),
        ("count(quality==1) == n_pairs_quality",
         int((p.quality == 1).sum()), f["n_pairs_quality"]),
        ("count(quality & well_phased) == n_pairs_well_phased",
         int(((p.quality == 1) & (p.well_phased == 1)).sum()),
         f["n_pairs_well_phased"]),
    ]
    bad = [(w, a, b) for w, a, b in checks if a != b]
    if bad:
        for w, a, b in bad:
            print(f"  RECONCILE FAIL [{sid}] {w}: computed {a} != logged {b}")
        sys.exit(f"FATAL: funnel does not reconcile for {sid}; refusing to plot it.")

    # The ordering assertion.  If a refactor ever makes pairing QC-gated, this catches it:
    # pairs can only be drawn from the phase-filtered set, never from the QC-passed set.
    if f["n_pairs"] > min(f["n_day"], f["n_night"]):
        sys.exit(f"FATAL: {sid} has more pairs than day/night passes -- pairing is not "
                 f"one-to-one over `inwin`.")
    print(f"  [{sid}] reconciled: {f['n_hits']} -> {f['n_overpasses']} -> "
          f"{f['n_inwindow']} -> {f['n_read_ok']} -> {f['n_passed']} | "
          f"pairs {f['n_pairs']} -> {f['n_pairs_quality']} -> "
          f"{f['n_pairs_well_phased']}")
    return f


def date_coincidence(sid: str, gran: pd.DataFrame) -> dict:
    """Per-solar-date day/night availability -- the real cap on pair yield."""
    g = gran[(gran.station_id == sid) & gran.phase.isin(["day", "night"])]
    if g.empty:
        return {"both": 0, "day_only": 0, "night_only": 0, "n_dates": 0,
                "cap_same_date": 0, "table": pd.DataFrame()}
    t = (g.groupby(["solar_date", "phase"]).size().unstack(fill_value=0)
          .reindex(columns=["day", "night"], fill_value=0))
    both  = t[(t.day > 0) & (t.night > 0)]
    donly = t[(t.day > 0) & (t.night == 0)]
    nonly = t[(t.day == 0) & (t.night > 0)]
    return {
        "both": int(len(both)), "day_only": int(len(donly)),
        "night_only": int(len(nonly)), "n_dates": int(len(t)),
        "cap_same_date": int(np.minimum(both.day, both.night).sum()),
        "table": t,
    }


def verify_well_phased(sid: str, gran: pd.DataFrame, pair: pd.DataFrame, K: dict):
    """Re-derive `well_phased` from the granule CSV and check it matches the pairs CSV.

    This is the signed-vs-abs trap.  The rule is

        abs(hours_from_solar_noon - THERMAL_PEAK_LAG_H) <= WELL_PHASED_DAY_H

    i.e. hfsn in [-0.5, +4.5] -- ASYMMETRIC, biased to the afternoon by the 2 h thermal
    lag.  Taking abs(hfsn) first silently admits morning granules.  Catching that here is
    what entitles the figure legends to state the rule.
    """
    g = gran[gran.station_id == sid].set_index("granule_ur")
    p = pair[pair.station_id == sid]
    if p.empty:
        return True
    hfsn = p.day_ur.map(g.hours_from_solar_noon)
    derived = ((hfsn - K["THERMAL_PEAK_LAG_H"]).abs() <= K["WELL_PHASED_DAY_H"]) & \
              (p.night_tst >= K["WELL_PHASED_NIGHT_TST"])
    ok = bool((derived.astype(int) == p.well_phased).all())
    if not ok:
        n = int((derived.astype(int) != p.well_phased).sum())
        print(f"  WARN [{sid}] well_phased re-derivation differs on {n}/{len(p)} pairs")
    naive = (hfsn.abs() - K["THERMAL_PEAK_LAG_H"]).abs() <= K["WELL_PHASED_DAY_H"]
    n_trap = int(((naive.astype(int) == 1) & (derived.astype(int) == 0)).sum())
    print(f"  [{sid}] well_phased re-derived: match={ok}; "
          f"the abs() trap would wrongly admit {n_trap} morning pairs")
    return ok


# ============================================================
# F1 -- THE FLOWCHART
# ============================================================

def stage_table(K: dict) -> list[dict]:
    """The thirteen stages, with the gate and the constant that sets it."""
    return [
        dict(n=0,  name="station latitude",     loc=":852",
             gate=f"abs(lat) <= LAT_LIMIT = {K['LAT_LIMIT']}",
             note=f"NOT the {K['LAT_NOMINAL']} in the catalogue prose", lane="pre", block=True),
        dict(n=1,  name="CMR point query",      loc=":347-352",
             gate=f"concept {K['CONCEPT_ID']} (v002)",
             note=f"point=, temporal clamped to {K['MISSION_START'][:10]}", lane="pre", block=True),
        dict(n=2,  name="malformed record",     loc=":389-390",
             gate="granule_ur and utc both present", note="", lane="pre", block=True),
        dict(n=3,  name="reprocessing dedupe",  loc=":396-401",
             gate="newest production per (tile, utc)",
             note="dropped URs NOT retained", lane="pre", block=True),
        dict(n=4,  name="orbit dedupe",         loc=":412-420",
             gate="one per (orbit, scene)",
             note="losers kept as alt_urs fallbacks", lane="pre", block=True),
        dict(n=5,  name="day / night / crossover", loc=":241-245",
             gate=f"crossover if {K['CROSSOVER_LO']} <= elev <= {K['CROSSOVER_HI']}",
             note="SOLAR ELEVATION ALONE -- clock windows deleted", lane="fork", block=True),
        dict(n=6,  name="crossover gate",       loc=":956",
             gate="phase in (day, night)",
             note="crossover rows written, never read", lane="both", block=True),
        dict(n=7,  name="window clip",          loc=":490-506",
             gate=f"n_px/{K['N_PX_EXPECTED']} >= {K['WINDOW_FRAC_MIN']}",
             note="alt-tile retry at :962-969", lane="both", block=True),
        dict(n=8,  name="QC bitfield",          loc=":441-461",
             gate="mand in {0,1} & dataq==0 & (lst_acc>=1 | qc==0)",
             note="qc==0 means UNPOPULATED, not worst-case", lane="both", block=True),
        dict(n=9,  name="cloud / water / nodata", loc=":540-544",
             gate="cloud==0 & water==0 & both valid",
             note="cloud is BINARY, not a bitfield", lane="both", block=True),
        dict(n=10, name="view zenith",          loc=":534-538",
             gate="recorded, NEVER applied",
             note="and 56.8% missing network-wide", lane="both", block=False),
        dict(n=11, name="image clear fraction", loc=":548-561",
             gate=f"clear_frac >= CLEAR_FRAC_MIN = {K['CLEAR_FRAC_MIN']}",
             note=f"denominator is the expected {K['N_PX_EXPECTED']} px", lane="both", block=True),
        dict(n=12, name="pairing",              loc=":608-626",
             gate="latest night after day, before next-day solar noon",
             note="greedy 1:1; over inwin, NOT over passed_qc", lane="join", block=True),
        dict(n=13, name="pair labels",          loc=":630-649",
             gate=f"quality = both clear; well_phased = |hfsn-{K['THERMAL_PEAK_LAG_H']}|"
                  f"<={K['WELL_PHASED_DAY_H']} & night_tst>={K['WELL_PHASED_NIGHT_TST']}",
             note="FLAGS, not gates", lane="join", block=False),
    ]


def fig_flowchart(K: dict, f: dict | None, outdir: Path, tag: str = ""):
    stages = stage_table(K)
    fig, ax = plt.subplots(figsize=(11.5, 13.5))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.35, len(stages) + 0.15)

    counts = {}
    if f:
        counts = {
            0: None, 1: f["n_hits"], 2: None,
            3: -f["n_dupe_reproc"], 4: -f["n_dupe_orbit"],
            5: f["n_overpasses"], 6: -f["n_crossover"],
            7: None, 8: None, 9: None, 10: None,
            11: f["n_passed"], 12: f["n_pairs"], 13: f["n_pairs_well_phased"],
        }

    for i, s in enumerate(stages):
        y = len(stages) - i - 1
        if s["lane"] == "fork":
            fc, ec, lw = "#fdeee8", C_DAY, 1.6
        elif s["lane"] == "join":
            fc, ec, lw = "#e8f6f0", C_OK, 1.6
        elif not s["block"]:
            fc, ec, lw = SURFACE, INK3, 1.0
        else:
            fc, ec, lw = "#eef4fd", RAMP[4], 1.0
        ax.add_patch(FancyBboxPatch(
            (0.45, y + 0.08), 7.2, 0.82,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            linestyle="--" if not s["block"] else "-"))
        ax.text(0.62, y + 0.63, f"{s['n']}", color=ec, fontsize=10, fontweight="bold")
        ax.text(1.12, y + 0.63, s["name"], color=INK, fontsize=10, fontweight="bold")
        ax.text(7.55, y + 0.63, s["loc"], color=INK3, fontsize=7.5, ha="right",
                family="monospace")
        ax.text(1.12, y + 0.36, s["gate"], color=INK2, fontsize=8, family="monospace")
        if s["note"]:
            ax.text(1.12, y + 0.16, s["note"], color=INK3, fontsize=7.5, style="italic")
        if not s["block"]:
            ax.text(7.55, y + 0.20, "non-blocking", color=INK3, fontsize=7,
                    ha="right", style="italic")

        c = counts.get(s["n"])
        if c is not None:
            col = C_REJECT if c < 0 else INK
            txt = f"−{abs(c):,}" if c < 0 else f"{c:,}"
            ax.text(8.05, y + 0.45, txt, color=col, fontsize=11, fontweight="bold")
        if i < len(stages) - 1:
            # head points DOWN, the direction granules travel
            ax.annotate("", xy=(4.05, y - 0.06), xytext=(4.05, y + 0.06),
                        arrowprops=dict(arrowstyle="-|>", color=INK3, lw=1.0))

    # the fork/join bracket
    yf = len(stages) - 5 - 1
    yj = len(stages) - 12 - 1
    ax.plot([8.9, 9.35, 9.35, 8.9], [yf + 0.5, yf + 0.5, yj + 0.5, yj + 0.5],
            color=C_DAY, lw=1.4, clip_on=False)
    ax.text(9.5, (yf + yj) / 2 + 0.5, "day and night lanes\nrun INDEPENDENTLY\nhere",
            color=C_DAY, fontsize=8, va="center", rotation=90, ha="center",
            fontweight="bold")

    title = "ECOSTRESS census: the thirteen filters"
    sub = ("constants parsed live from census_ecostress.py  ·  stage 12 reads `inwin`, "
           "NOT `passed_qc`")
    if f:
        title += f"  —  {f['station_id']}"
        sub = (f"lat {f['lat']:.2f}  ·  {f['network']}  ·  {f['elevation_m']:.0f} m  ·  "
               f"Köppen {f['kg_macro']}   |   " + sub)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, y=0.985)
    ax.set_title(sub, fontsize=8, color=INK2, pad=14, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, outdir, f"F1_flowchart{tag}")


# ============================================================
# F2 -- FORK-AND-JOIN WATERFALL
# ============================================================

def fig_waterfall(f: dict, K: dict, outdir: Path):
    sid = f["station_id"]
    fig = plt.figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.55, wspace=0.18)

    # -- top: the shared trunk, stages 1-5
    ax = fig.add_subplot(gs[0, :])
    trunk = [("CMR hits", f["n_hits"], None),
             ("− reprocessing dupes", f["n_hits"] - f["n_dupe_reproc"], f["n_dupe_reproc"]),
             ("− orbit dupes  (= overpasses)", f["n_overpasses"], f["n_dupe_orbit"]),
             ("− crossover  (= in-window)", f["n_inwindow"], f["n_crossover"])]
    y = np.arange(len(trunk))[::-1]
    for i, (lab, val, drop) in enumerate(trunk):
        hatch = "///" if i in (1, 2) else None
        ax.barh(y[i], val, height=0.62, color=RAMP[i + 2], edgecolor=SURFACE,
                linewidth=2, hatch=hatch)
        ax.text(val * 1.012, y[i], f"{val:,}", va="center", color=INK,
                fontsize=9, fontweight="bold")
        if drop:
            ax.text(val * 0.985, y[i], f"−{drop:,}  ({drop / f['n_hits'] * 100:.0f}% of hits)",
                    va="center", ha="right", color=SURFACE, fontsize=8, fontweight="bold")
    ax.set_yticks(y, [t[0] for t in trunk])
    ax.set_xlim(0, f["n_hits"] * 1.14)
    ax.set_title("trunk — stages 1–5, every granule together", loc="left")
    ax.text(1.0, 1.06, "hatched = aggregate only, the dropped URs were never retained",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=INK3, style="italic")
    ax.grid(axis="y", visible=False)

    # -- bottom left/right: the two independent lanes
    for k, lane in enumerate(("day", "night")):
        ax = fig.add_subplot(gs[1, k])
        n_in  = f[f"n_{lane}"]
        steps = [("in-window", n_in, None),
                 ("read ok", f[f"n_{lane}_read_ok"], n_in - f[f"n_{lane}_read_ok"]),
                 (f"clear_frac ≥ {K['CLEAR_FRAC_MIN']}", f[f"n_{lane}_passed"],
                  f[f"n_{lane}_read_ok"] - f[f"n_{lane}_passed"]),
                 ("entered a pair", f["n_pairs"], f[f"n_{lane}_unpaired"])]
        col = C_DAY if lane == "day" else C_NIGHT
        y = np.arange(len(steps))[::-1]
        for i, (lab, val, drop) in enumerate(steps):
            hatch = "xxx" if i == 3 else None
            ax.barh(y[i], val, height=0.6, color=col, alpha=1.0 - 0.16 * i,
                    edgecolor=SURFACE, linewidth=2, hatch=hatch)
            ax.text(val * 1.015, y[i], f"{val:,}", va="center", color=INK,
                    fontsize=9, fontweight="bold")
            if drop:
                pct = drop / n_in * 100 if n_in else 0
                ax.text(max(val * 0.98, n_in * 0.02), y[i], f"−{drop:,} ({pct:.0f}%)",
                        va="center", ha="right", color=SURFACE, fontsize=8,
                        fontweight="bold")
        ax.set_yticks(y, [s[0] for s in steps])
        ax.set_xlim(0, max(n_in, 1) * 1.2)
        ax.set_title(f"{lane} lane", loc="left", color=col)
        ax.grid(axis="y", visible=False)
        if k == 0:
            ax.text(0, -0.28, "cross-hatched = DERIVED; census_ecostress.py:625 drops "
                              "unpaired passes with a bare `continue`,\nrecording neither "
                              "a row nor a counter",
                    transform=ax.transAxes, fontsize=7.5, color=C_DERIVED, style="italic")

    fig.suptitle(f"{sid} — the funnel FORKS at stage 5 and rejoins at stage 12",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.5, 0.925,
             f"pairing consumes the in-window set ({f['n_inwindow']:,}), never the "
             f"QC-passed set ({f['n_passed']:,}) — those two numbers are on different branches",
             ha="center", fontsize=8.5, color=INK2)
    save(fig, outdir, f"F2_waterfall_{sid}")


# ============================================================
# F3 -- SOLAR ELEVATION ACROSS THE WHOLE RECORD
# ============================================================

def fig_solar_timeseries(sid: str, gran: pd.DataFrame, pair: pd.DataFrame,
                         f: dict, K: dict, outdir: Path):
    g = gran[gran.station_id == sid].sort_values("utc_ts")
    p = pair[pair.station_id == sid]
    lat = f["lat"]
    lo, hi = K["CROSSOVER_LO"], K["CROSSOVER_HI"]

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1.5, 0.55],
                                          "hspace": 0.12})
    t0 = g.utc_ts.min()
    t1 = g.utc_ts.max()
    days = pd.date_range(t0.tz_localize(None).normalize(),
                         t1.tz_localize(None).normalize(), freq="D")
    doy = days.dayofyear.values.astype(float)

    # ---- panel 1: solar elevation
    ax = axes[0]
    ax.axhspan(lo, hi, color=C_CROSS, alpha=0.30, zorder=0, lw=0)
    ax.axhline(0, color=INK3, lw=0.8, ls=":", zorder=1)
    ax.plot(days, max_elevation(doy, lat), color=INK3, lw=1.0, ls="--", zorder=2,
            label="sun's elevation at solar noon")
    for phase in ("crossover", "day", "night"):
        sub = g[g.phase == phase]
        if sub.empty:
            continue
        passed = sub.passed_qc == 1
        ax.scatter(sub.utc_ts[passed], sub.solar_elev[passed], s=7,
                   color=PHASE_COLOR[phase], alpha=0.85, linewidths=0, zorder=4)
        ax.scatter(sub.utc_ts[~passed], sub.solar_elev[~passed], s=9,
                   facecolors="none", edgecolors=PHASE_COLOR[phase],
                   linewidths=0.5, alpha=0.65, zorder=3)
    ax.set_ylabel("solar elevation at overpass  (deg)")
    ax.set_ylim(-90, 92)
    ax.set_yticks([-90, -60, -30, lo, 0, hi, 30, 60, 90])
    ax.text(0.004, (hi - lo) / 2 + lo, f"  CROSSOVER  [{lo:g}, {hi:g}]  →  discarded, "
                                       f"never read   (−{f['n_crossover']:,})",
            transform=ax.get_yaxis_transform(), va="center", fontsize=8,
            color="#57564f", fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", color=C_DAY, label="day", ms=5),
               Line2D([], [], marker="o", ls="", color=C_NIGHT, label="night", ms=5),
               Line2D([], [], marker="o", ls="", color=C_CROSS, label="crossover (dropped)",
                      ms=5),
               Line2D([], [], marker="o", ls="", mfc="none", mec=INK2,
                      label=f"hollow = failed clear_frac ≥ {K['CLEAR_FRAC_MIN']}", ms=5),
               Line2D([], [], ls="--", color=INK3, label="solar-noon elevation")]
    ax.legend(handles=handles, loc="lower left", ncol=5, fontsize=7.5,
              bbox_to_anchor=(0, 1.005))
    fig.suptitle(f"{sid}  —  every ECOSTRESS overpass in the record, and what the "
                 f"filters do to it", fontsize=12.5, fontweight="bold", color=INK,
                 x=0.125, ha="left", y=0.965)

    # ---- panel 2: true solar time -- the ISS precession
    ax = axes[1]
    sr_m, sr_e = tst_at_elevation(doy, lat, 0.0)
    hi_m, hi_e = tst_at_elevation(doy, lat, hi)
    lo_m, lo_e = tst_at_elevation(doy, lat, lo)
    # the crossover band in (date, TST) space: between the -5 and +10 contours,
    # on both the morning and the evening limb
    ax.fill_between(days, lo_m, hi_m, color=C_CROSS, alpha=0.30, lw=0, zorder=0)
    ax.fill_between(days, hi_e, lo_e, color=C_CROSS, alpha=0.30, lw=0, zorder=0)
    ax.plot(days, sr_m, color=INK3, lw=0.9, ls=":", zorder=2)
    ax.plot(days, sr_e, color=INK3, lw=0.9, ls=":", zorder=2)
    for phase in ("crossover", "day", "night"):
        sub = g[g.phase == phase]
        if sub.empty:
            continue
        ax.scatter(sub.utc_ts, sub.tst, s=6, color=PHASE_COLOR[phase],
                   alpha=0.8, linewidths=0, zorder=4)
    ax.set_ylabel("true solar time at overpass  (h)")
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 4))
    bbox = dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none",
                alpha=0.88)
    ax.text(0.995, 0.965, "diagonal stripes = the ISS overpass local time precessing "
                          "through 24 h every ~60 days",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2,
            style="italic", bbox=bbox, zorder=6)
    ax.text(0.004, 0.02, "dotted = sunrise / sunset;  shaded = the crossover band in "
                         "time-of-day space",
            transform=ax.transAxes, fontsize=7.5, color=INK3, bbox=bbox, zorder=6)

    # ---- panel 3: what actually became a pair
    ax = axes[2]
    if not p.empty:
        ax.eventplot([p.day_ts.dropna()], lineoffsets=[0.78], linelengths=[0.30],
                     colors=[INK3], linewidths=0.9)
        q = p[p.quality == 1]
        w = p[(p.quality == 1) & (p.well_phased == 1)]
        if not q.empty:
            ax.eventplot([q.day_ts.dropna()], lineoffsets=[0.46], linelengths=[0.30],
                         colors=[C_OK], linewidths=0.9)
        if not w.empty:
            ax.eventplot([w.day_ts.dropna()], lineoffsets=[0.14], linelengths=[0.30],
                         colors=[C_DERIVED], linewidths=1.1)
    ax.set_yticks([0.78, 0.46, 0.14],
                  [f"candidate pair  ({f['n_pairs']})",
                   f"+ both halves clear  ({f['n_pairs_quality']})",
                   f"+ well-phased  ({f['n_pairs_well_phased']})"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("date")
    if p.empty:
        ax.text(0.5, 0.5, "NO PAIRS — nothing survives to this row",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=C_REJECT, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    save(fig, outdir, f"F3_solar_timeseries_{sid}")


# ============================================================
# F3c -- UTC -> TRUE SOLAR TIME -> SOLAR ELEVATION
# ============================================================

def fig_utc_to_solar(sid: str, gran: pd.DataFrame, f: dict, K: dict, outdir: Path):
    """The conversion chain, one panel per stage, so its usefulness is checkable.

    The census never classifies on UTC.  It converts UTC -> true solar time (longitude
    + equation of time) -> solar elevation (TST + declination + latitude), and classifies
    on elevation alone.  How much that conversion BUYS is a function of longitude: at a
    station near Greenwich UTC and TST coincide and the chain looks like a no-op; at
    120 deg west the UTC hour is actively misleading about day and night.
    """
    g = gran[gran.station_id == sid].sort_values("utc_ts").copy()
    if g.empty:
        return
    lat, lon = f["lat"], f["lon"]
    g["utc_h"] = (g.utc_ts.dt.hour + g.utc_ts.dt.minute / 60.0
                  + g.utc_ts.dt.second / 3600.0)
    # TST - UTC, unwrapped onto (-12, +12]
    d = ((g.tst - g.utc_h + 12.0) % 24.0) - 12.0
    lon_off = lon / 15.0
    eot = (d - lon_off + 12.0) % 24.0 - 12.0        # residual == equation of time

    fig = plt.figure(figsize=(15.2, 9.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.05, 1.0], hspace=0.16, wspace=0.05)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for a in axes[:2]:
        a.sharex(axes[2])
        a.tick_params(labelbottom=False)   # manual sharex does not hide these

    days = pd.date_range(g.utc_ts.min().tz_localize(None).normalize(),
                         g.utc_ts.max().tz_localize(None).normalize(), freq="D")
    doy = days.dayofyear.values.astype(float)

    def scat(ax, ycol):
        for phase in ("crossover", "day", "night"):
            sub = g[g.phase == phase]
            if not sub.empty:
                ax.scatter(sub.utc_ts, sub[ycol], s=6, color=PHASE_COLOR[phase],
                           alpha=0.8, linewidths=0, zorder=3)

    bbox = dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none",
                alpha=0.9)

    # -- 1. what the granule filename gives you
    ax = axes[0]
    scat(ax, "utc_h")
    ax.set_ylabel("UTC hour of acquisition")
    ax.set_ylim(0, 24); ax.set_yticks(range(0, 25, 6))
    ax.set_title(f"{sid}  —  step 1:  UTC, straight off the granule", loc="left")
    ax.text(0.004, 0.94, "orange and blue are interleaved here iff UTC is a poor proxy "
                         "for local time at this longitude",
            transform=ax.transAxes, va="top", fontsize=8, color=INK2, style="italic",
            bbox=bbox, zorder=6)

    # -- 2. after the longitude + equation-of-time correction
    ax = axes[1]
    sr_m, sr_e = tst_at_elevation(doy, lat, 0.0)
    ax.fill_between(days, sr_m, sr_e, color=C_DAY, alpha=0.07, lw=0, zorder=0)
    ax.plot(days, sr_m, color=INK3, lw=0.9, ls=":", zorder=2)
    ax.plot(days, sr_e, color=INK3, lw=0.9, ls=":", zorder=2)
    scat(ax, "tst")
    ax.set_ylabel("true solar time  (h)")
    ax.set_ylim(0, 24); ax.set_yticks(range(0, 25, 6))
    ax.set_title(f"step 2:  TST = UTC + lon/15 + EoT   "
                 f"(lon {lon:+.3f}° → {lon_off:+.2f} h;  "
                 f"EoT spans {eot.min() * 60:+.0f} to {eot.max() * 60:+.0f} min)",
                 loc="left")
    ax.text(0.004, 0.06, "shaded = between sunrise and sunset", transform=ax.transAxes,
            fontsize=7.5, color=INK3, bbox=bbox, zorder=6)

    # -- 3. the quantity the filter actually reads
    ax = axes[2]
    ax.axhspan(K["CROSSOVER_LO"], K["CROSSOVER_HI"], color=C_CROSS, alpha=0.30, lw=0,
               zorder=0)
    ax.plot(days, max_elevation(doy, lat), color=INK3, lw=1.0, ls="--", zorder=2)
    scat(ax, "solar_elev")
    ax.set_ylabel("solar elevation  (deg)")
    ax.set_ylim(-90, 92)
    ax.set_yticks([-90, -45, K["CROSSOVER_LO"], K["CROSSOVER_HI"], 45, 90])
    ax.set_xlabel("date")
    ax.set_title(f"step 3:  elevation = f(TST, declination, lat {lat:.2f}°)   "
                 f"— the ONLY quantity census_ecostress.py:241 classifies on", loc="left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.text(0.004, 0.50, f"  crossover [{K['CROSSOVER_LO']:g}, {K['CROSSOVER_HI']:g}] → "
                         f"discarded", transform=ax.get_yaxis_transform(),
            va="center", fontsize=8, color="#57564f", fontweight="bold", zorder=6)

    # -- the worked table, so individual numbers are checkable
    axt = fig.add_subplot(gs[:, 1])
    axt.set_axis_off()
    sample = g[g.phase.isin(["day", "night"])]
    if len(sample) > 12:
        sample = sample.iloc[np.linspace(0, len(sample) - 1, 12).astype(int)]
    lines = [f"{'UTC':<17}{'TST':>7}{'elev':>8}  phase", "─" * 43]
    for _, r in sample.iterrows():
        lines.append(f"{str(r.utc)[:16]:<17}{r.tst:>7.2f}{r.solar_elev:>8.1f}  {r.phase}")
    axt.text(0.0, 1.0, "\n".join(lines), transform=axt.transAxes, va="top", ha="left",
             family="monospace", fontsize=7.4, color=INK)
    delta = (g.tst - g.utc_h).abs().median()
    verdict = ("UTC is nearly useless to convert here — this station sits on the\n"
               "prime meridian, so UTC and solar time already agree."
               if abs(lon_off) < 0.5 else
               "UTC cannot be read as local time here: the same UTC hour lands in\n"
               "daylight or in darkness depending on the date.")
    axt.text(0.0, 0.30, f"longitude offset  {lon_off:+.2f} h\n"
                        f"median |TST − UTC|  {delta:.2f} h\n\n{verdict}",
             transform=axt.transAxes, va="top", ha="left", fontsize=8, color=INK2)
    axt.set_title("worked examples", loc="left")

    save(fig, outdir, f"F3c_utc_to_solar_{sid}")


# ============================================================
# F3b / F4 -- PHASE SPACE
# ============================================================

def fig_solar_geometry(sid: str, gran: pd.DataFrame, f: dict, K: dict, outdir: Path,
                       color_by_clear: bool = False):
    g = gran[gran.station_id == sid]
    lo, hi = K["CROSSOVER_LO"], K["CROSSOVER_HI"]
    fig = plt.figure(figsize=(8.6, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.04, wspace=0.04)
    ax  = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)

    ax.axhspan(lo, hi, color=C_CROSS, alpha=0.30, lw=0, zorder=0)
    if color_by_clear:
        gg = g[g.read_ok == 1]
        sc = ax.scatter(gg.tst, gg.solar_elev, c=gg.clear_frac, s=11,
                        cmap="Blues", vmin=0, vmax=1, linewidths=0,
                        alpha=0.9, zorder=3)
        cb = fig.colorbar(sc, ax=axr, fraction=0.5, pad=0.35)
        cb.set_label("clear_frac", fontsize=8)
        cb.ax.axhline(K["CLEAR_FRAC_MIN"], color=C_REJECT, lw=1.4)
        name, ttl = "F4_solar_geometry_qc", "coloured by clear_frac (read_ok only)"
    else:
        for phase in ("crossover", "day", "night"):
            sub = g[g.phase == phase]
            if sub.empty:
                continue
            ax.scatter(sub.tst, sub.solar_elev, s=9, color=PHASE_COLOR[phase],
                       alpha=0.75, linewidths=0, zorder=3, label=phase)
        ax.legend(loc="upper left", fontsize=8)
        name, ttl = "F3b_solar_geometry", "coloured by phase"

    ax.set_xlabel("true solar time  (h)")
    ax.set_ylabel("solar elevation  (deg)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_ylim(-90, 92)
    ax.axhline(0, color=INK3, lw=0.8, ls=":")

    axt.hist(g.tst.dropna(), bins=48, range=(0, 24), color=RAMP[3], edgecolor=SURFACE,
             linewidth=0.5)
    axt.set_axis_off()
    axr.hist(g.solar_elev.dropna(), bins=45, range=(-90, 90), orientation="horizontal",
             color=RAMP[3], edgecolor=SURFACE, linewidth=0.5)
    axr.set_axis_off()

    axt.set_title(f"{sid} — overpass phase space, {ttl}", loc="left", fontsize=11)
    save(fig, outdir, f"{name}_{sid}")


# ============================================================
# F5 -- WHY QC FAILED
# ============================================================

def fig_qc_attribution(sid: str, gran: pd.DataFrame, K: dict, outdir: Path):
    g = gran[gran.station_id == sid]
    fail = g[(g.read_ok == 1) & (g.passed_qc == 0)]
    ok   = g[(g.read_ok == 1) & (g.passed_qc == 1)]
    bad  = g[(g.phase.isin(["day", "night"])) & (g.read_ok == 0)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1.1, 1.5], "wspace": 0.30})

    comps = [("frac_cloud", "cloud"), ("frac_water", "water"),
             ("frac_mand00", "QC mand=00"), ("frac_mand01", "QC mand=01 (degraded)"),
             ("frac_lstacc_ge2", "LST acc ≥ 2"), ("valid_frac", "valid (mand ≤ 1)")]
    ax = axes[0]
    xs = np.arange(len(comps))
    mf = [fail[c].mean() if len(fail) else np.nan for c, _ in comps]
    mo = [ok[c].mean() if len(ok) else np.nan for c, _ in comps]
    ax.bar(xs - 0.2, mo, 0.38, color=C_OK, edgecolor=SURFACE, linewidth=2,
           label=f"passed  (n={len(ok):,})")
    ax.bar(xs + 0.2, mf, 0.38, color=C_REJECT, edgecolor=SURFACE, linewidth=2,
           label=f"failed  (n={len(fail):,})")
    ax.set_xticks(xs, [l for _, l in comps], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("mean fraction of the 1024-px window")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.set_title("which predicate separates pass from fail", loc="left")

    ax = axes[1]
    if len(fail):
        ax.hist(fail.clear_frac.dropna(), bins=26, range=(0, 1), color=C_REJECT,
                edgecolor=SURFACE, linewidth=1.2, alpha=0.9)
    ax.axvline(K["CLEAR_FRAC_MIN"], color=INK, lw=1.6)
    ax.text(K["CLEAR_FRAC_MIN"], ax.get_ylim()[1] * 0.96,
            f" CLEAR_FRAC_MIN = {K['CLEAR_FRAC_MIN']}", fontsize=8, color=INK,
            va="top", fontweight="bold")
    ax.set_xlabel("clear_frac")
    ax.set_ylabel("granules")
    ax.set_title("how far the failures fall short", loc="left")

    ax = axes[2]
    if len(bad):
        reasons = (bad.error.fillna("(blank)").astype(str)
                      .str.replace(r"\d+", "N", regex=True).str.slice(0, 46)
                      .value_counts().head(6))
        ax.barh(np.arange(len(reasons))[::-1], reasons.values, height=0.6,
                color=C_DERIVED, edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(np.arange(len(reasons))[::-1], reasons.index, fontsize=7.5)
        for i, v in enumerate(reasons.values):
            ax.text(v * 1.02, np.arange(len(reasons))[::-1][i], f"{v:,}", va="center",
                    fontsize=8, color=INK)
        ax.set_xlim(0, reasons.values.max() * 1.22)
    else:
        ax.text(0.5, 0.5, "no read failures", transform=ax.transAxes, ha="center",
                color=INK3)
        ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_title(f"read failures, stage 7  (n={len(bad):,})", loc="left")

    fig.suptitle(f"{sid} — attributing the QC rejections", fontsize=12,
                 fontweight="bold", color=INK, y=1.03)
    save(fig, outdir, f"F5_qc_attribution_{sid}")


# ============================================================
# F6 -- CLEAR-FRACTION SWEEP
# ============================================================

def fig_clear_sweep(stations: list[str], gran: pd.DataFrame, K: dict, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.22})
    ax = axes[0]
    for i, sid in enumerate(stations):
        g = gran[(gran.station_id == sid) & (gran.read_ok == 1)]
        if g.empty:
            continue
        ax.hist(g.clear_frac.dropna(), bins=26, range=(0, 1), histtype="step",
                lw=1.8, color=RAMP[min(i * 2 + 2, 9)], label=f"{sid}  (n={len(g):,})",
                density=True)
    ax.axvline(K["CLEAR_FRAC_MIN"], color=C_REJECT, lw=1.8)
    ax.text(K["CLEAR_FRAC_MIN"], ax.get_ylim()[1] * 0.97,
            f" CLEAR_FRAC_MIN = {K['CLEAR_FRAC_MIN']}", color=C_REJECT, fontsize=8,
            va="top", fontweight="bold")
    ax.set_xlabel("clear_frac over the 2.24 km station window")
    ax.set_ylabel("density")
    ax.legend(loc="upper left")
    ax.set_title("where the images sit relative to the image-level floor", loc="left")

    ax = axes[1]
    thr = np.array([0.5, 0.7, 0.9])
    w = 0.8 / max(len(stations), 1)
    for i, sid in enumerate(stations):
        g = gran[(gran.station_id == sid) & (gran.read_ok == 1)]
        if g.empty:
            continue
        surv = [(g.clear_frac >= t).mean() * 100 for t in thr]
        ax.bar(np.arange(3) + i * w - 0.4 + w / 2, surv, w * 0.9,
               color=RAMP[min(i * 2 + 2, 9)], edgecolor=SURFACE, linewidth=2, label=sid)
        for j, v in enumerate(surv):
            ax.text(np.arange(3)[j] + i * w - 0.4 + w / 2, v + 1.5, f"{v:.0f}",
                    ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(3), [f"≥ {t}" for t in thr])
    ax.set_ylabel("% of reads surviving")
    ax.set_ylim(0, 108)
    ax.set_title("the §36.12 sweep", loc="left")
    save(fig, outdir, "F6_clear_frac_sweep")


# ============================================================
# F7 -- THE PAIRS THAT FORMED
# ============================================================

def fig_pairing(sid: str, pair: pd.DataFrame, f: dict, K: dict, outdir: Path):
    p = pair[pair.station_id == sid]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0),
                             gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.22})

    ax = axes[0]
    lo_h = K["THERMAL_PEAK_LAG_H"] - K["WELL_PHASED_DAY_H"]
    hi_h = K["THERMAL_PEAK_LAG_H"] + K["WELL_PHASED_DAY_H"]
    if not p.empty:
        for _, r in p.iterrows():
            col = C_OK if r.quality == 1 else C_CROSS
            ax.plot([r.day_tst, r.night_tst], [r.day_elev, r.night_elev],
                    color=col, lw=1.6 if r.well_phased else 0.7,
                    alpha=0.85 if r.quality else 0.35,
                    ls="-" if r.well_phased else ":", zorder=3 if r.quality else 2)
        ax.scatter(p.day_tst, p.day_elev, s=16, color=C_DAY, zorder=4, linewidths=0)
        ax.scatter(p.night_tst, p.night_elev, s=16, color=C_NIGHT, zorder=4, linewidths=0)
        ax.axvspan(K["WELL_PHASED_NIGHT_TST"], 24, color=C_DERIVED, alpha=0.05, lw=0)
    else:
        ax.text(0.5, 0.5, "NO CANDIDATE PAIRS\n\nnothing to draw — by construction",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                color=C_REJECT, fontweight="bold")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("true solar time  (h)")
    ax.set_ylabel("solar elevation  (deg)")
    ax.set_title(f"{sid} — each candidate pair, day half → night half", loc="left")
    ax.legend(handles=[
        Line2D([], [], color=C_OK, lw=1.8, label=f"both halves clear  "
                                                 f"({f['n_pairs_quality']})"),
        Line2D([], [], color=C_CROSS, lw=1.0, ls=":", label="one half cloudy"),
        Line2D([], [], color=INK2, lw=1.8, label="solid = well-phased"),
    ], loc="upper right", fontsize=7.5)

    ax = axes[1]
    if not p.empty:
        ax.hist(p.dt_hours.dropna(), bins=np.arange(0, 27, 1), color=RAMP[4],
                edgecolor=SURFACE, linewidth=1.2)
        ax.set_xlim(0, 26)
        med = p.dt_hours.median()
        ax.axvline(med, color=C_REJECT, lw=1.4)
        ax.text(med, ax.get_ylim()[1] * 0.96, f" median {med:.1f} h", fontsize=8,
                color=C_REJECT, va="top", fontweight="bold")
        gaps = "bimodal — two discrete orbital configurations, not a continuum"
        ax.text(0.5, -0.30, gaps, transform=ax.transAxes, ha="center", fontsize=8,
                color=INK2, style="italic")
    else:
        ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center", va="center",
                fontsize=20, color=INK3)
    ax.set_xlabel("dt_hours  (day → night separation)")
    ax.set_ylabel("pairs")
    ax.set_title("upper bound is next-day solar noon, not a fixed dt", loc="left")
    save(fig, outdir, f"F7_pairing_{sid}")


# ============================================================
# F7b -- WHY SO FEW PAIRS
# ============================================================

def fig_pairing_loss(sid: str, gran: pd.DataFrame, f: dict, co: dict, outdir: Path):
    fig = plt.figure(figsize=(14.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.15, 1.0], wspace=0.28)

    # ---- calendar of day/night availability
    ax = fig.add_subplot(gs[0, 0])
    t = co["table"]
    if len(t):
        years = sorted({d.year for d in t.index})
        ymap = {y: i for i, y in enumerate(years)}
        code = np.full((len(years), 366), np.nan)
        for d, row in t.iterrows():
            v = 2.0 if (row.day > 0 and row.night > 0) else (0.0 if row.day > 0 else 1.0)
            code[ymap[d.year], d.dayofyear - 1] = v
        # Only "BOTH" can produce a pair, so only "BOTH" gets a saturated colour --
        # the half-days are muted tints of the same two hues.  Equal saturation here
        # would hide the 17% inside a wall of stripes.
        cmap = matplotlib.colors.ListedColormap(["#f7cdbc", "#bdd5f2", C_OK])
        cmap.set_bad(GRID)
        ax.imshow(np.ma.masked_invalid(code), aspect="auto", cmap=cmap, vmin=-0.5,
                  vmax=2.5, interpolation="nearest",
                  extent=(0, 366, len(years) - 0.5, -0.5))
        ax.set_yticks(range(len(years)), years, fontsize=8)
        ax.set_xticks([0, 90, 181, 273, 365], ["Jan", "Apr", "Jul", "Oct", "Dec"])
        ax.grid(False)
    ax.set_title("every solar date with an in-window overpass", loc="left")
    ax.legend(handles=[Patch(color="#f7cdbc", label=f"day only  ({co['day_only']})"),
                       Patch(color="#bdd5f2", label=f"night only  ({co['night_only']})"),
                       Patch(color=C_OK, label=f"BOTH → pairable  ({co['both']})")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8)
    pct = co["both"] / co["n_dates"] * 100 if co["n_dates"] else 0
    ax.text(0, 1.10, f"only {co['both']} of {co['n_dates']} dates ({pct:.0f}%) can "
                     f"produce a pair at all", transform=ax.transAxes,
            fontsize=8.5, color=C_REJECT, fontweight="bold")

    # ---- the fork/join flow
    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    nd, nn, npr = f["n_day"], f["n_night"], f["n_pairs"]
    mx = max(nd, nn, 1)
    for y0, n, lab, col in ((7.4, nd, "day passes", C_DAY),
                            (2.6, nn, "night passes", C_NIGHT)):
        h = 2.6 * n / mx
        ax.add_patch(Rectangle((0.3, y0 - h / 2), 1.5, h, color=col))
        ax.text(1.05, y0 + h / 2 + 0.35, f"{n:,}\n{lab}", ha="center", fontsize=9,
                color=INK, fontweight="bold", va="bottom")
        hp = 2.6 * npr / mx
        ax.fill_between([1.8, 6.4], [y0 - hp / 2, 5 - hp / 2], [y0 + hp / 2, 5 + hp / 2],
                        color=col, alpha=0.32, lw=0)
        un = f[f"n_{'day' if col == C_DAY else 'night'}_unpaired"]
        hu = 2.6 * un / mx
        yo = y0 + (1.4 if col == C_DAY else -1.4)
        ax.fill_between([1.8, 6.4], [y0 - hu / 2, yo - hu / 2], [y0 + hu / 2, yo + hu / 2],
                        color=C_DERIVED, alpha=0.20, lw=0)
        ax.text(6.6, yo, f"{un:,} unpaired\n({un / max(n, 1) * 100:.0f}%)", fontsize=8,
                color=C_DERIVED, va="center", fontweight="bold")
    hp = 2.6 * npr / mx
    ax.add_patch(Rectangle((6.4, 5 - hp / 2), 1.3, hp, color=C_OK))
    ax.text(7.05, 5 - hp / 2 - 0.3, f"{npr:,}\npairs", ha="center", va="top",
            fontsize=10, color=INK, fontweight="bold")
    ax.set_title("stage 12 — greedy one-to-one", loc="left")
    ax.text(0, -0.02, "unpaired flows are DERIVED (n_phase − n_pairs);\n"
                      "census_ecostress.py:625 records nothing",
            transform=ax.transAxes, fontsize=7.5, color=C_DERIVED, style="italic")

    # ---- the ordering correction
    ax = fig.add_subplot(gs[0, 2])
    steps = [("in-window", f["n_inwindow"], ""),
             ("candidate pairs", f["n_pairs"], "orbital coincidence"),
             ("both halves clear", f["n_pairs_quality"], "cloud"),
             ("well-phased", f["n_pairs_well_phased"], "phase geometry")]
    y = np.arange(len(steps))[::-1]
    for i, (lab, v, why) in enumerate(steps):
        ax.barh(y[i], v, height=0.6, color=RAMP[i * 2 + 2], edgecolor=SURFACE, linewidth=2)
        ax.text(v + f["n_inwindow"] * 0.015, y[i], f"{v:,}", va="center",
                fontsize=9, fontweight="bold", color=INK)
        if i:
            prev = steps[i - 1][1]
            drop = (1 - v / prev) * 100 if prev else 0
            # fixed column, clear of every value label -- the bars past the first are
            # all short, so a constant x cannot collide with them
            ax.text(f["n_inwindow"] * 0.42, y[i], f"−{drop:.0f}%   {why}",
                    ha="left", va="center", fontsize=8, color=C_REJECT,
                    fontweight="bold")
    ax.set_yticks(y, [s[0] for s in steps], fontsize=8)
    # Linear, deliberately.  A log axis would flatter the survivors and hide that the
    # first step alone throws away ~9 granules in 10.
    ax.set_xlim(0, f["n_inwindow"] * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_title("the true ordering (linear x)", loc="left")
    ax.text(0, -0.20, f"`passed_qc` = {f['n_passed']:,} is NOT a step on this path —\n"
                      f"it is attached afterwards, as a label",
            transform=ax.transAxes, fontsize=7.5, color=INK2, style="italic")

    fig.suptitle(f"{sid} — why {f['n_inwindow']:,} in-window granules yield only "
                 f"{f['n_pairs']:,} pairs", fontsize=13, fontweight="bold", color=INK,
                 y=1.04)
    save(fig, outdir, f"F7b_pairing_loss_{sid}")


# ============================================================
# F8 / F9
# ============================================================

def fig_timeline(sid: str, gran: pd.DataFrame, pair: pd.DataFrame, outdir: Path):
    g = gran[gran.station_id == sid]
    p = pair[pair.station_id == sid]
    yrs = sorted(g.year.dropna().astype(int).unique())
    if not yrs:
        return
    rows = {
        "crossover (dropped)": [int(((g.year == y) & (g.phase == "crossover")).sum())
                                for y in yrs],
        "read failed":         [int(((g.year == y) & (g.phase != "crossover") &
                                     (g.read_ok == 0)).sum()) for y in yrs],
        "QC failed":           [int(((g.year == y) & (g.read_ok == 1) &
                                     (g.passed_qc == 0)).sum()) for y in yrs],
        "passed QC":           [int(((g.year == y) & (g.passed_qc == 1)).sum())
                                for y in yrs],
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    ax = axes[0]
    bottom = np.zeros(len(yrs))
    for (lab, vals), col in zip(rows.items(), [C_CROSS, C_DERIVED, C_REJECT, C_OK]):
        ax.bar(yrs, vals, bottom=bottom, color=col, edgecolor=SURFACE, linewidth=2,
               label=lab)
        bottom += np.array(vals)
    ax.set_ylabel("overpasses")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=8)
    ax.set_title(f"{sid} — the funnel year by year", loc="left", pad=22)

    ax = axes[1]
    if not p.empty:
        py = p.day_solar_date.astype(str).str.slice(0, 4).astype(int)
        ax.bar(yrs, [int((py == y).sum()) for y in yrs], color=RAMP[4],
               edgecolor=SURFACE, linewidth=2, label="candidate pairs")
        pq = p[p.quality == 1].day_solar_date.astype(str).str.slice(0, 4).astype(int)
        ax.bar(yrs, [int((pq == y).sum()) for y in yrs], color=C_OK,
               edgecolor=SURFACE, linewidth=2, label="both halves clear")
        ax.legend(ncol=2, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no pairs in any year", transform=ax.transAxes, ha="center",
                va="center", color=C_REJECT, fontweight="bold")
    ax.set_ylabel("pairs")
    ax.set_xlabel("year")
    save(fig, outdir, f"F8_timeline_{sid}")


def fig_network(stations: list[str], gran: pd.DataFrame, logd: pd.DataFrame,
                outdir: Path, floor: int = 20):
    ok = logd[logd.status == "ok"].copy()
    meta = gran.groupby("station_id").agg(lat=("lat", "first"),
                                          elev=("elevation_m", "first"),
                                          kg=("kg_macro", "first"))
    ok = ok.join(meta, on="station_id")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"wspace": 0.26})

    for ax, (xcol, xlab) in zip(axes[:2], [("lat", "station latitude (deg)"),
                                           ("elev", "elevation (m)")]):
        ax.scatter(ok[xcol].abs() if xcol == "lat" else ok[xcol],
                   ok.n_pairs_well_phased, s=14, color=INK3, alpha=0.5, linewidths=0)
        for sid in stations:
            r = ok[ok.station_id == sid]
            if r.empty:
                continue
            x = abs(float(r[xcol].iloc[0])) if xcol == "lat" else float(r[xcol].iloc[0])
            ax.scatter([x], [float(r.n_pairs_well_phased.iloc[0])], s=70, zorder=5,
                       color=C_REJECT, edgecolors=SURFACE, linewidths=1.5)
            ax.annotate(sid, (x, float(r.n_pairs_well_phased.iloc[0])),
                        textcoords="offset points", xytext=(6, 5), fontsize=7.5,
                        color=INK)
        ax.axhline(floor, color=C_OK, lw=1.4, ls="--")
        ax.text(ax.get_xlim()[1], floor, f" §36.16 floor = {floor} ", ha="right",
                va="bottom", fontsize=7.5, color=C_OK, fontweight="bold")
        ax.set_xlabel(xlab)
        ax.set_ylabel("well-phased quality pairs")

    ax = axes[2]
    grp = ok.groupby("kg").agg(n=("station_id", "size"),
                               med=("n_pairs_well_phased", "median"),
                               clearing=("n_pairs_well_phased",
                                         lambda s: (s >= floor).sum()))
    grp = grp.sort_values("clearing", ascending=False)
    yy = np.arange(len(grp))[::-1]
    ax.barh(yy, grp.n, height=0.62, color=GRID, edgecolor=SURFACE, linewidth=2,
            label="stations")
    ax.barh(yy, grp.clearing, height=0.62, color=C_OK, edgecolor=SURFACE, linewidth=2,
            label=f"clearing ≥{floor}")
    ax.set_yticks(yy, grp.index, fontsize=8)
    ax.set_xlabel("stations")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8)
    ax.set_title("by Köppen macro-class", loc="left")

    n_clear = int((ok.n_pairs_well_phased >= floor).sum())
    fig.suptitle(f"network context — {len(ok)} stations censused, {n_clear} clear the "
                 f"§36.16 ≥{floor}-pair floor", fontsize=12, fontweight="bold",
                 color=INK, y=1.04)
    save(fig, outdir, "F9_network_context")


# ============================================================
# JSON FOR THE INTERACTIVE PAGE
# ============================================================

def emit_json(stations, gran, pair, funnels, coincid, K, outdir: Path):
    gcols = ["granule_ur", "utc", "tst", "hours_from_solar_noon", "solar_elev",
             "solar_date_str", "year", "phase", "read_ok", "clear_frac", "valid_frac",
             "window_frac", "frac_cloud", "frac_water", "frac_mand00", "frac_mand01",
             "frac_lstacc_ge2", "vza_mean_abs", "passed_qc", "error"]
    pcols = ["day_ur", "night_ur", "day_utc", "night_utc", "dt_hours", "day_tst",
             "night_tst", "day_elev", "night_elev", "elev_drop", "day_solar_date",
             "night_solar_date", "well_phased", "day_clear", "night_clear", "quality"]
    payload = {
        "constants": {k: K[k] for k in
                      ["CROSSOVER_LO", "CROSSOVER_HI", "CLEAR_FRAC_MIN", "WINDOW_FRAC_MIN",
                       "N_PX_EXPECTED", "LAT_LIMIT", "LAT_NOMINAL", "THERMAL_PEAK_LAG_H",
                       "WELL_PHASED_DAY_H", "WELL_PHASED_NIGHT_TST", "MISSION_START",
                       "CONCEPT_ID", "TILE_M", "PIXEL_M"]},
        "stages": stage_table(K),
        "stations": {},
    }
    for sid in stations:
        g = gran[gran.station_id == sid][gcols]
        p = pair[pair.station_id == sid][pcols]
        payload["stations"][sid] = {
            "funnel": funnels[sid],
            "coincidence": {k: v for k, v in coincid[sid].items() if k != "table"},
            "granules": json.loads(g.to_json(orient="records")),
            "pairs": json.loads(p.to_json(orient="records")),
        }
    out = outdir / "viz_data.json"
    out.write_text(json.dumps(payload))
    print(f"  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    ap.add_argument("--outdir", default=str(ROOT / "fig" / "ecostress_filter_viz"))
    ap.add_argument("--floor", type=int, default=20, help="§36.16 pairs-per-station floor")
    ap.add_argument("--dpi", type=int, default=300, help="raster resolution")
    ap.add_argument("--formats", default="png,pdf",
                    help="output formats; pdf is the vector one for the thesis")
    ap.add_argument("--emit-json", action="store_true")
    args = ap.parse_args()

    global FORMATS
    FORMATS = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    style(dpi=args.dpi)
    print(f"style: scienceplots science+no-latex, {args.dpi} dpi, formats {FORMATS}")

    K = load_constants()
    print("constants parsed from census_ecostress.py:")
    for k in ("CROSSOVER_LO", "CROSSOVER_HI", "CLEAR_FRAC_MIN", "WINDOW_FRAC_MIN",
              "N_PX_EXPECTED", "LAT_LIMIT", "THERMAL_PEAK_LAG_H", "WELL_PHASED_DAY_H",
              "WELL_PHASED_NIGHT_TST"):
        print(f"    {k:22s} = {K[k]}")

    print(f"\nloading census CSVs ...")
    gran, pair, logd = load_data(stations)
    print(f"  granules {len(gran):,}   pairs {len(pair):,}   log {len(logd):,}")

    print("\nreconciling funnels against the census log:")
    funnels, coincid = {}, {}
    for sid in stations:
        funnels[sid] = funnel(sid, gran, pair, logd)
        coincid[sid] = date_coincidence(sid, gran)
        verify_well_phased(sid, gran, pair, K)
        c = coincid[sid]
        print(f"    dates {c['n_dates']} = both {c['both']} + day-only {c['day_only']}"
              f" + night-only {c['night_only']};  same-date cap {c['cap_same_date']}")

    print("\ndrawing ...")
    fig_flowchart(K, None, outdir)
    fig_clear_sweep(stations, gran, K, outdir)
    fig_network(stations, gran, logd, outdir, floor=args.floor)
    for sid in stations:
        f = funnels[sid]
        fig_flowchart(K, f, outdir, tag=f"_{sid}")
        fig_waterfall(f, K, outdir)
        fig_solar_timeseries(sid, gran, pair, f, K, outdir)
        fig_utc_to_solar(sid, gran, f, K, outdir)
        fig_solar_geometry(sid, gran, f, K, outdir, color_by_clear=False)
        fig_solar_geometry(sid, gran, f, K, outdir, color_by_clear=True)
        fig_qc_attribution(sid, gran, K, outdir)
        fig_pairing(sid, pair, f, K, outdir)
        fig_pairing_loss(sid, gran, f, coincid[sid], outdir)
        fig_timeline(sid, gran, pair, outdir)
        print(f"  {sid} done")

    if args.emit_json:
        emit_json(stations, gran, pair, funnels, coincid, K, outdir)

    print(f"\nall figures in {outdir}")


if __name__ == "__main__":
    main()
