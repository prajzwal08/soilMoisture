"""Publication styles for the evaluation figures (§66, user 2026-10-05).

Every eval plot script keeps its colour house style by default; `--style bw|paper` calls
`apply(globals(), style)` at the top of main(), which swaps the script's module-level tables
(colours, hatches, line styles, font-size multiplier, tick rotation, dpi) and sets rcParams.

  paper  (the user's choice): scientific style, ONLY blue / green / red, Times-like serif text,
         larger annotation text, crowded category ticks rotated 90 deg, 600 dpi, editable PDF text.
         Splits: OOS blue, OOT green, OOST red.
  bw     black-and-white: grey levels + hatches (splits), line styles + markers (depths).

Tables a script may define (only names it uses matter): DEPTH_COLORS, SPLIT_COLORS, SPLIT_HATCH,
SPLIT_LS, SPLIT_MARKER, HEX_CMAP, PRED_COLOR, OBS_COLOR, OOT_SHADE, BW, FS, XROT, DPI.
"""
import matplotlib.pyplot as plt

# Option H (user pick 2026-10-05, replaces Nature/NPG): dark blue / light blue / orange.
# Colour-blind safe (dataviz validate_palette.js: worst CVD dE 22.3 deutan, normal-vision dE 28.8).
# BLUE=OOS (dark blue), GREEN=OOT (light blue), RED=OOST (orange).
BLUE, GREEN, RED = "#08519C", "#6BAED6", "#E6550D"

GREYS = {"oos": "0.15", "oot": "0.45", "oost": "0.70", "val": "0.88", "train": "0.30"}
HATCH = {"oos": "", "oot": "////", "oost": "....", "val": "xxxx", "train": "\\\\\\\\"}
SPLIT_LINESTYLE = {"oos": "-", "oot": "--", "oost": ":", "val": "-.", "train": "-"}
SPLIT_MARKERS = {"oos": "o", "oot": "s", "oost": "^", "val": "D", "train": "v"}
DEPTH_LINESTYLE = {"0-10": "-", "10-30": "--", "30-100": ":"}
DEPTH_MARKERS = {"0-10": "o", "10-30": "s", "30-100": "^"}

_COMMON = {
    "axes.edgecolor": "black", "axes.labelcolor": "black", "axes.linewidth": 0.8,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "text.usetex": False, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
}
RC_BW = {
    **_COMMON,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.prop_cycle": plt.cycler(color=["black"]),
    "lines.linewidth": 1.0, "hatch.linewidth": 0.5, "hatch.color": "black", "image.cmap": "Greys",
}
RC_PAPER = {
    **_COMMON,
    # Times New Roman is not installed on Snellius; Nimbus Roman is its metric-identical clone,
    # STIX matches it for maths and is the fallback shipped with matplotlib.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
    "figure.titlesize": 13,
    "axes.prop_cycle": plt.cycler(color=[BLUE, GREEN, RED]),
    "lines.linewidth": 1.3, "image.cmap": "Blues",
    # bold text everywhere: axis labels, tick numbers, legends, annotations (user 2026-10-05)
    "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
    "figure.titleweight": "bold", "mathtext.default": "bf", "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
}


def apply(g: dict, style: str = "bw") -> None:
    """Swap the calling script's tables (its globals()) for the chosen publication style."""
    if style == "paper":
        plt.rcParams.update(RC_PAPER)
        if "DEPTH_COLORS" in g:
            g["DEPTH_COLORS"] = {d: "black" for d in g["DEPTH_COLORS"]}   # titles/labels in black
        if "SPLIT_COLORS" in g:
            pal = {"oos": BLUE, "oot": GREEN, "oost": RED, "val": "0.55", "train": "0.30"}
            g["SPLIT_COLORS"] = {s: pal.get(s, "0.5") for s in g["SPLIT_COLORS"]}
        g["SPLIT_HATCH"] = {}
        g["SPLIT_LS"] = dict(SPLIT_LINESTYLE)
        g["SPLIT_MARKER"] = dict(SPLIT_MARKERS)
        g["HEX_CMAP"] = "Blues"
        # time-series prediction line: Python-logo blue #3776AB, solid (user 2026-10-05)
        g["PRED_COLOR"], g["OBS_COLOR"], g["OOT_SHADE"] = "#3776AB", "black", RED
        g["BW"] = False
        g["FS"], g["XROT"], g["DPI"] = 1.5, 90, 600
        g["PAPER"], g["CS"], g["BOX_ALPHA"] = True, 1.45, 1.0
        return
    plt.rcParams.update(RC_BW)
    if "DEPTH_COLORS" in g:
        g["DEPTH_COLORS"] = {d: "black" for d in g["DEPTH_COLORS"]}
    if "SPLIT_COLORS" in g:
        g["SPLIT_COLORS"] = {s: GREYS.get(s, "0.5") for s in g["SPLIT_COLORS"]}
    g["SPLIT_HATCH"] = dict(HATCH)
    g["SPLIT_LS"] = dict(SPLIT_LINESTYLE)
    g["SPLIT_MARKER"] = dict(SPLIT_MARKERS)
    g["DEPTH_LS"] = dict(DEPTH_LINESTYLE)
    g["DEPTH_MARKER"] = dict(DEPTH_MARKERS)
    g["HEX_CMAP"] = "Greys"
    g["PRED_COLOR"], g["OBS_COLOR"], g["OOT_SHADE"] = "black", "0.55", "0.6"
    g["BW"] = True
    g["FS"], g["XROT"], g["DPI"] = 1.0, None, 600
    g["BOX_ALPHA"] = 1.0
