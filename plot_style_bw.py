"""Black-and-white publication style for the evaluation figures (§66, user 2026-10-05).

Every eval plot script keeps its colour style by default; `--style bw` calls `apply(globals())`
at the top of main(), which swaps the script's module-level colour tables for grey + hatch +
line-style encodings and sets one shared rcParams block. Identity is never carried by colour
alone: splits get a grey level AND a hatch, depths get a line style AND a marker.

Tables a script may define (only those present are replaced):
    DEPTH_COLORS  -> all black (titles, ticks, lines)
    SPLIT_COLORS  -> grey levels, darkest = the headline split (OOS)
    SPLIT_HATCH   -> hatches for filled boxes / bars
    SPLIT_LS      -> line styles for split curves
    DEPTH_LS, DEPTH_MARKER
    HEX_CMAP      -> "Greys" for density plots
    PRED_COLOR, OBS_COLOR, OOT_SHADE -> time-series encodings
"""
import matplotlib.pyplot as plt

GREYS = {"oos": "0.15", "oot": "0.45", "oost": "0.70", "val": "0.88", "train": "0.30"}
HATCH = {"oos": "", "oot": "////", "oost": "....", "val": "xxxx", "train": "\\\\\\\\"}
SPLIT_LINESTYLE = {"oos": "-", "oot": "--", "oost": ":", "val": "-.", "train": "-"}
SPLIT_MARKERS = {"oos": "o", "oot": "s", "oost": "^", "val": "D", "train": "v"}
DEPTH_LINESTYLE ={"0-10": "-", "10-30": "--", "30-100": ":"}
DEPTH_MARKERS = {"0-10": "o", "10-30": "s", "30-100": "^"}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": "black", "axes.labelcolor": "black", "axes.linewidth": 0.6,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.prop_cycle": plt.cycler(color=["black"]),
    "lines.linewidth": 1.0, "hatch.linewidth": 0.5, "hatch.color": "black",
    "image.cmap": "Greys", "text.usetex": False,
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in Illustrator
}


def apply(g: dict) -> None:
    """Swap the calling script's colour tables (its globals()) for BW encodings."""
    plt.rcParams.update(RC)
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
    g["PRED_COLOR"] = "black"
    g["OBS_COLOR"] = "0.55"
    g["OOT_SHADE"] = "0.6"
    g["BW"] = True
