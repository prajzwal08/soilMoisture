"""
Tier 1 probe — where does the spatially-rich L12 signal get flattened?

Tier 0 showed the frozen L12 anchor tokens are spatially structured, yet the
model's 224x224 SM map is smooth. This script loads the TRAINED downstream
weights (temporal transformer + U-Net decoder) and renders the FULL spatial
progression on ONE forward pass per station, so the collapse can be located:

    raw scene (S2 RGB / S1 VV)          the imagery the anchor tokens came from
      -> L12 anchor tokens [196,768]    (batch["anchor_l12"])       PCA-RGB 14x14
      -> temporal transformer + norm
      -> 196 spatial tokens reshaped = BOTTLENECK [768,14,14]       PCA-RGB 14x14
         (== pre-hook input to decoder.bottle_proj)
      -> U-Net decoder conv1/2/3       feature maps 28 / 56 / 112   channel-mean
      -> SM map [n_depths,224,224]      (model forward return)       224x224

The "transformer output" and "bottleneck" taps are the SAME tensor (the spatial
slice reshaped), so the genuine taps are: L12 in -> bottleneck -> decoder stages
-> output. This separates the ATTENTION side (L12->bottleneck) from the DECODER
side (bottleneck->output) both visually (the maps) and quantitatively (cos/disp).

Decision (checklist 1.4):
  * cos jumps L12->bottleneck AND spatial dispersion drops -> attention homogenises (upstream)
  * bottleneck stays structured but decoder maps flatten    -> decoder smooths (single-pixel loss)

Reuses Tier-0 helpers from visualize_embeddings.py and the loader from ckpt_utils.py.
TerraMind is NOT needed (tokens cached). Builds the dataset on only the 4 target
stations (not all ~255) via a subset splits CSV.

Usage:
    python tier1_probe.py [--checkpoint PATH] [--out tier1_output]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zarr

import dataset as _dataset
from dataset import SoilMoistureDataset, SM_DEPTHS
from ckpt_utils import load_checkpoint
from visualize_embeddings import (
    pca_rgb, panel_metrics, open_tokens, open_raw, _dstr,
    raw_s2_rgb, raw_s1_gray, build_disk_index,
)

# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "splits_csv" : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv",
    "era5_stats" : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json",
    "checkpoint" : "/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/"
                   "phase1_sm_only/baseline_huber_memmap/best.pt",
}

# Same four stations as the Tier-0 figures.
TIER0_STATIONS = [
    "ISMN_Berlin_DWDBerlin-Spaeth",
    "ISMN_Berlin_PSA7Ruebezahl",
    "ISMN_COSMOS-UK_Balruddery",
    "ISMN_CW3E_YucaipaValley",
]

TARGET_DOY = 180                                    # in-window sample nearest mid-summer
ORBIT_NAME = {0: "s2", 1: "s1_asc", 2: "s1_desc"}   # anchor_orbit id -> token group


def patch_open_zarr_no_marker():
    """The 4 Tier-0 stations are held-out (val/oos) and lack the `.complete`
    marker that dataset._open_zarr requires, though every required group
    (era5/labels/soil/dem/lulc/s2/l12) is present and Tier-0 read them directly.
    Bypass ONLY the marker gate — no data is modified, all other logic unchanged."""
    def _open(station_dir, category):
        path = _dataset.ZARR_ROOT / category / station_dir.name
        if not path.exists():
            return None
        try:
            return zarr.open_consolidated(str(path), mode="r")
        except KeyError:
            return zarr.open_group(str(path), mode="r")
    _dataset._open_zarr = _open
    print("  Patched _open_zarr to bypass .complete marker (held-out stores).")


def _dir_name(r) -> str:
    """Reproduce dataset.py's station dir-name / station_key from a splits row."""
    if str(r.get("source_network", "")) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def subset_splits_csv(splits_csv: str, stations: list, out_path: Path) -> Path:
    """Write a splits CSV with only `stations` so the dataset builds on 4 stations
    (not ~255). Matches dataset.py's dir-name convention."""
    df = pd.read_csv(splits_csv)
    df["_dir"] = df.apply(_dir_name, axis=1)
    sub   = df[df["_dir"].isin(stations)]
    found = sorted(set(sub["_dir"]))
    missing = [s for s in stations if s not in found]
    if missing:
        print(f"  WARNING — stations not in splits CSV: {missing}")
    sub.drop(columns=["_dir"]).to_csv(out_path, index=False)
    print(f"  Subset splits CSV ({len(sub)} rows, {len(found)} stations) -> {out_path}")
    return out_path


# ── metrics ──────────────────────────────────────────────────────────────────

def rel_spatial_std(tok: np.ndarray) -> float:
    """Scale-free spatial dispersion of a (196,D) token grid: per-feature std
    across the 196 cells / per-feature magnitude, averaged. ~0 = spatially flat."""
    x = tok.astype(np.float32)
    return float((x.std(axis=0) / (np.abs(x).mean(axis=0) + 1e-8)).mean())


def norm_std(field: np.ndarray) -> float:
    """Normalised spatial std of a 2-D map: std / |mean|."""
    return float(field.std() / (abs(field.mean()) + 1e-8))


def token_metrics(tok: np.ndarray) -> dict:
    pm = panel_metrics(tok)
    _, var_ratio = pca_rgb(tok)
    return {
        "offdiag_cos"    : pm["offdiag_cos"],
        "neighbor_ac"    : pm["neighbor_ac"],
        "pc1_var_ratio"  : float(var_ratio[0]),
        "rel_spatial_std": rel_spatial_std(tok),
    }


# ── raw scene recovery ───────────────────────────────────────────────────────

def find_anchor_scene(name, cat, orbit_id, l12_np):
    """Recover the raw imagery the anchor tokens came from by matching anchor_l12
    back to the token store, then render S2 RGB (or S1 VV). Returns (img, label)."""
    if not np.any(l12_np):                          # all-zero anchor => none found
        return None, "no anchor"
    orbit = ORBIT_NAME[int(orbit_id)]
    try:
        g = open_tokens(name, cat)
        l12_all = np.asarray(g[f"{orbit}/l12"][:]).astype(np.float32)   # (N,196,768)
        diffs = np.abs(l12_all - l12_np[None].astype(np.float32)).reshape(len(l12_all), -1).mean(1)
        idx = int(diffs.argmin())
        date = str(_dstr(g[f"{orbit}/dates"][:])[idx])
        raw = open_raw(name)
        if raw is None:
            return None, f"{orbit} {date} (no raw)"
        if orbit == "s2":
            img = raw_s2_rgb(raw, date)
        else:
            img = raw_s1_gray(raw, orbit.split("_")[1], date)          # asc / desc
        return img, f"{orbit.upper()} {date}"
    except Exception as e:
        return None, f"{orbit} (err: {type(e).__name__})"


# ── sample selection ─────────────────────────────────────────────────────────

def pick_sample(ds, station_key: str):
    idxs = [i for i, s in enumerate(ds.samples) if s["station_key"] == station_key]
    if not idxs:
        return None
    year = max(ds.samples[i]["year"] for i in idxs)
    idxs = [i for i in idxs if ds.samples[i]["year"] == year]
    return min(idxs, key=lambda i: abs(ds.samples[i]["doy"] - TARGET_DOY))


# ── probe one station ────────────────────────────────────────────────────────

@torch.no_grad()
def probe_station(model, ds, idx, taps, disk_idx, device):
    s     = ds.samples[idx]
    name  = s["station_key"]
    batch = ds[idx]

    def _prep(v):
        if not isinstance(v, torch.Tensor):
            return v
        v = v.unsqueeze(0).to(device)
        return v.float() if v.is_floating_point() else v   # fp16 tokens -> fp32 to match weights

    batch = {k: _prep(v) for k, v in batch.items()}

    taps.clear()
    sm_map = model(batch)                                    # (1,n_depths,224,224)

    l12_np = batch["anchor_l12"][0].float().cpu().numpy()    # (196,768)  tap 1
    bott   = taps["bottleneck"][0].float().cpu().numpy()     # (768,14,14) tap 2
    tout   = bott.reshape(bott.shape[0], -1).T               # (196,768) post-transformer
    out    = sm_map[0].float().cpu().numpy()                 # (n_depths,224,224) tap 3

    # decoder-stage channel-mean spatial maps (28/56/112)
    dec = {k: taps[k][0].float().cpu().numpy().mean(0) for k in ("conv1", "conv2", "conv3")}

    # raw scene the anchor came from
    cat = disk_idx.get(name, "sm_only")
    scene_img, scene_lbl = find_anchor_scene(name, cat, batch["anchor_orbit"][0].item(), l12_np)

    return {
        "station"      : name,
        "year"         : int(s["year"]),
        "doy"          : int(s["doy"]),
        "l12"          : token_metrics(l12_np),
        "bottleneck"   : token_metrics(tout),
        "dec_nstd"     : {k: norm_std(v) for k, v in dec.items()},
        "out_nstd"     : [norm_std(out[d]) for d in range(out.shape[0])],
        "scene_label"  : scene_lbl,
        # figure-only arrays (not serialised)
        "_scene"       : scene_img,
        "_l12_rgb"     : pca_rgb(l12_np)[0],
        "_bott_rgb"    : pca_rgb(tout)[0],
        "_dec"         : dec,
        "_out"         : out[0],
    }


# ── figure ───────────────────────────────────────────────────────────────────

COLS = ["Raw scene", "L12 in", "Transformer out", "Decoder 28²",
        "Decoder 56²", "Decoder 112²", "Output SM 224² (0-10 cm)"]


def make_figure(results, out_path, epoch):
    n = len(results)
    fig, axes = plt.subplots(n, len(COLS), figsize=(3.1 * len(COLS), 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    fig.suptitle(f"Tier 1 — spatial progression  |  baseline_huber_memmap epoch {epoch}",
                 fontsize=13, fontweight="bold", y=1.0)

    for row, r in enumerate(results):
        a = axes[row]

        # col 0: raw scene
        if r["_scene"] is not None:
            a[0].imshow(r["_scene"], interpolation="nearest")
        else:
            a[0].text(0.5, 0.5, "n/a", ha="center", va="center")
        a[0].set_ylabel(r["station"].replace("ISMN_", ""), fontsize=8)
        a[0].set_title(r["scene_label"], fontsize=7)

        # col 1: L12 PCA-RGB
        a[1].imshow(r["_l12_rgb"], interpolation="nearest")
        a[1].set_title(f"cos={r['l12']['offdiag_cos']:.3f} disp={r['l12']['rel_spatial_std']:.2f}",
                       fontsize=7)

        # col 2: bottleneck PCA-RGB
        a[2].imshow(r["_bott_rgb"], interpolation="nearest")
        a[2].set_title(f"cos={r['bottleneck']['offdiag_cos']:.3f} "
                       f"disp={r['bottleneck']['rel_spatial_std']:.2f}", fontsize=7)

        # cols 3-5: decoder stage maps
        for j, k in enumerate(("conv1", "conv2", "conv3"), start=3):
            im = a[j].imshow(r["_dec"][k], cmap="viridis", interpolation="nearest")
            a[j].set_title(f"n-std={r['dec_nstd'][k]:.3f}", fontsize=7)
            plt.colorbar(im, ax=a[j], fraction=0.046, pad=0.04)

        # col 6: output SM map
        im = a[6].imshow(r["_out"], cmap="YlOrBr_r", vmin=0.0, vmax=0.5, interpolation="nearest")
        a[6].set_title(f"n-std={r['out_nstd'][0]:.3f}", fontsize=7)
        plt.colorbar(im, ax=a[6], fraction=0.046, pad=0.04)

        for j in range(len(COLS)):
            a[j].set_xticks([]); a[j].set_yticks([])

    for j, t in enumerate(COLS):
        axes[0, j].annotate(t, xy=(0.5, 1.22), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── verdict (checklist 1.4) ──────────────────────────────────────────────────

def verdict(results):
    def mean(g): return float(np.mean([g(r) for r in results]))
    l12_cos  = mean(lambda r: r["l12"]["offdiag_cos"])
    bot_cos  = mean(lambda r: r["bottleneck"]["offdiag_cos"])
    l12_disp = mean(lambda r: r["l12"]["rel_spatial_std"])
    bot_disp = mean(lambda r: r["bottleneck"]["rel_spatial_std"])
    out_std  = mean(lambda r: r["out_nstd"][0])
    dec_std  = {k: mean(lambda r, k=k: r["dec_nstd"][k]) for k in ("conv1", "conv2", "conv3")}

    cos_jump   = bot_cos - l12_cos
    disp_ratio = bot_disp / (l12_disp + 1e-8)

    lines = [
        "── Tier 1 net call (1.4) ──",
        f"  L12         : cos={l12_cos:.3f}  disp={l12_disp:.3f}",
        f"  Bottleneck  : cos={bot_cos:.3f}  disp={bot_disp:.3f}",
        f"  Decoder 28/56/112 n-std: "
        f"{dec_std['conv1']:.3f} / {dec_std['conv2']:.3f} / {dec_std['conv3']:.3f}",
        f"  Output map  : n-std={out_std:.4f}",
        f"  cos jump (L12->bottleneck)    = {cos_jump:+.3f}",
        f"  dispersion ratio (bott / L12) = {disp_ratio:.2f}",
    ]
    attention = (cos_jump > 0.15) or (disp_ratio < 0.5)
    if attention:
        call = ("ATTENTION-SIDE: the temporal transformer homogenises the tokens (cos rises / "
                "spatial dispersion collapses before the decoder). Fix upstream — spatial "
                "supervision / curb full self-attention averaging.")
    else:
        call = ("DECODER-SIDE: bottleneck keeps its spatial structure but the decoder maps "
                "flatten toward the output. The U-Net + single-pixel (112,112) loss never "
                "learns off-centre detail. Fix downstream — multi-pixel / auxiliary spatial loss.")
    lines.append(f"  VERDICT: {call}")
    return "\n".join(lines), call


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    ap.add_argument("--out", default="tier1_output")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg, epoch = load_checkpoint(Path(args.checkpoint), device)

    # build dataset on only the 4 target stations
    patch_open_zarr_no_marker()
    subset_csv = subset_splits_csv(DEFAULTS["splits_csv"], TIER0_STATIONS,
                                   out_dir / "_subset_splits.csv")
    print("Building dataset (4 target stations)...")
    ds = SoilMoistureDataset(
        splits_csv      = str(subset_csv),
        era5_stats_path = DEFAULTS["era5_stats"],
        years           = list(range(2016, 2024)),
        category_filter = ["sm_only"],
        split_filter    = ["val", "oos"],
        training        = False,
    )
    disk_idx = build_disk_index()

    # hooks: transformer output, bottleneck (decoder input), decoder stages
    taps = {}
    model.transformer_norm.register_forward_hook(
        lambda m, i, o: taps.__setitem__("ctx", o.detach()))
    model.decoder.bottle_proj.register_forward_pre_hook(
        lambda m, i: taps.__setitem__("bottleneck", i[0].detach()))
    for k in ("conv1", "conv2", "conv3"):
        getattr(model.decoder, k).register_forward_hook(
            lambda m, i, o, k=k: taps.__setitem__(k, o.detach()))

    results = []
    for st in TIER0_STATIONS:
        idx = pick_sample(ds, st)
        if idx is None:
            print(f"  !! {st}: no samples found — skipping")
            continue
        r = probe_station(model, ds, idx, taps, disk_idx, device)
        print(f"  {st}  y{r['year']} doy{r['doy']}  [{r['scene_label']}]  "
              f"L12 cos={r['l12']['offdiag_cos']:.3f} disp={r['l12']['rel_spatial_std']:.2f} -> "
              f"bott cos={r['bottleneck']['offdiag_cos']:.3f} disp={r['bottleneck']['rel_spatial_std']:.2f}  "
              f"out n-std={r['out_nstd'][0]:.4f}")
        results.append(r)

    if not results:
        print("No stations probed — aborting.")
        return

    fig_path = out_dir / "tier1_probe.png"
    make_figure(results, fig_path, epoch)
    print(f"Saved figure: {fig_path}")

    txt, call = verdict(results)
    print("\n" + txt + "\n")

    ser = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    (out_dir / "tier1_metrics.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "epoch": int(epoch), "stations": ser, "verdict": call},
        indent=2))
    print(f"Saved metrics: {out_dir / 'tier1_metrics.json'}")


if __name__ == "__main__":
    main()
