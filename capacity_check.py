"""
Step 0 capacity check (runbook §16.4) — CAN the decoder paint spatial structure?

Tier-1 verdict was DECODER-SIDE: the 224² SM map is flat because supervision is a
single pixel (112,112), not because the conv decoder lacks the capacity to produce
structure. This script tests that directly: freeze everything up to the bottleneck,
then overfit ONLY the decoder for a few hundred steps against a *dense* synthetic
structured target (all 224² pixels supervised). If the output goes from flat →
structured and correlates with the target, decoder capacity is fine and supervision
is confirmed as the lever (green-light Step 1 proxy-consistency loss). If it stays
flat, the bilinear layers themselves are limiting (do Step 2 architecture first).

The target is a deterministic synthetic field (sinusoids + Gaussian bumps) — no raw
store / band-index dependencies, so a scheduled unattended run can't fail on inputs.

Usage:
    python capacity_check.py [--checkpoint PATH] [--station NAME] [--steps 400] [--lr 1e-3]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ckpt_utils import load_checkpoint
from dataset import SoilMoistureDataset
from tier1_probe import (
    DEFAULTS, patch_open_zarr_no_marker, subset_splits_csv, pick_sample,
)


def synthetic_target(H: int = 224) -> np.ndarray:
    """Deterministic structured field in ~SM range [0.1, 0.4]. No randomness."""
    yy, xx = np.mgrid[0:H, 0:H] / float(H)
    f = (np.sin(2 * np.pi * 2 * xx) * np.cos(2 * np.pi * 3 * yy)
         + 0.6 * np.exp(-((xx - 0.30) ** 2 + (yy - 0.60) ** 2) / 0.02)
         - 0.5 * np.exp(-((xx - 0.70) ** 2 + (yy - 0.30) ** 2) / 0.03))
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    return (0.1 + 0.3 * f).astype(np.float32)


def norm_std(a: np.ndarray) -> float:
    return float(a.std() / (abs(a.mean()) + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    ap.add_argument("--station", default="ISMN_Berlin_PSA7Ruebezahl")  # S2 anchor in Tier-1
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="capacity_output")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  station: {args.station}  steps: {args.steps}")

    model, cfg, epoch = load_checkpoint(Path(args.checkpoint), device)

    patch_open_zarr_no_marker()
    subset_csv = subset_splits_csv(DEFAULTS["splits_csv"], [args.station],
                                   out_dir / "_subset_splits.csv")
    ds = SoilMoistureDataset(
        splits_csv      = str(subset_csv),
        era5_stats_path = DEFAULTS["era5_stats"],
        years           = list(range(2016, 2024)),
        category_filter = ["sm_only"],
        split_filter    = ["val", "oos"],
        training        = False,
    )
    idx = pick_sample(ds, args.station)
    if idx is None:
        print(f"No sample for {args.station} — aborting."); return

    raw = ds[idx]
    batch = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in raw.items()}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            batch[k] = v.float()

    target = torch.from_numpy(synthetic_target()).to(device)          # (224,224)

    # freeze everything except the decoder
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.eval()                                                       # no dropout noise
    opt = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

    with torch.no_grad():
        before = model(batch)[0, 0].float().cpu().numpy()             # (224,224) flat baseline

    losses = []
    for step in range(args.steps):
        opt.zero_grad()
        pred = model(batch)[:, 0]                                      # (1,224,224) depth 0
        loss = torch.mean((pred[0] - target) ** 2)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss={loss.item():.5f}")

    with torch.no_grad():
        after = model(batch)[0, 0].float().cpu().numpy()

    tgt = target.cpu().numpy()
    corr = float(np.corrcoef(after.ravel(), tgt.ravel())[0, 1])
    ns_before, ns_after, ns_tgt = norm_std(before), norm_std(after), norm_std(tgt)

    capable = (ns_after > 5 * ns_before) and (corr > 0.7)
    verdict = ("CAPACITY CONFIRMED — decoder CAN paint structure under dense supervision; "
               "the flat maps are a SUPERVISION problem. Green-light Step 1 (proxy-consistency loss)."
               if capable else
               "CAPACITY LIMITED — decoder stays flat even with a dense target; the bilinear "
               "layers may be the bottleneck. Do Step 2 (learned upsampling) first.")

    print(f"\n── Capacity check ──")
    print(f"  output norm-std:  before={ns_before:.4f}  after={ns_after:.4f}  target={ns_tgt:.4f}")
    print(f"  corr(after,target)= {corr:.3f}   final loss={losses[-1]:.5f}")
    print(f"  VERDICT: {verdict}\n")

    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for a, img, ttl in zip(
        ax[:3],
        [tgt, before, after],
        [f"Synthetic target\nn-std={ns_tgt:.3f}",
         f"Output BEFORE\nn-std={ns_before:.4f}",
         f"Output AFTER {args.steps} steps\nn-std={ns_after:.4f} corr={corr:.2f}"]):
        im = a.imshow(img, cmap="YlOrBr_r", vmin=0.0, vmax=0.5)
        a.set_title(ttl, fontsize=10); a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    ax[3].plot(losses); ax[3].set_title("dense MSE loss"); ax[3].set_xlabel("step")
    fig.suptitle(f"Step 0 capacity check — {args.station}  (epoch {epoch})  |  {verdict[:60]}...",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "capacity_check.png", dpi=130, bbox_inches="tight")
    print(f"Saved: {out_dir / 'capacity_check.png'}")


if __name__ == "__main__":
    main()
