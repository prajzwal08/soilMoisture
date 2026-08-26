"""FROZEN BASELINE SNAPSHOT — do not edit.

The pooled U-Net-temporal architecture as of 2026-08-26, kept so the `cls_depth_star_reg`
checkpoint can still be loaded and run after model.py / dataset.py / train.py were made
patchwise-only. Imports are repointed at the _unet snapshots so this set is self-contained
and does not depend on the live modules. Git tag: baseline-unet-temporal.
"""
"""Checkpoint compatibility utilities."""
import torch
from pathlib import Path
from model_unet import SoilMoistureModel


def remap_checkpoint_keys(state_dict: dict) -> dict:
    """Map keys from pre-refactor checkpoints to the current model architecture.

    Two breaking changes:
      1. transformer.layers.X.*  →  transformer_layers.X.layer.*
         (nn.TransformerEncoder → ModuleList[DropPathTransformerLayer])
      2. {era5,sif,twsa}_mlp.2.* → {era5,sif,twsa}_mlp.3.*
         (3-layer Sequential → 4-layer with Dropout at index 2)
    """
    new_sd = {}
    for k, v in state_dict.items():
        # Transformer key rename
        if k.startswith("transformer.layers."):
            rest   = k[len("transformer.layers."):]
            idx, _, field = rest.partition(".")
            k = f"transformer_layers.{idx}.layer.{field}"
        # MLP index shift (Dropout inserted at position 2)
        for prefix in ("era5_mlp", "sif_mlp", "twsa_mlp"):
            old_pfx = f"{prefix}.2."
            if k.startswith(old_pfx):
                k = f"{prefix}.3." + k[len(old_pfx):]
                break
        new_sd[k] = v
    return new_sd


def load_checkpoint(ckpt_path: Path, device):
    """Load SoilMoistureModel from checkpoint, remapping legacy key names."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = SoilMoistureModel(
        n_depths       = cfg.get("n_depths", 3),
        d_model        = cfg.get("d_model",  768),
        n_heads        = cfg.get("n_heads",  12),
        n_layers       = cfg.get("n_layers", 6),
        drop_path_rate = cfg.get("drop_path_rate", 0.0),
        use_cls_depth  = cfg.get("use_cls_depth", False),
    ).to(device)

    sd      = remap_checkpoint_keys(ckpt["model"])
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # transformer_norm is new (old model had no post-transformer LayerNorm);
    # it initialises as identity so eval results are valid.
    new_keys = {"transformer_norm.weight", "transformer_norm.bias"}
    real_missing = [k for k in missing if k not in new_keys]
    if real_missing:
        print(f"  WARNING — unexpected missing keys: {real_missing[:6]} ...")
    if unexpected:
        print(f"  WARNING — unexpected keys in ckpt: {unexpected[:4]} ...")

    model.eval()
    print(f"  Epoch {ckpt['epoch']}  best_val_loss={ckpt.get('best_val_loss','N/A')}")
    return model, cfg, ckpt["epoch"]
