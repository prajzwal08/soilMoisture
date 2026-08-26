"""Checkpoint compatibility utilities."""
import torch
from pathlib import Path
from model import SoilMoistureModel


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
        use_cls_depth  = cfg.get("use_cls_depth", True),
        driver_mode    = cfg.get("driver_mode", "memory"),
        driver_layers  = cfg.get("driver_layers", 2),
    ).to(device)

    # strict=True, deliberately. The old loader used strict=False with a tolerance written for
    # one pre-2026-06 checkpoint, and the cost was that a mismatched checkpoint produced a
    # RANDOMLY-INITIALISED model that ran and printed entirely plausible numbers. Every eval
    # script and figure in this project funnels through this function (§35.22).
    if cfg.get("arch") == "unet" or any(k.startswith(("decoder.", "transformer_layers."))
                                        for k in ckpt["model"]):
        raise RuntimeError(
            f"{ckpt_path} holds a POOLED U-NET checkpoint. This module only builds the "
            "patchwise architecture. Load it with ckpt_utils_unet.load_checkpoint instead "
            "(model_unet.py / dataset_unet.py, tag baseline-unet-temporal)."
        )
    model.load_state_dict(remap_checkpoint_keys(ckpt["model"]), strict=True)

    model.eval()
    print(f"  arch={arch}  epoch {ckpt['epoch']}  "
          f"best_val_loss={ckpt.get('best_val_loss','N/A')}  sha={cfg.get('git_sha','?')[:8]}")
    return model, cfg, ckpt["epoch"]
