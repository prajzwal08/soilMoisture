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
    """Load the §48 SoilMoistureModel from a checkpoint, strictly.

    Only checkpoints stamped arch == "s48" are accepted. Every earlier architecture shares
    key prefixes with this one (the U-Net had `decoder.*` and `transformer_layers.*` too), so
    a key-prefix test cannot tell them apart; the stamp can. Older checkpoints load from
    their own code: tag `baseline-unet-temporal` (ckpt_utils_unet.py) for the pooled U-Net,
    tags `pw_stage2a-ep9` / `pre-s48-build` for the patchwise arm.
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    arch = cfg.get("arch")
    if arch != "s48":
        raise RuntimeError(
            f"{ckpt_path} is arch={arch!r}, not 's48'. This module builds only the §48 model. "
            f"Pooled U-Net: ckpt_utils_unet.load_checkpoint (tag baseline-unet-temporal). "
            f"Patchwise: check out tag pre-s48-build.")

    # Normalisation provenance (§35.28), extended to the §48 contracts. Each hash is checked
    # against the file the CHECKPOINT names, not a hardcoded filename: the old check hashed
    # csvs/era5_stats.json while training read era5_stats18.json, so a correct checkpoint
    # always reported MISMATCH and the warning trained everyone to ignore it.
    #
    # Warn rather than raise: an old or moved file should stay loadable for inspection.
    import hashlib as _hl
    _root = Path(__file__).resolve().parent / "csvs"
    _files = {
        "era5_stats":   cfg.get("era5_stats"),
        "driver_stats": cfg.get("driver_stats"),
        "fine_stats":   cfg.get("fine_stats"),
        "lst_stats":    cfg.get("lst_stats"),
    }
    for _name, _path in _files.items():
        _want = cfg.get(f"{_name}_sha")
        if not _want or _want == "unknown" or not _path:
            print(f"  [provenance] {ckpt_path.name} records no {_name}_sha/path — which "
                  f"constants it was trained with cannot be verified.")
            continue
        _p = Path(_path)
        if not _p.exists():
            _p = _root / _p.name
        try:
            _have = _hl.sha256(_p.read_bytes()).hexdigest()[:16]
        except Exception as _e:
            print(f"  [provenance] WARNING: cannot hash {_p} ({_e})")
            continue
        if _have != _want:
            print(f"  [provenance] *** MISMATCH *** {_name}: checkpoint trained with "
                  f"{_want}, {_p} on disk is {_have}. The model is being fed DIFFERENT "
                  f"constants than it was trained with; every number from this evaluation "
                  f"is suspect. Restore the matching file first.")

    model = SoilMoistureModel(
        n_depths         = cfg.get("n_depths", 3),
        d_model          = cfg.get("d_model",  768),
        n_heads          = cfg.get("n_heads",  12),
        n_layers         = cfg.get("n_layers", 6),
        drop_path_rate   = cfg.get("drop_path_rate", 0.0),
        fine_skips       = cfg.get("fine_skips", "cnn"),
        modality_dropout = cfg.get("modality_dropout", 0.2),
    ).to(device)

    # strict=True, deliberately: a mismatched checkpoint must never become a randomly
    # initialised model that runs and prints plausible numbers (§35.22).
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"  arch={arch}  epoch {ckpt['epoch']}  "
          f"best_val_loss={ckpt.get('best_val_loss','N/A')}  sha={cfg.get('git_sha','?')[:8]}")
    return model, cfg, ckpt["epoch"]
