"""CPU smoke for the 160 m tile-map path (plot_tile_sm_map.py). Run before its GPU job.

Asserts the three things that path depends on and that nothing else exercises:
  1. pixel -> token mapping equals the runbook's (105, 62, 100, 20, 44, 172)
  2. the dataset accepts token_sel='all' and emits K=196 per-patch tensors
  3. the checkpoint's model returns (B, 196, n_depths) for that input -- i.e. the patch
     blocks really are patch-agnostic, so a model trained on K=1 scores all 196

Usage:  sbatch slurm/smoke_tilemap.sh
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SoilMoistureDataset          # noqa: E402
from ckpt_utils import load_checkpoint           # noqa: E402
import plot_tile_sm_map as M                     # noqa: E402

TILE = "ISMN_TxSON_CR200-18"
# (row, col) of the six stations inside that tile, from csvs/txson_readouts.csv
PIX = [(112, 112), (72, 105), (114, 43), (25, 109), (62, 33), (193, 65)]
WANT = [105, 62, 100, 20, 44, 172]


def main():
    got = [M.token_of(r, c) for r, c in PIX]
    print(f"token indices: {got}")
    assert got == WANT, f"token mapping {got} disagrees with the runbook {WANT}"
    print("PASS  token mapping matches the runbook")

    model, cfg, ep = load_checkpoint(
        M.CKPT_ROOT / "pw_stage2a_L3" / "best.pt", torch.device("cpu"))
    print(f"      checkpoint epoch {ep}, trained token_sel={cfg.get('token_sel')!r}")

    splits = pd.read_csv(M.SPLITS)
    key = splits.apply(
        lambda r: (f"ISMN_{r['network']}_{r['station_name']}"
                   if str(r["source_network"]) == "ISMN"
                   else f"{r['source_network']}_{r['station_id']}"), axis=1)
    sub = splits[key == TILE]
    assert not sub.empty, f"{TILE} not in station_splits.csv"
    split_name = str(sub.iloc[0]["split"])
    tmp = Path("_smoke_tile.csv")
    sub.to_csv(tmp, index=False)
    print(f"      {TILE} split={split_name}")

    try:
        ds = SoilMoistureDataset(
            splits_csv=str(tmp), era5_stats_path=str(M.ERA5_STATS), years=[2019],
            category_filter=cfg.get("category_filter", ["sm_only"]),
            split_filter=[split_name], training=False,
            token_sel="all", shm_dir=None)
        assert len(ds) > 0, "token_sel='all' produced no samples"
        print(f"PASS  dataset accepts token_sel='all' — {len(ds)} dates")

        batch = torch.utils.data.default_collate([ds[0], ds[1]])
        print(f"      dem_tok {tuple(batch['dem_tok'].shape)}  "
              f"s2_hist {tuple(batch['s2_hist'].shape)}")
        assert batch["dem_tok"].shape[1] == 196, \
            f"expected K=196, got {batch['dem_tok'].shape[1]}"

        with torch.no_grad():
            mu = model(batch)
        print(f"      model output {tuple(mu.shape)}")
        assert mu.ndim == 3 and mu.shape[1] == 196, \
            f"expected (B,196,D), got {tuple(mu.shape)}"
        g = mu[0, :, 0].numpy().reshape(14, 14)
        print(f"PASS  14x14 map — range {g.min():.4f}–{g.max():.4f}  "
              f"spread {g.max() - g.min():.4f}")
        print("      at the six station tokens: "
              + "  ".join(f"{mu[0, t, 0].item():.4f}" for t in WANT))
    finally:
        tmp.unlink(missing_ok=True)

    print("\nALL SMOKE CHECKS PASSED — safe to submit slurm/tile_sm_map.sh")


if __name__ == "__main__":
    main()
