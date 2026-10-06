"""§70 effective resolution, GPU half: full 20 m maps for every TxSON tile, plain and with the
within-tile spatial inputs block-averaged (ablation.coarsen_item).

One dataset over the 40 TxSON tiles (the eval_predict `network` split: train+val+oos stations,
TRAIN_YEARS), then one pass per block size b (in 20 m map pixels):

    b = 0     plain pass. Full maps -> maps_L0.npy (N, 3, 112, 112) fp16 + index_L0.parquet
    b > 0     coarsened pass. Full maps are NOT kept (8 x 6.8 GB); per sample and depth it writes
              the map-wide RMS(map_b - map_0) and the spatial SD of both maps -> summary_L{b}

Every pass also writes the value at every station pixel of csvs/txson_readouts.csv
(map pixel = readout row // 2, col // 2) -> readouts_L{b}.parquet, the input of Tests A-C.

Usage
-----
    python effres_maps.py --blocks 0,1,2 --tiles ISMN_TxSON_CR200-18 --out-dir eval_output/effres_txson_smoke
    python effres_maps.py                    # full: blocks 0,2,4,8,14,28,56,112, all 40 tiles
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ablation import BlockAverageDataset
from ckpt_utils import load_checkpoint
from dataset import SM_DEPTHS, SoilMoistureDataset
from eval_predict import _git_sha, _make_key, worker_init_fn
from splits_config import SM_CATEGORIES, TRAIN_YEARS

REPO       = Path(__file__).resolve().parent
SPLITS     = REPO / "csvs" / "station_splits.csv"
READOUTS   = REPO / "csvs" / "txson_readouts.csv"
ERA5_STATS = REPO / "csvs" / "era5_stats18.json"
RUN        = "nolst_L3_wu200_20261005"
CKPT       = f"/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/{RUN}/best.pt"
MAP_PX, STRIDE = 112, 2


def readouts_frame(rd, ds, pix) -> pd.DataFrame:
    """[(sample idx, pixel rows of pix[tile], values (3, n_pix))] -> one long frame."""
    idx  = np.concatenate([np.full(v.shape[1], i) for i, _, v in rd])
    prow = np.concatenate([r for _, r, _ in rd])
    vals = np.concatenate([v for _, _, v in rd], axis=1)                    # (3, M)
    tile = np.array([ds.samples[i]["station_key"] for i in idx])
    stn  = {t: g["station"].to_numpy() for t, g in pix.items()}
    ctr  = {t: g["is_centre"].astype(bool).to_numpy() for t, g in pix.items()}
    station = [stn[t][r] for t, r in zip(tile, prow)]
    centre  = [ctr[t][r] for t, r in zip(tile, prow)]
    base = pd.DataFrame({"idx": idx, "tile": tile, "station": station, "is_centre": centre,
                         "year": [ds.samples[i]["year"] for i in idx],
                         "doy":  [ds.samples[i]["doy"] for i in idx]})
    return pd.concat([base.assign(depth=dep, pred=vals[d]) for d, dep in enumerate(SM_DEPTHS)],
                     ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--blocks", default="0,2,4,8,14,28,56,112",
                   help="block sizes in 20 m map pixels; 0 = plain pass, must come first")
    p.add_argument("--tiles", nargs="*", default=None, help="subset of tile keys (smoke)")
    p.add_argument("--out-dir", default=str(REPO / "eval_output" / "effres_txson"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    blocks = [int(b) for b in args.blocks.split(",")]
    if blocks[0] != 0:
        raise SystemExit("--blocks must start with 0: every coarsened pass is compared to it")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model, cfg, epoch = load_checkpoint(Path(args.ckpt), device)
    model.eval()

    # Station pixels. station_split in the readout CSV predates §47 -> re-map from the splits.
    ro = pd.read_csv(READOUTS)
    if args.tiles:
        ro = ro[ro["tile"].isin(args.tiles)]
    splits = pd.read_csv(SPLITS)
    splits["key"] = splits.apply(_make_key, axis=1)
    cur = dict(zip(splits["key"], splits["split"]))
    ro["station_split"] = ro["station"].map(cur)
    ro["tile_split"]    = ro["tile"].map(cur)
    ro["mrow"], ro["mcol"] = ro["row"] // STRIDE, ro["col"] // STRIDE
    pix = {t: g.reset_index(drop=True) for t, g in ro.groupby("tile")}
    print(f"{len(pix)} tiles, {len(ro)} station pixels")

    sub = splits[splits["key"].isin(pix)]
    tile_csv = out / "_tiles.csv"
    sub.drop(columns="key").to_csv(tile_csv, index=False)
    ds = SoilMoistureDataset(
        splits_csv      = str(tile_csv),
        era5_stats_path = str(ERA5_STATS),
        years           = list(TRAIN_YEARS),
        category_filter = cfg.get("category_filter", list(SM_CATEGORIES)),
        split_filter    = ["train", "val", "oos"],
        training        = False,
    )
    N = len(ds)
    print(f"{N:,} samples over {len({s['station_key'] for s in ds.samples})} tiles")

    map0 = np.lib.format.open_memmap(out / "maps_L0.npy", mode="w+",
                                     dtype=np.float16, shape=(N, 3, MAP_PX, MAP_PX))
    manifest = dict(run=RUN, ckpt=args.ckpt, epoch=int(epoch), git_sha=_git_sha(),
                    blocks=blocks, n_samples=N, tiles=sorted(pix),
                    generated=datetime.now().isoformat(timespec="seconds"), timing={})

    for b in blocks:
        dsb = ds if b == 0 else BlockAverageDataset(ds, b)
        loader = DataLoader(dsb, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            worker_init_fn=worker_init_fn,
                            prefetch_factor=2 if args.num_workers > 0 else None)
        rd, summ, index = [], [], []
        i0, t0 = 0, time.time()
        with torch.no_grad():
            for batch in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sm = model(batch)["sm"].float()                    # (B, 3, 112, 112)
                B = sm.shape[0]
                keys = list(batch["station_key"])
                yrs  = np.asarray(batch["year"]); dys = np.asarray(batch["doy"])
                if b == 0:
                    map0[i0:i0 + B] = sm.cpu().numpy().astype(np.float16)
                    index.append(pd.DataFrame({"idx": np.arange(i0, i0 + B), "tile": keys,
                                               "year": yrs, "doy": dys}))
                else:
                    m0 = torch.from_numpy(np.asarray(map0[i0:i0 + B], np.float32)).to(sm.device)
                    rms = (sm - m0).pow(2).mean(dim=(-2, -1)).sqrt()     # (B, 3)
                    s0  = m0.std(dim=(-2, -1)); sb = sm.std(dim=(-2, -1))
                    for d, dep in enumerate(SM_DEPTHS):
                        summ.append(pd.DataFrame({
                            "idx": np.arange(i0, i0 + B), "depth": dep,
                            "rms_diff": rms[:, d].cpu().numpy(),
                            "sd_L0": s0[:, d].cpu().numpy(), "sd_Lb": sb[:, d].cpu().numpy()}))
                smc = sm.cpu().numpy()
                for k in range(B):
                    g = pix[keys[k]]
                    v = smc[k][:, g["mrow"].to_numpy(), g["mcol"].to_numpy()]   # (3, n_pix)
                    rd.append((i0 + k, g.index.to_numpy(), v))
                i0 += B
        secs = time.time() - t0
        assert i0 == N, f"loader yielded {i0} of {N} samples"
        readouts_frame(rd, ds, pix).to_parquet(out / f"readouts_L{b}.parquet", index=False)
        if b == 0:
            map0.flush()
            pd.concat(index, ignore_index=True).to_parquet(out / "index_L0.parquet", index=False)
        else:
            pd.concat(summ, ignore_index=True).to_parquet(out / f"summary_L{b}.parquet", index=False)
        manifest["timing"][str(b)] = round(secs, 1)
        print(f"  block {b:>3} ({b * 20} m): {N:,} samples in {secs/60:.1f} min")

    ro.to_csv(out / "readouts_pixels.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
