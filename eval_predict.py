"""
Dump per-sample predictions for the held-out evaluation splits (§22).

One GPU pass over OOS / OOT / OOST writes a long-format parquet per split:
one row per (station, date, depth).  Every metric and figure downstream is a
CPU-only pass over these tables -- no GPU, no dataset rebuild.

Outputs to eval_output/:
    predictions_{oos,oot,oost}.parquet
    manifest.json                        -- run, checkpoint, epoch, split defs, counts

Usage:
    python eval_predict.py --run-name cls_depth_star_reg --ckpt best.pt
    python eval_predict.py --run-name cls_depth_star_reg --splits val   # metric gate
    python eval_predict.py --run-name cls_depth_star_reg --max-stations 5 --splits oos
"""
import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel
from train import CudaPrefetcher
from ckpt_utils import load_checkpoint
from shm_preload import preload_l12_to_shm         # §35.33 parallel L12 staging
from ablation import AblationDataset, MODALITIES     # §24 modality shuffling

CKPT_ROOT  = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
ERA5_STATS = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json")
OUT_DIR    = Path("/gpfs/work3/0/prjs1968/soilMoisture/eval_output")

# §22.2.  "val" is not a held-out split -- it exists only to reproduce the
# training-time numbers (§22.6 hard gate) and is never run by default.
EVAL_SPLITS = {
    "oos":  dict(split_filter=["oos"],          years=list(range(2016, 2023))),
    "oot":  dict(split_filter=["train", "val"], years=[2023]),
    "oost": dict(split_filter=["oos"],          years=[2023]),
    "val":  dict(split_filter=["val"],          years=list(range(2016, 2023))),
    # §26.  A dense network spans all three splits, so the station set comes from
    # --pixel-csv rather than from the `split` column.  The per-station split is
    # still carried into the output as `tile_split` / `station_split`.
    "network": dict(split_filter=["train", "val", "oos"],
                    years=list(range(2016, 2023))),
}

# Station counts measured by the §22.2 zarr probe.  This is a DATED REFERENCE, not an
# invariant: station_splits.csv has been rewritten since (the §35.27 driver-stats fix and
# the §35.29 tile-pair holdout both moved stations), so a mismatch here is expected drift
# as often as it is a fault.  Treat it as "compare against the probe", not "something
# broke" -- and re-measure it rather than editing the numbers to match a run.
EXPECTED_STATIONS = {"oos": 180, "oot": 399, "oost": 98, "val": 74}


def worker_init_fn(worker_id):
    np.random.seed(os.getpid() + worker_id)


def _make_key(r) -> str:
    """station_key as used by dataset.py (the zarr directory name)."""
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


@torch.no_grad()
def run_split(model, loader, device) -> dict:
    """Collect station-pixel predictions, targets and sample identity."""
    # STATION_ROW/COL are the U-Net-era 224x224 map centre (112, 112). The patchwise model
    # emits (B, K, n_depths) and never uses them, and it dropped the attributes — so resolve
    # them lazily instead of at function entry, where a missing attribute killed every
    # patchwise eval before the first batch.
    srow = getattr(SoilMoistureModel, "STATION_ROW", None)
    scol = getattr(SoilMoistureModel, "STATION_COL", None)
    preds, targets, keys, years, doys = [], [], [], [], []

    t0, n_batches = time.time(), len(loader)
    for i, batch in enumerate(CudaPrefetcher(loader, device)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(batch)
        # --arch patchwise emits (B, K, n_depths): the value IS the prediction and the dataset
        # already selected the station patch. No map, nothing to index. §35.20.
        if mu.ndim == 3:
            _p = mu[:, 0, :]
        elif srow is None:
            raise RuntimeError(
                f"model emitted {tuple(mu.shape)} (a pixel map) but "
                f"{type(model).__name__} has no STATION_ROW/STATION_COL to index it with")
        else:
            _p = mu[:, :, srow, scol]
        preds.append(_p.float().cpu().numpy())
        targets.append(batch["label"].float().cpu().numpy())
        keys.extend(batch["station_key"])
        years.append(np.asarray(batch["year"].cpu() if torch.is_tensor(batch["year"])
                                else batch["year"], dtype=np.int32))
        doys.append(np.asarray(batch["doy"].cpu() if torch.is_tensor(batch["doy"])
                               else batch["doy"], dtype=np.int32))

        if i % 200 == 0 and i:
            rate = (i + 1) / (time.time() - t0)
            eta  = (n_batches - i - 1) / max(rate, 1e-9) / 60
            print(f"    batch {i+1}/{n_batches}  {rate:.1f} b/s  ETA {eta:.1f} min",
                  flush=True)

    return dict(
        preds   = np.concatenate(preds,   axis=0),   # (N, n_depths)
        targets = np.concatenate(targets, axis=0),   # (N, n_depths)
        keys    = np.asarray(keys),                  # (N,)
        years   = np.concatenate(years,   axis=0),   # (N,)
        doys    = np.concatenate(doys,    axis=0),   # (N,)
    )


# ---------------------------------------------------------------------------
# §26 multi-pixel readout
#
# The model emits a full (B, n_depths, 224, 224) map but only pixel (112, 112)
# is ever supervised or read.  In a dense network the 2.24 km tiles overlap, so
# one tile's map also covers *other* stations at pixels that received no
# supervision.  This reads all of them out of the same forward pass.
# ---------------------------------------------------------------------------
class PixelMap:
    """Per-tile table of (row, col) readouts, padded to a fixed width K."""

    def __init__(self, df: pd.DataFrame):
        need = {"tile", "station", "row", "col"}
        missing = need - set(df.columns)
        if missing:
            raise SystemExit(f"--pixel-csv is missing column(s): {sorted(missing)}")

        self.tiles = sorted(df["tile"].unique())
        self.index = {t: i for i, t in enumerate(self.tiles)}
        n = len(self.tiles)
        self.K = int(df.groupby("tile").size().max())

        # Pad with the centre pixel; the mask decides what is emitted, so the
        # padded lanes are computed and discarded (cheap, keeps the gather dense).
        c = SoilMoistureModel.STATION_ROW
        self.rows  = np.full((n, self.K), c, np.int64)
        self.cols  = np.full((n, self.K), c, np.int64)
        self.valid = np.zeros((n, self.K), bool)
        self.meta: list[list[dict | None]] = [[None] * self.K for _ in range(n)]

        for tile, g in df.groupby("tile"):
            i = self.index[tile]
            for k, r in enumerate(g.itertuples()):
                self.rows[i, k]  = int(r.row)
                self.cols[i, k]  = int(r.col)
                self.valid[i, k] = True
                self.meta[i][k]  = {
                    "station":    str(r.station),
                    "row":        int(r.row),
                    "col":        int(r.col),
                    "offset_px":  int(getattr(r, "offset_px", 0)),
                    "is_centre":  bool(getattr(r, "is_centre", r.station == tile)),
                }

    def keys(self) -> list[str]:
        return list(self.tiles)


@torch.no_grad()
def run_split_pixels(model, loader, device, pmap: PixelMap) -> dict:
    """Like run_split, but reads K pixels per sample out of the same map."""
    rows_t = torch.from_numpy(pmap.rows).to(device)      # (n_tiles, K)
    cols_t = torch.from_numpy(pmap.cols).to(device)
    preds, tiles, years, doys = [], [], [], []

    t0, n_batches = time.time(), len(loader)
    for i, batch in enumerate(CudaPrefetcher(loader, device)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(batch)                            # (B, D, 224, 224)
        B, D, H, W = mu.shape

        ti  = torch.as_tensor([pmap.index[k] for k in batch["station_key"]],
                              device=device)             # (B,)
        idx = rows_t[ti] * W + cols_t[ti]                 # (B, K) flat pixel index
        val = mu.reshape(B, D, H * W).gather(
            2, idx[:, None, :].expand(-1, D, -1))         # (B, D, K)

        preds.append(val.float().cpu().numpy())
        tiles.append(ti.cpu().numpy().astype(np.int32))
        years.append(np.asarray(batch["year"].cpu() if torch.is_tensor(batch["year"])
                                else batch["year"], dtype=np.int32))
        doys.append(np.asarray(batch["doy"].cpu() if torch.is_tensor(batch["doy"])
                               else batch["doy"], dtype=np.int32))

        if i % 200 == 0 and i:
            rate = (i + 1) / (time.time() - t0)
            eta  = (n_batches - i - 1) / max(rate, 1e-9) / 60
            print(f"    batch {i+1}/{n_batches}  {rate:.1f} b/s  ETA {eta:.1f} min",
                  flush=True)

    return dict(
        preds = np.concatenate(preds, axis=0),   # (N, D, K)
        tiles = np.concatenate(tiles, axis=0),   # (N,)
        years = np.concatenate(years, axis=0),
        doys  = np.concatenate(doys,  axis=0),
    )


def to_long_frame_pixels(res: dict, pmap: PixelMap) -> pd.DataFrame:
    """(N, D, K) -> one row per (tile, station, date, depth).

    Observations are NOT joined here: an off-centre readout is scored against a
    *different* station's record, which lives in that station's own zarr.  That
    join happens in combine_network.py.
    """
    n_depths = res["preds"].shape[1]
    frames = []
    for i_tile, tile in enumerate(pmap.tiles):
        sel = res["tiles"] == i_tile
        if not sel.any():
            continue
        yrs, dys = res["years"][sel], res["doys"][sel]
        for k in range(pmap.K):
            m = pmap.meta[i_tile][k]
            if m is None:
                continue
            for d, depth in enumerate(SM_DEPTHS[:n_depths]):
                frames.append(pd.DataFrame({
                    "tile":      tile,
                    "station":   m["station"],
                    "row":       np.int16(m["row"]),
                    "col":       np.int16(m["col"]),
                    "offset_px": np.int16(m["offset_px"]),
                    "is_centre": m["is_centre"],
                    "year":      yrs,
                    "doy":       dys,
                    "depth":     depth,
                    "pred":      res["preds"][sel, d, k].astype(np.float32),
                }))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["date"] = (pd.to_datetime(df["year"].astype(str) + "-01-01")
                  + pd.to_timedelta(df["doy"] - 1, unit="D"))
    df["depth"]   = df["depth"].astype("category")
    df["tile"]    = df["tile"].astype("string")
    df["station"] = df["station"].astype("string")
    return df[["tile", "station", "row", "col", "offset_px", "is_centre",
               "year", "doy", "date", "depth", "pred"]]


def to_long_frame(res: dict, split_name: str, train_split_map: dict) -> pd.DataFrame:
    """Wide (N, n_depths) arrays -> long format, one row per (sample, depth).

    Rows where the observation is NaN (depth absent at that station) are dropped.
    """
    n_samples, n_depths = res["preds"].shape
    frames = []
    for d, depth in enumerate(SM_DEPTHS[:n_depths]):
        obs = res["targets"][:, d]
        keep = ~np.isnan(obs)
        if not keep.any():
            continue
        frames.append(pd.DataFrame({
            "station_key": res["keys"][keep],
            "year":        res["years"][keep],
            "doy":         res["doys"][keep],
            "depth":       depth,
            "pred":        res["preds"][keep, d].astype(np.float32),
            "obs":         obs[keep].astype(np.float32),
        }))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # (year, doy) -> calendar date.  doy is 1-indexed, so subtract one day.
    df["date"] = (pd.to_datetime(df["year"].astype(str) + "-01-01")
                  + pd.to_timedelta(df["doy"] - 1, unit="D"))
    df["eval_split"]  = split_name
    df["train_split"] = df["station_key"].map(train_split_map).astype("string")
    df["depth"]       = df["depth"].astype("category")
    df["station_key"] = df["station_key"].astype("string")

    return df[["station_key", "year", "doy", "date", "depth",
               "pred", "obs", "eval_split", "train_split"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name",     default="cls_depth_star_reg")
    p.add_argument("--ckpt",         default="best.pt")
    p.add_argument("--batch-size",   type=int, default=128)
    p.add_argument("--num-workers",  type=int, default=8)
    p.add_argument("--splits",       nargs="+", default=["oos", "oot", "oost"],
                   choices=list(EVAL_SPLITS))
    p.add_argument("--max-stations", type=int, default=None,
                   help="Cap stations per split (smoke-test mode)")
    p.add_argument("--out-dir",      default=str(OUT_DIR))
    p.add_argument("--no-shm",       action="store_true",
                   help="skip the parallel /dev/shm L12 preload and let the dataset read "
                        "zarr per station on one core (the pre-§35.33 behaviour; ~8 min "
                        "per 74 stations). Use only to isolate a preload bug.")
    # Defaults to the job's own core allocation rather than a hardcoded 64: over-forking
    # past --cpus-per-task just makes the workers contend for the same cores.
    p.add_argument("--shm-workers",  type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)),
                   help="processes for the preload (default: $SLURM_CPUS_PER_TASK)")
    # Station chunking -- same pattern as precompute_terramind.py.  The dataset
    # preloads L12 into RAM, so OOT (774 stations, ~156 GB) spends ~65 min in
    # init.  Splitting it across parallel jobs cuts both peak RAM and wall time
    # at the same total GPU cost.  eval_metrics.py globs the chunks back together.
    p.add_argument("--csv-start-idx", type=int, default=None,
                   help="First row of the split-filtered station CSV (inclusive)")
    p.add_argument("--csv-end-idx",   type=int, default=None,
                   help="Last row of the split-filtered station CSV (exclusive)")
    p.add_argument("--tag",           default=None,
                   help="Suffix for output files, e.g. --tag c0 -> "
                        "predictions_oot_c0.parquet")
    # §24 modality shuffling.  Run --ablate era5 FIRST as the positive control: if
    # shuffling ERA5 does not move the metrics, the permutation never reached the
    # model and every satellite condition would be a false negative.
    p.add_argument("--ablate",      default="none",
                   choices=["none"] + MODALITIES,
                   help="replace this modality with another sample's (§24)")
    p.add_argument("--ablate-mode", default="cross_station",
                   choices=["cross_station", "within_station"],
                   help="cross_station: different site, same season (kills site "
                        "identity).  within_station: same site, different season "
                        "(kills temporal state)")
    p.add_argument("--seed",        type=int, default=0,
                   help="permutation seed -- run several, one shuffle can be lucky")
    # §26 multi-pixel readout.  The CSV comes from build_network_readouts.py and
    # lists, per tile, every station that falls inside that tile's 224x224 map.
    # The station set for the run is the CSV's `tile` column, so --pixel-csv
    # implies --splits network.
    p.add_argument("--pixel-csv", default=None,
                   help="read out every (row, col) in this table from each tile's "
                        "map, not just the centre pixel (§26)")
    p.add_argument("--pixel-tiles", nargs="*", default=None,
                   help="restrict --pixel-csv to these tiles (smoke test)")
    p.add_argument("--station-flag", default=None,
                   help="restrict to stations flagged true in this station_splits.csv "
                        "column, e.g. ablation_oos (50 stratified OOS stations). The "
                        "ablation is a PAIRED comparison against the same stations in "
                        "the baseline parquet, so a subset costs power, not validity.")
    args = p.parse_args()
    if args.station_flag and (args.csv_start_idx is not None
                              or args.csv_end_idx is not None):
        raise SystemExit("--station-flag and --csv-start/end-idx are mutually exclusive")

    pmap = None
    if args.pixel_csv:
        if args.ablate != "none":
            raise SystemExit("--pixel-csv and --ablate are not supported together")
        pix = pd.read_csv(args.pixel_csv)
        if args.pixel_tiles:
            pix = pix[pix["tile"].isin(args.pixel_tiles)]
            if pix.empty:
                raise SystemExit(f"no rows left after --pixel-tiles {args.pixel_tiles}")
        pmap = PixelMap(pix)
        args.splits = ["network"]
        print(f"Pixel readout: {len(pix)} (tile, station) pairs over "
              f"{len(pmap.tiles)} tiles, K={pmap.K}, "
              f"{int(pix['is_centre'].sum()) if 'is_centre' in pix else 0} centre / "
              f"{len(pix) - (int(pix['is_centre'].sum()) if 'is_centre' in pix else 0)}"
              f" off-centre")

    # auto-tag so an ablation can never overwrite the baseline artefacts
    if args.ablate != "none":
        abl_tag = f"{args.ablate}_{args.ablate_mode}_s{args.seed}"
        args.tag = f"{args.tag}_{abl_tag}" if args.tag else abl_tag

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The checkpoint is loaded onto the CPU FIRST, deliberately. The shm preload below
    # forks a 64-process Pool, and forking a process that already holds a CUDA context is
    # unsafe; `torch.cuda.is_available()` is enough to initialise one. Loading on CPU here
    # and moving to the GPU afterwards keeps the fork clean without reading the 527 MB
    # checkpoint twice just to recover token_sel.
    ckpt_path = CKPT_ROOT / args.run_name / args.ckpt
    model, cfg, epoch = load_checkpoint(ckpt_path, torch.device("cpu"))

    # --arch patchwise predicts on the 14x14 token grid, not a 224x224 pixel map, so the
    # PixelMap gather in run_split_pixels indexes an axis that does not exist. Reject rather
    # than let it produce a wrongly-shaped answer.
    if cfg.get("arch") == "patchwise" and getattr(args, "pixel_csv", None):
        raise SystemExit(
            "--pixel-csv is a 224x224-map feature and is meaningless for --arch patchwise: "
            "the model emits one value per 160 m token. Use token indices instead (§28.9)."
        )

    # ── L12 → /dev/shm, in parallel (§35.33) ──────────────────────────────────
    # Without this the dataset falls back to `zg["s2/l12"][:, tsl, :]` per station on ONE
    # core inside __init__, once per split. Measured on job 26091958: the VAL split spent
    # ~8 min building the dataset and ~46 s running the model, and OOT is 5x larger.
    # train.py has had the parallel path since §35.31 (2733 s -> 47.4 s); eval never did.
    #
    # Staged ONCE for every split up front, not per split: OOS and OOST are the same
    # stations, and OOT is train+val, so per-split staging would re-read most of them.
    shm_dir = None
    if not args.no_shm and args.splits and args.splits != ["network"]:
        shm_dir = Path(f"/dev/shm/sm_l12_eval_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
        shm_dir.mkdir(parents=True, exist_ok=True)
        import atexit, shutil
        atexit.register(lambda: shutil.rmtree(shm_dir, ignore_errors=True))
        t_shm = time.perf_counter()
        preload_l12_to_shm(
            splits_csv      = str(SPLITS_CSV),
            category_filter = cfg.get("category_filter", ["sm_only"]),
            shm_dir         = shm_dir,
            # Every split's station pool, deduplicated inside the preloader. Caps are None:
            # evaluation never subsets, and --max-stations is a smoke-test flag whose extra
            # staging costs seconds.
            split_caps      = [(EVAL_SPLITS[s]["split_filter"], None) for s in args.splits],
            token_sel       = cfg.get("token_sel", "station"),
            workers         = args.shm_workers,
            label           = "SHM",
        )
        print(f"[SHM] Preload done in {time.perf_counter() - t_shm:.1f}s  ({shm_dir})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        raise SystemExit("CUDA required -- CudaPrefetcher and autocast assume a GPU")
    model = model.to(device)

    # token_sel='all' restores the ~30 MB/sample IPC payload that _cpu_pyramid_pool was written
    # to eliminate (its docstring records a ~437 GB queue and epoch-boundary OOM kills). At the
    # default batch size of 128 across 8 workers that is several GB per prefetched batch.
    if cfg.get("token_sel") == "all" and args.batch_size > 8:
        print(f"[eval] token_sel='all': capping --batch-size {args.batch_size} -> 8 "
              f"(~30 MB/sample crosses the DataLoader IPC barrier)")
        args.batch_size = 8

    splits_df = pd.read_csv(SPLITS_CSV)
    train_split_map = dict(zip(splits_df.apply(_make_key, axis=1), splits_df["split"]))

    # Splits may be produced by separate jobs (OOT needs ~156 GB of L12 in RAM,
    # the others ~39 GB, so they are submitted separately).  Merge into any
    # existing manifest rather than clobbering the other job's entries.
    manifest_path = out_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            print(f"  WARNING -- unreadable {manifest_path}, starting fresh")
    prior_splits = manifest.get("splits", {})

    manifest = {
        "run_name":        args.run_name,
        "checkpoint":      str(ckpt_path),
        "epoch":           int(epoch),
        "best_val_loss":   float(cfg.get("best_val_loss", float("nan")))
                           if isinstance(cfg.get("best_val_loss"), (int, float)) else None,
        "category_filter": cfg.get("category_filter", ["sm_only"]),
        "depths":          SM_DEPTHS,
        "generated":       datetime.now().isoformat(timespec="seconds"),
        "max_stations":    args.max_stations,
        "splits":          dict(prior_splits),   # keep the other job's entries
    }

    for split_name in args.splits:
        scfg = EVAL_SPLITS[split_name]
        print(f"\n{'='*66}")
        print(f"Split: {split_name.upper()}  filter={scfg['split_filter']}  "
              f"years={scfg['years'][0]}-{scfg['years'][-1]}")

        # The dataset preloads every station's L12 tokens into RAM: ~39 GB for
        # oos, ~156 GB for oot (train+val).  Without freeing the previous split
        # first, `ds = SoilMoistureDataset(...)` would hold both at once and
        # peak near 195 GB.  Free explicitly before constructing the next.
        ds = loader = res = df = None
        gc.collect()

        # Chunking: slice the split-filtered rows and hand the dataset a
        # temporary CSV holding only this chunk's stations, so it preloads
        # only their L12 tokens.
        active_csv = str(SPLITS_CSV)
        if pmap is not None:
            # The station set IS the tile set: hand the dataset only those rows so
            # it preloads only their L12 tokens.
            keys = splits_df.apply(_make_key, axis=1)
            sub  = splits_df[keys.isin(pmap.keys())]
            miss = set(pmap.keys()) - set(keys[keys.isin(pmap.keys())])
            if miss:
                print(f"  WARNING {len(miss)} tiles not in station_splits.csv: "
                      f"{sorted(miss)[:5]}")
            tile_csv = out_dir / f"_tiles_{split_name}.csv"
            sub.to_csv(tile_csv, index=False)
            active_csv = str(tile_csv)
            print(f"  Pixel-csv tiles: {len(sub)} stations "
                  f"({dict(sub['split'].value_counts())})")
        if args.station_flag:
            sub = splits_df[splits_df["split"].isin(scfg["split_filter"])]
            keep = sub[args.station_flag].astype(str).str.lower().isin(
                ["true", "1", "yes"])
            sub = sub[keep]
            flag_csv = out_dir / f"_flag_{split_name}_{args.station_flag}.csv"
            sub.to_csv(flag_csv, index=False)
            active_csv = str(flag_csv)
            print(f"  Station flag {args.station_flag}: {len(sub)} stations")
        if args.csv_start_idx is not None or args.csv_end_idx is not None:
            sub = splits_df[splits_df["split"].isin(scfg["split_filter"])]
            lo  = args.csv_start_idx or 0
            hi  = args.csv_end_idx if args.csv_end_idx is not None else len(sub)
            sub = sub.iloc[lo:hi]
            chunk_csv = out_dir / f"_chunk_{split_name}_{lo}_{hi}.csv"
            sub.to_csv(chunk_csv, index=False)
            active_csv = str(chunk_csv)
            print(f"  Chunk rows [{lo}:{hi}] of {split_name} → {len(sub)} stations")

        ds = SoilMoistureDataset(
            splits_csv      = active_csv,
            era5_stats_path = str(ERA5_STATS),
            years           = scfg["years"],
            category_filter = cfg.get("category_filter", ["sm_only"]),
            split_filter    = scfg["split_filter"],
            training        = False,
            max_stations    = args.max_stations,
            # Staged above for every split at once. None falls back to the serial
            # per-station zarr read inside __init__ (see the preload comment).
            shm_dir         = shm_dir,
            # Recovered from the checkpoint, never re-specified on the CLI: a mismatch here
            # would feed pooled keys to a patchwise model (KeyError) or the reverse.
            token_sel       = cfg.get("token_sel"),
        )
        n_stations = len({s["station_key"] for s in ds.samples})
        expected   = EXPECTED_STATIONS.get(split_name)
        chunked    = args.csv_start_idx is not None or args.csv_end_idx is not None
        print(f"  {len(ds):,} samples | {n_stations} stations "
              f"(§22.2 probe: {expected})")
        if (args.max_stations is None and not chunked
                and expected and abs(n_stations - expected) > 5):
            print(f"  NOTE -- {n_stations - expected:+d} vs the §22.2 probe. That probe "
                  f"is a dated reference, not an invariant: station_splits.csv has been "
                  f"rewritten since. Confirm the delta is a known split change before "
                  f"reading the metrics; do not assume either way.")

        if len(ds) == 0:
            print("  No samples -- skipping")
            continue

        if args.ablate != "none":
            ds = AblationDataset(ds, args.ablate, args.ablate_mode, args.seed)
            print(ds.report())

        loader = DataLoader(
            ds,
            batch_size         = args.batch_size,
            shuffle            = False,
            num_workers        = args.num_workers,
            pin_memory         = True,
            worker_init_fn     = worker_init_fn,
            persistent_workers = False,
            prefetch_factor    = 2 if args.num_workers > 0 else None,
        )

        t0 = time.time()
        if pmap is not None:
            res = run_split_pixels(model, loader, device, pmap)
            df  = to_long_frame_pixels(res, pmap)
            key_col = "station"
        else:
            res = run_split(model, loader, device)
            df  = to_long_frame(res, split_name, train_split_map)
            key_col = "station_key"
        mins = (time.time() - t0) / 60

        if df.empty:
            print("  All observations NaN -- nothing written")
            continue

        n_nan_pred = int(df["pred"].isna().sum())
        if n_nan_pred:
            print(f"  WARNING -- {n_nan_pred} NaN predictions")

        suffix   = f"_{args.tag}" if args.tag else ""
        out_path = out_dir / f"predictions_{split_name}{suffix}.parquet"
        df.to_parquet(out_path, index=False, compression="snappy")

        print(f"  {len(df):,} rows | {df[key_col].nunique()} stations | "
              f"{mins:.1f} min")
        if pmap is not None:
            print(f"  readouts: {df.groupby(['tile','station']).ngroups} "
                  f"(tile, station) pairs | "
                  f"centre {int(df.is_centre.sum()):,} rows / "
                  f"off-centre {int((~df.is_centre).sum()):,} rows")
        print(f"  rows per depth: "
              f"{df.groupby('depth', observed=True).size().to_dict()}")
        print(f"  Saved: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

        manifest["splits"][f"{split_name}{suffix}"] = {
            "split_filter": scfg["split_filter"],
            "years":        [int(scfg["years"][0]), int(scfg["years"][-1])],
            "n_stations":   int(df[key_col].nunique()),
            "n_rows":       int(len(df)),
            "n_samples":    int(len(ds)),
            "rows_by_depth": {str(k): int(v) for k, v in
                              df.groupby("depth", observed=True).size().items()},
            "nan_pred":     n_nan_pred,
            "minutes":      round(mins, 1),
            **({"pixel_csv":   args.pixel_csv,
                "n_readouts":  int(df.groupby(["tile", "station"]).ngroups),
                "n_offcentre": int(df.groupby(["tile", "station"])
                                   .is_centre.first().eq(False).sum())}
               if pmap is not None else {}),
        }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {manifest_path}")
    print("\n=== DONE ===")
    for name, m in manifest["splits"].items():
        print(f"  {name:>5s}  {m['n_stations']:>4d} stations  "
              f"{m['n_rows']:>9,d} rows  {m['minutes']:>5.1f} min")


if __name__ == "__main__":
    main()
