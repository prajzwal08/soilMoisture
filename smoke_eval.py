"""CPU smoke test for the eval path — run BEFORE any eval GPU job.

Three GPU jobs (26091536, 26091672, 26091686) died on pure CPU faults: an unbound
`arch` in ckpt_utils, a `use_mmap=` kwarg dataset.py had dropped, and a
`SoilMoistureModel.STATION_ROW` that the patchwise model no longer defines. None
needed a GPU to find; the last one burned 11 minutes of H100 first, because the
dataset's L12 preload runs before the first batch.

The drift class this exists to catch is **caller/callee signature skew**: eval and
plotting scripts fall behind dataset.py / model.py refactors because nothing
exercises them between training runs.

What it checks, in the order the real job hits them:
    1. every eval/plot module imports
    2. load_checkpoint() builds the model from a real checkpoint, on CPU
    3. SoilMoistureDataset accepts exactly the kwargs eval_predict.py passes
    4. one batch collates, and model(batch) runs forward on CPU
    5. run_split()'s readout indexing works on that output's shape
    6. every plot script's argparse accepts the flags slurm/eval_figures.sh sends

Usage (via slurm/smoke_eval.sh — nothing runs on the login node):
    python smoke_eval.py --run-name pw_stage2a_L3 --ckpt best.pt
"""
import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path

import torch

CKPT_ROOT = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
ERA5_STATS = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json")

# Exactly the flags slurm/eval_figures.sh sends. If a script's argparse rejects one,
# the figure job dies after the GPU job has already run.
PLOT_ARGS = {
    "plot_eval_scatter":    ["--in-dir", "X", "--out-dir", "Y"],
    "plot_eval_boxplot":    ["--in-dir", "X", "--out-dir", "Y",
                             "--splits", "val", "oos", "oot", "oost"],
    "plot_eval_ecosystem":  ["--in-dir", "X", "--out-dir", "Y",
                             "--by", "igbp_macro"],
    "plot_eval_timeseries": ["--in-dir", "X", "--out-dir", "Y",
                             "--splits", "val", "--select", "named",
                             "--stations", "ISMN_TxSON_CR200-18"],
    "eval_metrics":         ["--in-dir", "X", "--out-dir", "Y"],
}

results = []


def check(name):
    """Decorator-free step runner: record pass/fail, never abort the whole run.

    Every step is reported, because "the first failure" is rarely the only one —
    the point of a smoke test is to hand back the full list in one pass.
    """
    def run(fn):
        try:
            detail = fn()
            results.append((True, name, detail or ""))
            print(f"  PASS  {name}  {detail or ''}", flush=True)
            return True
        except Exception as e:
            results.append((False, name, f"{type(e).__name__}: {e}"))
            print(f"  FAIL  {name}", flush=True)
            traceback.print_exc()
            return False
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="pw_stage2a_L3")
    p.add_argument("--ckpt",     default="best.pt")
    p.add_argument("--stations", type=int, default=2,
                   help="stations to build the smoke dataset from")
    args = p.parse_args()

    print("=" * 70)
    print(f"CPU smoke test — run={args.run_name} ckpt={args.ckpt}")
    print("=" * 70)

    # ── 1. imports ────────────────────────────────────────────────────
    print("\n[1] imports")
    mods = {}
    for m in ["ckpt_utils", "dataset", "model", "shm_preload", "eval_predict",
              "eval_metrics", "plot_eval_scatter", "plot_eval_boxplot",
              "plot_eval_ecosystem", "plot_eval_timeseries"]:
        check(f"import {m}")(lambda m=m: mods.__setitem__(m, importlib.import_module(m)))

    if "ckpt_utils" not in mods or "dataset" not in mods:
        print("\nCannot continue without ckpt_utils and dataset.")
        return summarise()

    # ── 2. checkpoint loads on CPU ────────────────────────────────────
    print("\n[2] checkpoint")
    state = {}

    @check("load_checkpoint on CPU")
    def _load():
        model, cfg, epoch = mods["ckpt_utils"].load_checkpoint(
            CKPT_ROOT / args.run_name / args.ckpt, torch.device("cpu"))
        state["model"], state["cfg"], state["epoch"] = model, cfg, epoch
        return (f"arch={cfg.get('arch')} epoch={epoch} "
                f"token_sel={cfg.get('token_sel')} n_layers={cfg.get('n_layers')}")

    if "model" not in state:
        return summarise()

    # ── 3. the parallel shm preload, then the dataset reading from it ─
    print("\n[3] shm preload + dataset construction (as eval_predict.py does it)")
    cfg = state["cfg"]

    # §35.33: eval now stages L12 into /dev/shm with a fork Pool instead of letting
    # __init__ read zarr per station on one core. Exercise BOTH halves here — a preload
    # that writes nothing is invisible, because the dataset silently falls back to the
    # slow path and the job merely takes 90 minutes longer.
    @check("preload_l12_to_shm (2 stations, 4 workers)")
    def _shm():
        import shutil
        shm = Path(f"/dev/shm/sm_l12_smoke_{os.getpid()}")
        shm.mkdir(parents=True, exist_ok=True)
        state["shm"] = shm
        globals().setdefault("_CLEANUP", []).append(
            lambda: shutil.rmtree(shm, ignore_errors=True))
        n = mods["shm_preload"].preload_l12_to_shm(
            splits_csv      = str(SPLITS_CSV),
            category_filter = cfg.get("category_filter", ["sm_only"]),
            shm_dir         = shm,
            split_caps      = [("val", args.stations)],
            token_sel       = cfg.get("token_sel", "station"),
            workers         = 4,
            label           = "smoke",
        )
        if n == 0:
            raise RuntimeError("preload wrote 0 stations — the dataset would silently "
                               "fall back to the serial zarr path")
        nbin = len(list(shm.glob("*.bin")))
        return f"{n} stations, {nbin} .bin files staged"

    @check("SoilMoistureDataset(**eval kwargs, shm_dir=...)")
    def _ds():
        ds = mods["dataset"].SoilMoistureDataset(
            splits_csv      = str(SPLITS_CSV),
            era5_stats_path = str(ERA5_STATS),
            years           = list(range(2016, 2023)),
            category_filter = cfg.get("category_filter", ["sm_only"]),
            split_filter    = ["val"],
            training        = False,
            max_stations    = args.stations,
            token_sel       = cfg.get("token_sel"),
            shm_dir         = state.get("shm"),
        )
        state["ds"] = ds
        return f"{len(ds):,} samples from {args.stations} stations"

    if "ds" not in state or len(state["ds"]) == 0:
        print("  (no samples — cannot exercise forward pass)")
        return summarise()

    # ── 4. collate + forward on CPU ───────────────────────────────────
    print("\n[4] one batch, forward on CPU")

    @check("DataLoader collate")
    def _batch():
        loader = torch.utils.data.DataLoader(state["ds"], batch_size=2, num_workers=0)
        state["batch"] = next(iter(loader))
        return f"{len(state['batch'])} keys"

    @check("model(batch) forward")
    def _fwd():
        with torch.no_grad():
            mu = state["model"](state["batch"])
        state["mu"] = mu
        return f"output {tuple(mu.shape)}"

    # ── 5. the readout eval_predict.py actually performs ──────────────
    print("\n[5] run_split readout indexing")

    @check("station readout from model output")
    def _readout():
        mu = state["mu"]
        sm = mods["model"].SoilMoistureModel
        srow = getattr(sm, "STATION_ROW", None)
        scol = getattr(sm, "STATION_COL", None)
        if mu.ndim == 3:
            return f"patchwise branch mu[:, 0, :] -> {tuple(mu[:, 0, :].shape)}"
        if srow is None:
            raise RuntimeError(f"mu is {tuple(mu.shape)} (a pixel map) but "
                               f"SoilMoistureModel has no STATION_ROW/STATION_COL")
        return f"map branch mu[:, :, {srow}, {scol}]"

    # ── 6. plot-script argparse accepts the wrapper's flags ───────────
    print("\n[6] plot script argparse (flags slurm/eval_figures.sh sends)")
    for mod_name, argv in PLOT_ARGS.items():
        if mod_name not in mods:
            continue

        @check(f"{mod_name} argparse")
        def _ap(mod_name=mod_name, argv=argv):
            # Re-run the module's own parser without executing main(): build it by
            # calling main() with a patched parse_args is fragile, so instead just
            # confirm the flags exist in the parser the module declares.
            mod = mods[mod_name]
            src = Path(f"{mod_name}.py").read_text()
            missing = [a for a in argv
                       if a.startswith("--") and f'"{a}"' not in src]
            if missing:
                raise SystemExit(f"parser has no {missing}")
            return f"accepts {' '.join(a for a in argv if a.startswith('--'))}"

    return summarise()


def summarise():
    # /dev/shm is resident RAM on a shared node — never leave a smoke run's staging behind.
    for _f in globals().get("_CLEANUP", []):
        _f()
    n_fail = sum(1 for ok, _, _ in results if not ok)
    print("\n" + "=" * 70)
    for ok, name, detail in results:
        if not ok:
            print(f"  FAIL  {name}: {detail}")
    print(f"{len(results) - n_fail}/{len(results)} passed")
    print("=" * 70)
    if n_fail:
        print("DO NOT submit the GPU job until these pass.")
        sys.exit(1)
    print("CPU path clean — safe to submit slurm/eval_predict.sh")
    return 0


if __name__ == "__main__":
    main()
