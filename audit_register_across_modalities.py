"""§27a.6 — why do the register dimensions show up in DEM and LULC? Are S2/S1 free of them?

§27a.3-27a.5 measured the static modalities only, and found:
    dim 328  spikes WITHIN one token (the attention-sink token), 100% of affected stations
    dim 87 / 126  large MEAN ACROSS STATIONS -> near-constant offset; zeroing 2 of 768 dims
                  drops median inter-station cosine for DEM from 0.783 to 0.157

Never checked: S2 and S1. Two competing explanations, with opposite consequences:

  (A) ENCODER-WIDE. ViTs are known to develop high-norm "register" tokens and a handful of
      massive-activation coordinates regardless of input (Darcet et al., Vision Transformers
      Need Registers; Sun et al., Massive Activations in LLMs). If so, S2/S1 carry the same
      dims and any fix must be applied to every modality, including the 196 unpooled anchor
      tokens.

  (B) SPECIFIC TO REDUNDANT INPUT. Registers are recruited in low-information patches. DEM
      and LULC are single-band, static and often near-uniform (the CR200-18 LULC sink token
      was a 16x16 patch of a single class; the DEM one spanned 1.7 m). S2 is 12-band and
      textured. If so, the artefact is concentrated where the input carries least, and the
      fix can be targeted.

The two are distinguishable:
  * do S2/S1 share the SAME dim indices?  -> (A)
  * does register strength rise as the input patch gets more uniform?  -> (B)
  * they are not exclusive: an encoder-wide direction can still be excited most by flat input.

Reads ONE acquisition per station per modality/layer (a single zarr index, ~300 kB), so the
whole sweep is a few hundred MB. CPU only, Pool(64).

    python audit_register_across_modalities.py [--workers 64] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

warnings.filterwarnings("ignore")

REPO     = Path(__file__).resolve().parent
TOK_ZARR = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SAT_ZARR = Path("/projects/prjs1968/satellite_zarr")
SPLITS   = REPO / "csvs" / "station_splits.csv"
OUT_JSON = REPO / "csvs" / "register_across_modalities.json"
OUT_CSV  = REPO / "csvs" / "register_across_modalities.csv"

GRID, D = 14, 768
CATEGORIES = ("sm_only", "sm_and_flux", "flux_only")
LAYERS = ("l3", "l6", "l9", "l12")
DYNAMIC = ("s2", "s1_asc", "s1_desc")
STATIC = ("dem", "lulc")
REG_FACTOR = 10.0
KNOWN = (87, 126, 328)          # dims found on the static modalities


def station_dir(row: pd.Series) -> str:
    if str(row["source_network"]) == "ISMN":
        return f"ISMN_{row['network']}_{row['station_name']}"
    return f"{row['source_network']}_{row['station_id']}"


def disk_index() -> dict[str, str]:
    idx = {}
    for cat in CATEGORIES:
        d = TOK_ZARR / cat
        if d.is_dir():
            for n in d.iterdir():
                idx.setdefault(n.name, cat)
    return idx


def load_one(task):
    """-> {key: (768,) token-mean vector, plus per-token norm stats and input relief}."""
    name, cat = task
    try:
        zg = zarr.open_consolidated(str(TOK_ZARR / cat / name))
    except Exception:                                       # noqa: BLE001
        return None
    out = {"station": name, "vec": {}, "stat": {}}

    for mod in STATIC:
        if mod not in zg:
            continue
        t = np.asarray(zg[mod][:], np.float32)
        out["vec"][f"{mod}/l12"] = t.mean(0)
        nn = np.linalg.norm(t, axis=-1)
        out["stat"][f"{mod}/l12"] = (float(nn.max() / max(np.median(nn), 1e-8)),)

    for mod in DYNAMIC:
        if f"{mod}/dates" not in zg:
            continue
        n = zg[f"{mod}/dates"].shape[0]
        if n == 0:
            continue
        i = n // 2                                          # one mid-record acquisition
        for lay in LAYERS:
            key = f"{mod}/{lay}"
            if key not in zg:
                continue
            t = np.asarray(zg[key][i], np.float32)
            out["vec"][key] = t.mean(0)
            nn = np.linalg.norm(t, axis=-1)
            out["stat"][key] = (float(nn.max() / max(np.median(nn), 1e-8)),)

    # input redundancy: how flat is the DEM patch, how uniform is the LULC patch?
    try:
        raw = zarr.open_group(str(SAT_ZARR / f"{name}.zarr"), mode="r")
        if "dem/data" in raw:
            dem = np.asarray(raw["dem/data"][0], np.float32)
            out["dem_relief"] = float(np.nanstd(dem))
        if "lulc/data" in raw:
            lulc = np.asarray(raw["lulc/data"][-1], np.uint8)
            _, cnt = np.unique(lulc, return_counts=True)
            out["lulc_purity"] = float(cnt.max() / cnt.sum())     # 1.0 = single class
    except Exception:                                       # noqa: BLE001
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    disk = disk_index()
    df = pd.read_csv(SPLITS).dropna(subset=["source_network", "network", "station_id"])
    names = [n for n in (station_dir(r) for _, r in df.iterrows()) if n in disk]
    if args.limit:
        names = names[:args.limit]
    print(f"{len(names)} stations\n")

    with Pool(args.workers) as pool:
        recs = [r for r in pool.map(load_one, [(n, disk[n]) for n in names], chunksize=4)
                if r is not None]

    keys = sorted({k for r in recs for k in r["vec"]})
    print(f"{'modality/layer':>14} {'n':>5} {'register dims':>26} {'mag share':>10} "
          f"{'var share':>10} {'tok max/med':>12}")
    print("-" * 84)

    summary, rows = {}, []
    for key in keys:
        have = [r for r in recs if key in r["vec"]]
        X = np.stack([r["vec"][key] for r in have])          # (S, 768)
        m = X.mean(0)
        reg = np.where(np.abs(m) > REG_FACTOR * np.median(np.abs(m)))[0]
        if reg.size == 0:
            reg = np.array([int(np.argmax(np.abs(m)))])
        v = X.var(0)
        mag = float((m[reg] ** 2).sum() / max((m ** 2).sum(), 1e-12))
        var = float(v[reg].sum() / max(v.sum(), 1e-12))
        tok = float(np.median([r["stat"][key][0] for r in have]))
        shown = ", ".join(str(d) for d in reg[:5]) + (" …" if reg.size > 5 else "")
        print(f"{key:>14} {len(have):>5} {shown:>26} {mag:>10.3f} {var:>10.3f} "
              f"{tok:>12.1f}")
        summary[key] = {"n": len(have), "register_dims": reg.tolist()[:12],
                        "n_register": int(reg.size), "mean_magnitude_share": mag,
                        "across_station_var_share": var, "median_token_max_over_median": tok,
                        "shares_known_dims": sorted(set(reg.tolist()) & set(KNOWN))}
        rows.append({"key": key, "n": len(have), "n_register": int(reg.size),
                     "mag_share": mag, "var_share": var, "tok_ratio": tok,
                     "dims": " ".join(map(str, reg.tolist()[:12]))})

    print(f"\n{'='*84}\nDo the dynamic modalities share the static ones' register dims "
          f"{KNOWN}?\n{'='*84}")
    for key in keys:
        sh = summary[key]["shares_known_dims"]
        print(f"  {key:>14}: {sh if sh else 'none'}")

    # (B): does register strength track how uniform the input patch is?
    print(f"\n{'='*84}\nIs the artefact stronger where the input is more redundant?"
          f"\n{'='*84}")
    for key, field, label in (("dem/l12", "dem_relief", "DEM relief (m, sd)"),
                              ("lulc/l12", "lulc_purity", "LULC purity (1 = one class)")):
        have = [r for r in recs if key in r["vec"] and field in r]
        if len(have) < 30:
            print(f"  {key}: not enough stations with {field}")
            continue
        strength = np.array([r["stat"][key][0] for r in have])
        x = np.array([r[field] for r in have])
        ok = np.isfinite(strength) & np.isfinite(x)
        r_p = float(np.corrcoef(x[ok], strength[ok])[0, 1])
        lo, hi = np.percentile(x[ok], [25, 75])
        print(f"  {key}: r({label}, token max/median norm) = {r_p:+.3f}   n={int(ok.sum())}")
        print(f"      strength in the flattest/purest quartile: "
              f"{np.median(strength[ok][x[ok] >= hi] if field == 'lulc_purity' else strength[ok][x[ok] <= lo]):.1f}x"
              f"   vs most varied quartile: "
              f"{np.median(strength[ok][x[ok] <= lo] if field == 'lulc_purity' else strength[ok][x[ok] >= hi]):.1f}x")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_CSV.name} and {OUT_JSON.name}")


if __name__ == "__main__":
    main()
