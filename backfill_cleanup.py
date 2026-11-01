"""
backfill_cleanup.py — §50: drop the pre-change copies AFTER a station verified (terramind)
===========================================================================================
Removes, per station, only what the backfill/repair left beside the live data:
  raw   satellite_zarr/{st}.zarr/{s2_prebackfill, s2_badrepair}
  token zarr_tokens/{cat}/{st}/{s2_prebackfill, cm_prebackfill}, *.prebackfill, *.prerepair
then re-consolidates the token store's metadata (the old groups were listed in .zmetadata).
The same data is in the 0b backup (verified by count + bytes, a-w). Token-store edits must run
under backfill_splice_guarded.sh --cleanup. Only stations passed in (the verify OK list).
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from backfill_merge import RAW_ROOT, TOK_ROOT, _cats  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stations-file", required=True)
a = ap.parse_args()
cats = _cats()
import zarr  # noqa: E402
for st in Path(a.stations_file).read_text().split():
    removed = []
    for d in ("s2_prebackfill", "s2_badrepair"):
        p = RAW_ROOT / f"{st}.zarr" / d
        if p.exists():
            shutil.rmtree(p); removed.append(f"raw/{d}")                     # noqa: E702
    sd = TOK_ROOT / cats[st] / st
    for d in ("s2_prebackfill", "cm_prebackfill"):
        if (sd / d).exists():
            shutil.rmtree(sd / d); removed.append(f"tok/{d}")                 # noqa: E702
    for f in list(sd.glob("*.prebackfill")) + list(sd.glob("*.prerepair")):
        f.unlink(); removed.append(f"tok/{f.name}")                          # noqa: E702
    if any(r.startswith("tok/") for r in removed):
        zarr.consolidate_metadata(str(sd))
    print(f"  {st}: removed {removed or 'nothing'}", flush=True)
