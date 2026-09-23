# ============================================================
# Mead (US-Ne1/Ne2/Ne3) soil-water-content probe
# ============================================================
# csvs/ameriflux_summary.csv records has_swc=False for US-Ne1 and
# US-Ne2, and US-Ne3 is absent from the inventory entirely.  That
# summary was built from the FLUXNET product; BASE is the raw
# half-hourly record and carries SWC_* columns the FLUXNET
# processing may not expose.  This downloads BASE-BADM for all
# three Mead sites and reports, per site:
#
#   - whether SWC_* columns exist at all
#   - per column: depth from the BIF, date range, valid fraction
#   - whether the record clears MIN_VALID_DAYS = 1095
#
# Downloads land in scratch, not in the project tree.
#
# Usage:
#   python check_mead_swc.py [--sites US-Ne1 US-Ne2 US-Ne3]
#                            [--policy CCBY4.0] [--out-dir ...]
# ============================================================

import argparse
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from download_ameriflux import download_badm          # noqa: E402

MEAD      = ["US-Ne1", "US-Ne2", "US-Ne3"]
OUT_DIR   = Path("/gpfs/scratch1/shared/pkhanal/ameriflux_base")
MISSING   = -9999.0
MIN_VALID_DAYS = 1095


def base_csv_from_zip(zpath: Path, workdir: Path) -> Path | None:
    """Extract the BASE half-hourly/hourly CSV from a BASE-BADM zip."""
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if re.search(r"_BASE_H[HR]_.*\.csv$", n)]
        if not names:
            return None
        zf.extract(names[0], workdir)
        return workdir / names[0]


def bif_swc_depths(zpath: Path, workdir: Path) -> dict:
    """Pull SWC sensor depths out of the BIF workbook, if present."""
    depths = {}
    try:
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if n.endswith(".xlsx")]
            if not names:
                return depths
            zf.extract(names[0], workdir)
            bif = pd.read_excel(workdir / names[0])
    except Exception as exc:
        print(f"    (BIF unreadable: {exc})")
        return depths

    cols = {c.upper(): c for c in bif.columns}
    grp = cols.get("VARIABLE_GROUP")
    var = cols.get("VARIABLE")
    val = cols.get("DATAVALUE")
    if not (grp and var and val):
        return depths

    sub = bif[bif[grp].astype(str).str.contains("VAR_INFO", na=False)]
    for _, r in sub.iterrows():
        name = str(r[var])
        if name.endswith("VAR_INFO_VARIABLE") or "HEIGHT" in name:
            pass
    # VAR_INFO rows are keyed by group index; pair VARNAME with HEIGHT
    pivot = {}
    for _, r in sub.iterrows():
        pivot.setdefault(r.get(cols.get("GROUP_ID", grp)), {})[str(r[var])] = r[val]
    for _, fields in pivot.items():
        nm = fields.get("VAR_INFO_VARIABLE", "")
        ht = fields.get("VAR_INFO_HEIGHT", None)
        if isinstance(nm, str) and nm.startswith("SWC") and ht is not None:
            depths[nm] = ht
    return depths


def probe(site: str, zpath: Path, workdir: Path):
    print(f"\n=== {site} ===")
    csv = base_csv_from_zip(zpath, workdir)
    if csv is None:
        print("  no BASE CSV inside the zip")
        return
    print(f"  BASE file: {csv.name}")

    df = pd.read_csv(csv, skiprows=2, na_values=[MISSING, str(int(MISSING))])
    swc_cols = [c for c in df.columns if c.upper().startswith("SWC")]
    if not swc_cols:
        print(f"  NO SWC columns.  ({len(df.columns)} columns, "
              f"{[c for c in df.columns[:12]]} ...)")
        return

    ts = pd.to_datetime(df["TIMESTAMP_START"].astype("Int64").astype(str),
                        format="%Y%m%d%H%M", errors="coerce")
    df = df.assign(_date=ts.dt.date)

    depths = bif_swc_depths(zpath, workdir)
    print(f"  SWC columns: {len(swc_cols)}")
    print(f"  {'column':<18}{'depth(m)':>10}{'first':>12}{'last':>12}"
          f"{'valid%':>9}{'days>=1obs':>12}")
    for c in sorted(swc_cols):
        s = df[c]
        ok = s.notna()
        if ok.sum() == 0:
            print(f"  {c:<18}{'-':>10}{'-':>12}{'-':>12}{0.0:>8.1f}%{0:>12}")
            continue
        d_ok = df.loc[ok, "_date"]
        ndays = d_ok.nunique()
        dep = depths.get(c, "")
        dep_s = f"{dep}" if dep != "" else "-"
        print(f"  {c:<18}{dep_s:>10}{str(d_ok.min()):>12}{str(d_ok.max()):>12}"
              f"{100*ok.mean():>8.1f}%{ndays:>12}")

    best = max(
        (df.loc[df[c].notna(), "_date"].nunique() for c in swc_cols), default=0
    )
    verdict = "PASSES" if best >= MIN_VALID_DAYS else "FAILS"
    print(f"  -> best column covers {best} days; {verdict} the {MIN_VALID_DAYS}-day rule")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=MEAD)
    ap.add_argument("--policy", default="CCBY4.0")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    workdir = args.out_dir / "_unzipped"
    workdir.mkdir(exist_ok=True)

    print(f"Requesting BASE-BADM for {args.sites} under policy {args.policy}")
    paths = download_badm(args.sites, out_dir=args.out_dir, data_policy=args.policy)
    got = {p.name.split("_")[0]: p for p in paths}
    print(f"\nDownloaded: {sorted(got)}")
    missing = [s for s in args.sites if s not in got]
    if missing:
        print(f"NOT RETURNED by the API under {args.policy}: {missing}")
        print("  (retry with --policy LEGACY if a site is under the legacy policy)")

    for site in args.sites:
        z = got.get(site) or next(iter(args.out_dir.glob(f"{site}_BASE-BADM.zip")), None)
        if z is None:
            print(f"\n=== {site} ===\n  no zip on disk")
            continue
        probe(site, z, workdir)


if __name__ == "__main__":
    main()
