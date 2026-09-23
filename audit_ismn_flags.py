# ============================================================
# ISMN quality-flag census
# ============================================================
# Answers two questions about the raw ISMN archive that the
# preprocessing pipeline's flag whitelist depends on:
#
#   1. Which ISMN quality-flag strings actually occur, and how
#      much data does each one carry?
#   2. Do C-codes (physically implausible) ever co-occur with
#      D-codes (spurious) in the same field?  If they do, the
#      rule in preprocessing_ISMN_soilMoisture.py:100
#
#          (flag == "G") | flag.str.startswith("D")
#
#      is order-sensitive: "D01,C01" is KEPT, "C01,D01" is
#      dropped, though both carry the same C-code.
#
# Reads the raw .stm files directly (no ismn package, no index
# build).  Data lines are "YYYY/MM/DD HH:MM value ismn_flag
# [provider_flag]"; both trailing columns are tabulated so the
# flag column is identified from the data, not assumed.
#
# Usage:
#   python audit_ismn_flags.py --ismn-dir /path/to/Data_separate_files_header_...
#                              [--workers 64] [--out csvs/ismn_flag_census.csv]
# ============================================================

import argparse
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

DATE_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")

DEFAULT_ISMN_DIR = (
    "/home/khanalp/data/ISMNsoilMoisture/"
    "Data_separate_files_header_20140101_20251231_13107_18mx_20260208"
)


def scan_file(path):
    """Tabulate flag strings in one .stm file.

    Returns (col3, col4, n_rows, n_unparsed) where col3/col4 are
    Counters over the two trailing columns of the data lines.
    """
    col3, col4 = Counter(), Counter()
    n_rows = n_unparsed = 0
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                tok = line.split()
                if len(tok) < 4 or not DATE_RE.match(tok[0]):
                    continue          # header or blank
                n_rows += 1
                col3[tok[3]] += 1
                if len(tok) >= 5:
                    col4[tok[4]] += 1
                else:
                    n_unparsed += 1
    except OSError as exc:
        print(f"  ! unreadable: {path} ({exc})", file=sys.stderr)
    return col3, col4, n_rows, n_unparsed


def classify(flag):
    """Split a flag string into its C-codes and D-codes."""
    codes = [c.strip() for c in flag.replace(";", ",").split(",") if c.strip()]
    return ([c for c in codes if c.upper().startswith("C")],
            [c for c in codes if c.upper().startswith("D")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ismn-dir", default=DEFAULT_ISMN_DIR)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default="csvs/ismn_flag_census.csv")
    args = ap.parse_args()

    root = Path(args.ismn_dir)
    if not root.is_dir():
        sys.exit(
            f"ERROR: raw ISMN archive not found at {root}\n"
            "       The .stm archive is required; the level-1 NetCDFs keep only\n"
            "       the 0/1/2 observed/gap-filled flag, not the ISMN codes.\n"
            "       Re-download from ismn.earth (env: soilmoisture) first."
        )

    files = sorted(root.rglob("*.stm"))
    print(f"Archive : {root}")
    print(f"Files   : {len(files)} .stm")
    if not files:
        sys.exit("ERROR: no .stm files under the archive root.")

    with Pool(args.workers) as pool:
        results = pool.map(scan_file, files, chunksize=32)

    col3, col4 = Counter(), Counter()
    n_rows = n_unparsed = 0
    for c3, c4, n, u in results:
        col3.update(c3)
        col4.update(c4)
        n_rows += n
        n_unparsed += u

    # The ISMN flag column is the one carrying G / C** / D** codes.
    def looks_ismn(counter):
        return sum(v for k, v in counter.items()
                   if k == "G" or re.match(r"^[CD]\d", k.upper()))

    flags = col3 if looks_ismn(col3) >= looks_ismn(col4) else col4
    which = "column 4 (tok[3])" if flags is col3 else "column 5 (tok[4])"

    print(f"Rows    : {n_rows:,}   (short lines: {n_unparsed:,})")
    print(f"ISMN flag column identified as {which}\n")

    # ---- per flag-string census -------------------------------------------
    rows = []
    for flag, n in flags.most_common():
        cs, dsc = classify(flag)
        kept = (flag == "G") or flag.startswith("D")     # the current rule
        rows.append({
            "flag": flag,
            "n": n,
            "frac": n / n_rows if n_rows else 0.0,
            "has_C": bool(cs),
            "has_D": bool(dsc),
            "C_codes": "|".join(cs),
            "D_codes": "|".join(dsc),
            "kept_by_current_rule": kept,
        })
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # ---- summary -----------------------------------------------------------
    def tot(mask):
        return int(df.loc[mask, "n"].sum())

    pct = lambda n: f"{n:,} ({100 * n / n_rows:.3f}%)" if n_rows else "0"

    print(f"Distinct flag strings : {len(df)}")
    print(f"  G (good)            : {pct(tot(df.flag == 'G'))}")
    print(f"  any C-code          : {pct(tot(df.has_C))}")
    print(f"  any D-code          : {pct(tot(df.has_D))}")
    print(f"  C and D together    : {pct(tot(df.has_C & df.has_D))}")
    print(f"  kept by current rule: {pct(tot(df.kept_by_current_rule))}")

    leak = df[df.has_C & df.kept_by_current_rule]
    print("\n--- THE BUG ---")
    if leak.empty:
        print("  No C-flagged value is kept by the current rule.")
        print("  The whitelist is safe as written; C exclusion is incidental")
        print("  but complete.")
    else:
        print(f"  {pct(int(leak.n.sum()))} of all values carry a C-code but are")
        print("  KEPT, because the flag string happens to start with 'D':")
        for _, r in leak.head(20).iterrows():
            print(f"    {r.flag:<24} n={r.n:>12,}  C={r.C_codes}")
        print("\n  Fix: reject on content, not on prefix --")
        print('    has_C = flags.str.contains("C")')
        print('    keep  = (flags == "G") | (flags.str.contains("D") & ~has_C)')

    print("\n--- cost per code (rows carrying it) ---")
    per_code = Counter()
    for _, r in df.iterrows():
        for c in (r.C_codes.split("|") if r.C_codes else []):
            per_code[c] += r.n
        for d in (r.D_codes.split("|") if r.D_codes else []):
            per_code[d] += r.n
    for code, n in sorted(per_code.items()):
        print(f"  {code:<6} {pct(n)}")

    print(f"\nCensus written to {out}")


if __name__ == "__main__":
    main()
