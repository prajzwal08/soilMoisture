"""
analyze_combined_coverage.py
============================
Compute the longest consecutive gap of uncovered months for each station
using the union of S1RTC and S2L2A monthly coverage. Apply three thresholds
and report how many stations fail each.

A month is "covered" if at least one S2L2A OR S1RTC tile exists that month.
Coverage period per station = min_year..max_year found in monthly_coverage.csv.

Outputs:
  csvs/combined_coverage.csv        — per-station gap and threshold flags
  fig/combined_coverage_gap_hist.png — histogram of max_gap_months

Usage:
    python analyze_combined_coverage.py
    python analyze_combined_coverage.py --coverage csvs/monthly_coverage.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
COVERAGE_CSV = Path("csvs/monthly_coverage.csv")
SPLITS_CSV   = Path("csvs/station_splits.csv")
OUT_CSV      = Path("csvs/combined_coverage.csv")
FIG_OUT      = Path("fig/combined_coverage_gap_hist.png")

# (label, max_gap_months_allowed, human description)
THRESHOLDS = [
    ("T1", 0, "gap > 0  (≥1 image every month)"),
    ("T2", 1, "gap > 1  (≥1 image every 2 months)"),
    ("T3", 2, "gap > 2  (≥1 image every 3 months)"),
]


# ── core gap function ─────────────────────────────────────────────────────────

def max_coverage_gap(station_rows: pd.DataFrame) -> int:
    """Return the longest consecutive run of uncovered months for one station.

    Builds union of S1 OR S2 covered (year, month) pairs, then walks every
    calendar month from the FIRST covered month to the LAST covered month.
    Leading/trailing empty months outside that window are not counted — they
    are artifacts of partial-year downloads at period boundaries, not real gaps.
    """
    covered: set[tuple[int, int]] = set()
    for _, row in station_rows.iterrows():
        year = int(row["year"])
        mp = str(row["months_present"]).strip()
        if mp:
            for m_str in mp.split(","):
                m_str = m_str.strip()
                if m_str:
                    covered.add((year, int(m_str)))

    if not covered:
        return 0

    sorted_months = sorted(covered)
    first_yr, first_mo = sorted_months[0]
    last_yr,  last_mo  = sorted_months[-1]

    max_gap = current_run = 0
    year, month = first_yr, first_mo
    while (year, month) <= (last_yr, last_mo):
        if (year, month) in covered:
            max_gap = max(max_gap, current_run)
            current_run = 0
        else:
            current_run += 1
        month += 1
        if month > 12:
            month = 1
            year += 1
    return max_gap


# ── split map ─────────────────────────────────────────────────────────────────

def build_split_map(splits_df: pd.DataFrame) -> dict[str, str]:
    """station folder name → split label, matching plot_image_stats.py convention."""
    def folder(row: pd.Series) -> str:
        sn, nw = row["source_network"], row["network"]
        return f"{sn}_{row['station_id']}" if sn == nw else f"{sn}_{nw}_{row['station_id']}"

    df = splits_df.copy()
    df["station"] = df.apply(folder, axis=1)
    return dict(zip(df["station"], df["split"]))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, default=COVERAGE_CSV)
    parser.add_argument("--splits",   type=Path, default=SPLITS_CSV)
    parser.add_argument("--out-csv",  type=Path, default=OUT_CSV)
    parser.add_argument("--out-fig",  type=Path, default=FIG_OUT)
    args = parser.parse_args()

    cov = pd.read_csv(args.coverage, dtype={"months_present": str, "year": int})
    spl = pd.read_csv(args.splits)
    split_map = build_split_map(spl)

    records = []
    for station, grp in cov.groupby("station", sort=True):
        gap = max_coverage_gap(grp)
        records.append({
            "station":        station,
            "split":          split_map.get(station, "unknown"),
            "max_gap_months": gap,
            "fails_T1":       gap > 0,
            "fails_T2":       gap > 1,
            "fails_T3":       gap > 2,
        })

    result = pd.DataFrame(records)
    n = len(result)

    # ── build report ─────────────────────────────────────────────────────────
    lines: list[str] = [
        "=" * 62,
        "COMBINED S1+S2 MONTHLY COVERAGE GAP ANALYSIS",
        "=" * 62,
        f"  Stations analysed : {n:,}",
        "  Coverage window   : first covered month -> last covered month",
        "  (leading/trailing partial-year boundary months excluded)",
        "",
        f"  {'Thresh':<6}  {'Description':<38}  {'Removed':>7}  {'Kept':>6}  {'%':>6}",
        f"  {'-'*6}  {'-'*38}  {'-'*7}  {'-'*6}  {'-'*6}",
    ]
    for label, _, desc in THRESHOLDS:
        col    = f"fails_{label}"
        n_fail = int(result[col].sum())
        n_keep = n - n_fail
        lines.append(f"  {label:<6}  {desc:<38}  {n_fail:>7,}  {n_keep:>6,}  {n_fail/n*100:>5.1f}%")
    lines += [
        "",
        "  Decision: train with all 1,010 stations (no exclusion on coverage gap)",
        "",
        "  Max-gap distribution:",
    ]
    for gap_val, cnt in result["max_gap_months"].value_counts().sort_index().items():
        bar = "█" * int(cnt / n * 80)
        lines.append(f"    gap={gap_val:>2}  {cnt:>4} stations  {bar}")
    lines.append("")
    lines.append("  Station list per threshold (stations that would be removed):")
    for label, _, desc in THRESHOLDS:
        col      = f"fails_{label}"
        failures = result[result[col]]["station"].sort_values().tolist()
        lines.append(f"\n  {label} ({desc}) — {len(failures)} stations:")
        for s in failures:
            lines.append(f"    {s}")
    lines.append("")

    report = "\n".join(lines)

    # ── console + text file ───────────────────────────────────────────────────
    print("\n" + report)
    txt_path = Path("text/combined_coverage_analysis.txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(report)
    print(f"  TXT  -> {txt_path}")

    # ── write CSV ─────────────────────────────────────────────────────────────
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_csv, index=False)
    print(f"  CSV  → {args.out_csv}")

    # ── plot ──────────────────────────────────────────────────────────────────
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    gaps        = result["max_gap_months"].values
    max_gap_obs = int(gaps.max())

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(-0.5, max_gap_obs + 1.5, 1.0)
    ax.hist(gaps, bins=bins, color="#2166ac", edgecolor="white", linewidth=0.6)

    colors = ["#d6604d", "#f4a582", "#92c5de"]
    for (label, max_allowed, desc), color in zip(THRESHOLDS, colors):
        n_fail = int(result[f"fails_{label}"].sum())
        ax.axvline(max_allowed + 0.5, color=color, lw=1.8, ls="--",
                   label=f"{label}: {desc}  (removes {n_fail:,})")

    ax.set_xlabel("Longest consecutive gap (months with zero S1 or S2 tiles)", fontsize=11)
    ax.set_ylabel("Number of stations", fontsize=11)
    ax.set_title(
        f"Combined S1+S2 monthly coverage gap — {n:,} stations\n"
        "(gap measured over min..max year range in data)",
        fontsize=11,
    )
    ax.set_xticks(range(0, max_gap_obs + 1))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {args.out_fig}")
    print("Done.")


if __name__ == "__main__":
    main()
