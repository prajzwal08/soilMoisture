"""
Plot observed SM time series before and after clipping to [0, 1]
for all flagged station-depth combinations from audit_sm_range.csv.
"""
import zarr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

ZARR_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/zarr/sm_only")
AUDIT_CSV  = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/audit_sm_range.csv")
OUT_DIR    = Path("/gpfs/work3/0/prjs1968/soilMoisture/plots/sm_range_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(AUDIT_CSV)
stations = df["station"].unique()

for station in stations:
    rows = df[df["station"] == station]
    zg     = zarr.open_group(str(ZARR_ROOT / station), mode="r")
    sm     = zg["labels/sm"][:]
    qc     = zg["labels/qc"][:] if "labels/qc" in zg else None
    depths = list(zg["labels/depths"][:])
    dates  = zg["labels/dates"][:]
    if qc is not None and qc.shape[1] != sm.shape[1]:
        qc = qc[:, -sm.shape[1]:]
    dts = [datetime.strptime(str(d), "%Y%m%d") for d in dates]

    bad_depths = list(rows["depth"])
    n_depths   = len(bad_depths)
    fig, axes  = plt.subplots(n_depths, 1, figsize=(14, 4 * n_depths), squeeze=False)
    fig.suptitle(station.replace("ISMN_SNOTEL_", "SNOTEL: "), fontsize=13, fontweight="bold")

    for ax_row, depth in zip(axes, bad_depths):
        ax = ax_row[0]
        i  = depths.index(depth)
        sm_i = sm[i].astype(float)
        if qc is not None:
            sm_i[qc[i] != 0] = np.nan   # observed only

        sm_clipped = np.where((sm_i < 0) | (sm_i > 1), np.nan, sm_i)

        n_oor = int(rows[rows["depth"] == depth]["n_out_of_range"].values[0])
        pct   = float(rows[rows["depth"] == depth]["pct_out_of_range"].values[0])
        sm_max = float(rows[rows["depth"] == depth]["sm_max"].values[0])

        ax.plot(dts, sm_i,       color="tomato",      lw=0.7, alpha=0.9, label=f"Original (max={sm_max:.2f})")
        ax.plot(dts, sm_clipped, color="steelblue",   lw=0.7, alpha=0.9, label=f"Clipped [0–1] ({n_oor} days, {pct}% removed)")
        ax.axhline(1.0, color="red",   lw=1, ls="--", alpha=0.5)
        ax.axhline(0.0, color="black", lw=0.5, ls="--", alpha=0.3)
        ax.set_ylabel("SM (m³/m³)")
        ax.set_title(f"Depth: {depth}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / f"{station}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.name}")

print(f"\nAll plots saved to {OUT_DIR}")
