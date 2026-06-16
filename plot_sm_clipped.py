"""
Plot SM time series after clipping to [0, 1] for all flagged stations.
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

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr/sm_only")
AUDIT_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/audit_sm_range.csv")
OUT_DIR   = Path("/gpfs/work3/0/prjs1968/soilMoisture/plots/sm_clipped")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df       = pd.read_csv(AUDIT_CSV)
stations = df["station"].unique()
depth_colors = {"0-10": "#2196F3", "10-30": "#4CAF50", "30-100": "#FF9800"}

for station in stations:
    rows   = df[df["station"] == station]
    zg     = zarr.open_group(str(ZARR_ROOT / station), mode="r")
    sm     = zg["labels/sm"][:]
    qc     = zg["labels/qc"][:] if "labels/qc" in zg else None
    depths = list(zg["labels/depths"][:])
    dates  = zg["labels/dates"][:]
    if qc is not None and qc.shape[1] != sm.shape[1]:
        qc = qc[:, -sm.shape[1]:]
    dts = [datetime.strptime(str(d), "%Y%m%d") for d in dates]

    bad_depths = list(rows["depth"])
    fig, axes  = plt.subplots(len(bad_depths), 1,
                              figsize=(14, 3.5 * len(bad_depths)), squeeze=False)
    fig.suptitle(f"SNOTEL: {station.replace('ISMN_SNOTEL_', '')}  —  SM clipped to [0, 1]",
                 fontsize=13, fontweight="bold")

    for ax_row, depth in zip(axes, bad_depths):
        ax  = ax_row[0]
        i   = depths.index(depth)
        sm_i = sm[i].astype(float)
        if qc is not None:
            sm_i[qc[i] != 0] = np.nan

        # clip out-of-range to NaN
        sm_clipped = np.where((sm_i < 0) | (sm_i > 1), np.nan, sm_i)
        n_removed  = int(rows[rows["depth"] == depth]["n_out_of_range"].values[0])

        col = depth_colors.get(depth, "steelblue")
        ax.plot(dts, sm_clipped, color=col, lw=0.8, alpha=0.9)
        ax.fill_between(dts, sm_clipped, alpha=0.15, color=col)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0.0, color="black", lw=0.5, ls="--", alpha=0.3)
        ax.axhline(1.0, color="red",   lw=0.8, ls="--", alpha=0.4, label="upper bound (1.0)")
        ax.set_ylabel("SM (m³/m³)")
        ax.set_title(f"Depth: {depth}  ({n_removed} spike days removed)", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / f"{station}_clipped.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.name}")

print(f"\nAll plots saved to {OUT_DIR}")
