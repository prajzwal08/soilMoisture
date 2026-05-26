"""
Plot quality overview for 1 in every 10 AmeriFlux stations.
One PNG per station saved to plots/ameriflux_quality/.
Each figure shows all available variables as time series subplots.
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "no-latex"])

# ---------------------------------------------------------------------------
LEVEL1_DIR = Path("/home/khanalp/data/soilmoisture/ameriflux_level1")
OUT_DIR    = Path("/home/khanalp/code/PhD/soilMoisture/plots/ameriflux_quality")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLUX_VARS  = ["LE_F_MDS"]
FLUX_UNITS = {"LE_F_MDS": "W m-2"}


QC_THRESHOLD = 0.60  # QC >= this → measured (blue), else gap-filled (red)
COL_MEASURED = "steelblue"
COL_FILLED   = "tab:red"


def _point_colors(qc_frac: np.ndarray) -> list[str]:
    """Per-timestep color: blue if QC >= threshold, red otherwise."""
    return [COL_MEASURED if q >= QC_THRESHOLD else COL_FILLED for q in qc_frac]


def plot_station(nc_path: Path) -> None:
    ds = xr.open_dataset(nc_path)
    site_id   = ds.attrs.get("site_id", nc_path.stem)
    site_name = ds.attrs.get("site_name", "")
    lat       = ds.attrs.get("latitude", float("nan"))
    lon       = ds.attrs.get("longitude", float("nan"))
    igbp      = ds.attrs.get("IGBP", "")
    has_swc   = "soil_moisture" in ds

    time = ds["date_time"].values

    panels = []
    if has_swc:
        for depth in ds["depth"].values:
            sm   = ds["soil_moisture"].sel(depth=depth).values
            flag = ds["soil_moisture_qc"].sel(depth=depth).values if "soil_moisture_qc" in ds else None
            # convert flag (0=observed,1=gap-filled) to QC fraction (1=measured,0=filled)
            qc_frac = (1.0 - flag.astype(float)) if flag is not None else np.ones(len(sm))
            panels.append(("swc", depth, sm, qc_frac))

    for var in FLUX_VARS:
        if var in ds:
            qc_col  = f"{var}_QC"
            qc_frac = ds[qc_col].values.astype(float) if qc_col in ds else np.ones(len(time))
            panels.append(("flux", var, ds[var].values, qc_frac))

    n_panels = len(panels)
    if n_panels == 0:
        ds.close()
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    title = f"{site_id} - {site_name} | {igbp} | {lat:.2f}N, {lon:.2f}E"
    fig.suptitle(title, fontsize=9, y=1.0)

    for ax, panel in zip(axes, panels):
        kind, label, values, qc_frac = panel[0], panel[1], panel[2], panel[3]

        colors = _point_colors(np.nan_to_num(qc_frac, nan=0.0))
        ax.scatter(time, values, c=colors, s=2, linewidths=0)

        n_measured = int(np.sum(np.array(colors) == COL_MEASURED))
        n_total    = int(np.sum(~np.isnan(values)))
        pct_meas   = 100 * n_measured / max(n_total, 1)
        ax.text(0.01, 0.92, f"measured={pct_meas:.0f}%",
                transform=ax.transAxes, fontsize=6, va="top")

        if kind == "swc":
            ax.set_ylabel(f"SWC\n{label}\nm3 m-3", fontsize=7)
        else:
            ax.set_ylabel(f"{label}\n{FLUX_UNITS.get(label,'')}", fontsize=7)

        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    # shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_MEASURED, markersize=5, label="measured (QC>=0.75)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_FILLED,   markersize=5, label="gap-filled (QC<0.75)"),
    ]
    axes[0].legend(handles=handles, fontsize=6, loc="upper right", ncol=2)

    axes[-1].tick_params(axis="x", labelsize=6, rotation=30)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{site_id}.svg", dpi=120, bbox_inches="tight")
    plt.close(fig)
    ds.close()


def main():
    all_nc = sorted(
        list((LEVEL1_DIR / "swc_and_flux").glob("*.nc")) +
        list((LEVEL1_DIR / "flux_only").glob("*.nc"))
    )

    # 1 in every 10
    selected = all_nc[::10]
    print(f"Total stations: {len(all_nc)}  |  Plotting: {len(selected)}")

    for i, nc_path in enumerate(selected, 1):
        site_id = nc_path.stem.split("_")[0]
        print(f"  [{i}/{len(selected)}] {site_id}", flush=True)
        try:
            plot_station(nc_path)
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nSaved {len(selected)} SVG plots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
