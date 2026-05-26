"""
Exploratory statistics and plots for preprocessed AmeriFlux data.
Saves SVGs to plots/ameriflux_stats/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "no-latex"])

LEVEL1   = Path("/home/khanalp/data/soilmoisture/ameriflux_level1")
OUT_DIR  = Path("/home/khanalp/code/PhD/soilMoisture/plots/ameriflux_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Collect metadata from all NetCDF files
# ---------------------------------------------------------------------------
files = sorted(
    list((LEVEL1 / "swc_and_flux").glob("*.nc")) +
    list((LEVEL1 / "flux_only").glob("*.nc"))
)

records = []
for f in files:
    ds = xr.open_dataset(f)
    a  = ds.attrs
    time = pd.DatetimeIndex(ds["date_time"].values)
    n_days  = len(time)
    n_years = n_days / 365.25

    # LE valid days
    le_valid = int((~ds["LE_F_MDS"].isnull()).sum()) if "LE_F_MDS" in ds else 0

    records.append({
        "site_id":   a.get("site_id", f.stem),
        "has_swc":   a.get("has_soil_moisture", "False") == "True",
        "lat":       float(a.get("latitude",  np.nan)),
        "lon":       float(a.get("longitude", np.nan)),
        "igbp":      a.get("IGBP", "Unknown"),
        "climate":   a.get("climate_koeppen", "Unknown"),
        "start":     time[0],
        "end":       time[-1],
        "n_days":    n_days,
        "n_years":   round(n_years, 2),
        "le_valid_days": le_valid,
        "le_valid_frac": round(le_valid / max(n_days, 1), 2),
    })
    ds.close()

df = pd.DataFrame(records).sort_values("n_years", ascending=False)

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
swc = df[df["has_swc"]]
le_only = df[~df["has_swc"]]

print("=" * 60)
print("AmeriFlux Preprocessed Dataset Summary")
print("=" * 60)
print(f"Total stations          : {len(df)}")
print(f"  SWC + LE              : {len(swc)}")
print(f"  LE only               : {len(le_only)}")
print(f"Total station-years     : {df['n_years'].sum():.1f}")
print(f"Mean years/station      : {df['n_years'].mean():.1f}")
print(f"Median years/station    : {df['n_years'].median():.1f}")
print(f"Min years/station       : {df['n_years'].min():.1f}")
print(f"Max years/station       : {df['n_years'].max():.1f}")
print(f"Date range              : {df['start'].min().year} – {df['end'].max().year}")
print(f"Mean LE valid fraction  : {df['le_valid_frac'].mean():.2f}")
print()
print("IGBP distribution:")
print(df["igbp"].value_counts().to_string())
print()
print("Climate (Koeppen) distribution:")
# First letter only
df["climate_group"] = df["climate"].str[:1].replace("", "Unknown").fillna("Unknown")
print(df["climate_group"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Fig 1: Histogram of station duration
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.5))
bins = np.arange(2, df["n_years"].max() + 1, 0.5)
ax.hist(df["n_years"], bins=bins, color="steelblue", edgecolor="white", lw=0.4)
ax.axvline(df["n_years"].mean(), color="tab:red", lw=1, ls="--",
           label=f"Mean = {df['n_years'].mean():.1f} yr")
ax.set_xlabel("Station duration (years)")
ax.set_ylabel("Number of stations")
ax.set_title("Distribution of valid LE record length")
ax.legend(fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "duration_histogram.svg", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2: Gantt-style timeline per station (sorted by start date)
# ---------------------------------------------------------------------------
df_gantt = df.sort_values("start")
fig, ax = plt.subplots(figsize=(10, max(4, len(df_gantt) * 0.18)))
colors = ["#2166ac" if has else "#4dac26" for has in df_gantt["has_swc"]]
for i, (_, row) in enumerate(df_gantt.iterrows()):
    ax.barh(i, (row["end"] - row["start"]).days / 365.25,
            left=row["start"].year + row["start"].dayofyear / 365.25,
            height=0.7, color=colors[i], alpha=0.85)
ax.set_yticks(range(len(df_gantt)))
ax.set_yticklabels(df_gantt["site_id"], fontsize=4)
ax.set_xlabel("Year")
ax.set_title(f"Data coverage per station (n={len(df_gantt)})")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#2166ac", label="SWC + LE"),
    Patch(color="#4dac26", label="LE only"),
], fontsize=7, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "station_timeline.svg", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3: Stations active per year
# ---------------------------------------------------------------------------
years = range(2017, 2026)
counts = [df[(df["start"].dt.year <= y) & (df["end"].dt.year >= y)].shape[0] for y in years]
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(years, counts, color="steelblue", edgecolor="white", lw=0.4)
ax.set_xlabel("Year")
ax.set_ylabel("Number of active stations")
ax.set_title("Stations with valid LE data per year")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "stations_per_year.svg", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4: IGBP distribution
# ---------------------------------------------------------------------------
igbp_counts = df["igbp"].value_counts()
fig, ax = plt.subplots(figsize=(5, 3.5))
igbp_counts.plot.bar(ax=ax, color="steelblue", edgecolor="white", lw=0.4)
ax.set_xlabel("IGBP land cover")
ax.set_ylabel("Number of stations")
ax.set_title("IGBP distribution")
ax.tick_params(axis="x", rotation=45, labelsize=7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "igbp_distribution.svg", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5: Geographic distribution
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
sc = ax.scatter(df["lon"], df["lat"], c=df["n_years"],
                cmap="YlOrRd", s=20, edgecolors="k", lw=0.3, vmin=2)
plt.colorbar(sc, ax=ax, label="Valid years")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"AmeriFlux stations (n={len(df)})")
swc_pts = df[df["has_swc"]]
ax.scatter(swc_pts["lon"], swc_pts["lat"], s=40, facecolors="none",
           edgecolors="steelblue", lw=1, label="Has SWC")
ax.legend(fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "geographic_distribution.svg", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Save summary CSV
# ---------------------------------------------------------------------------
df.drop(columns=["climate_group"]).to_csv(
    "/home/khanalp/code/PhD/soilMoisture/csvs/ameriflux_summary.csv", index=False
)
print(f"\nSaved 5 plots -> {OUT_DIR}")
print(f"Saved summary CSV -> csvs/ameriflux_summary.csv")
