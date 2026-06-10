import zarr
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")


def _category(r):
    if r["has_soil_moisture"] and r["has_flux"]:
        return "sm_and_flux"
    if r["has_soil_moisture"]:
        return "sm_only"
    return "flux_only"


def _dir_name(r):
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def process(row):
    cat, station = row
    path = ZARR_ROOT / cat / station
    out = {"station": station, "category": cat,
           "min_s2": None, "min_s1_asc": None, "min_s1_desc": None}
    try:
        zg = zarr.open_consolidated(str(path), mode="r")
    except Exception:
        try:
            zg = zarr.open_group(str(path), mode="r")
        except Exception:
            return out

    for mod, key in [("s2", "min_s2"), ("s1_asc", "min_s1_asc"), ("s1_desc", "min_s1_desc")]:
        if f"{mod}/dates" in zg:
            dates = zg[f"{mod}/dates"][:]
            if len(dates):
                out[key] = int(min(int(d) for d in dates))
    return out


def main():
    df = pd.read_csv(SPLITS_CSV)
    df["cat"] = df.apply(_category, axis=1)
    df["dir_name"] = df.apply(_dir_name, axis=1)
    rows = list(zip(df["cat"], df["dir_name"]))

    with Pool(32) as pool:
        results = pool.map(process, rows)

    rdf = pd.DataFrame(results)

    for col in ["min_s2", "min_s1_asc", "min_s1_desc"]:
        valid = rdf[rdf[col].notna()]
        if len(valid):
            idx = valid[col].idxmin()
            print(f"{col}: earliest = {int(valid.loc[idx, col])}  station = {valid.loc[idx, 'station']}")
        else:
            print(f"{col}: no data")

    # overall earliest across all three
    all_mins = []
    for col in ["min_s2", "min_s1_asc", "min_s1_desc"]:
        all_mins.append(rdf[["station", col]].rename(columns={col: "date"}).dropna())
    combined = pd.concat(all_mins)
    combined["date"] = combined["date"].astype(int)
    idx = combined["date"].idxmin()
    print(f"\nOverall earliest s1/s2 image: {combined.loc[idx,'date']}  station = {combined.loc[idx,'station']}")


if __name__ == "__main__":
    main()
