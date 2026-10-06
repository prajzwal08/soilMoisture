"""§68 step A: Overijssel grid tiles + TWENTE control tiles -> a station_splits-schema CSV.

Every existing script derives its folder name with one of three rules (downloaders:
`{network}_{station_id}` when source_network == network; splits_config.station_dir_name:
`{source_network}_{station_id}` for non-ISMN; precompute_terramind likewise). Setting
source_network == network == "NLGRID" makes all three give `NLGRID_{station_id}`.

has_soil_moisture=True puts every tile in sm_only (False/False would be flux_only, which the frozen
checkpoint's category filter drops). Control tiles are centred EXACTLY on TWENTE stations inside the
province, so their inputs and predictions can be compared with the existing station stores (§68.5 S2/S4).

Output: {OUT_ROOT}/station_splits.csv  (OUT_ROOT on scratch: the project space is at 99.6 % of its inode quota)
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

TILES_CSV = "csvs/nl_tiles.csv"
PROV_FILE = "data/nl/provincie_2024.json"
STATION_SPLITS = "csvs/station_splits.csv"
NET = "NLGRID"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--province", default="Overijssel")
    ap.add_argument("--out-root", default="/gpfs/scratch1/shared/pkhanal/nl/data")
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20251231")
    args = ap.parse_args()

    t = pd.read_csv(TILES_CSV)
    t = t[t.keep & (t.province == args.province)].copy()
    t["station_id"] = [f"T{ix:03d}_{iy:03d}" for ix, iy in zip(t.ix, t.iy)]
    t["control_of"] = ""

    # control tiles: TWENTE stations whose location is inside the province polygon
    prov = gpd.read_file(PROV_FILE).to_crs(4326)
    poly = prov[prov.statnaam == args.province].union_all()
    st = pd.read_csv(STATION_SPLITS)
    st = st[st.network == "TWENTE"]
    pts = gpd.GeoSeries(gpd.points_from_xy(st.longitude, st.latitude), crs=4326)
    st = st[pts.within(poly).values]
    ctrl = pd.DataFrame({
        "station_id": [f"C_{n}" for n in st.station_name],
        "latitude": st.latitude.values, "longitude": st.longitude.values,
        "province": args.province, "land_frac": 1.0, "centre_on_land": True,
        "control_of": [f"ISMN_TWENTE_{n}" for n in st.station_name],
    })

    df = pd.concat([t[["station_id", "latitude", "longitude", "province", "x_rd", "y_rd", "ix", "iy",
                       "land_frac", "centre_on_land", "control_of"]], ctrl], ignore_index=True)
    df.insert(0, "source_network", NET)
    df.insert(1, "network", NET)
    df["station_name"] = df["station_id"]
    df["start_date"] = args.start
    df["end_date"] = args.end
    df["has_soil_moisture"] = True
    df["has_flux"] = False
    df["split"] = "nl_grid"
    df["soil_patch_ok"] = True            # placeholder; recomputed from the soil patch NaN % (§68.3 C4)
    df["is_control"] = df["control_of"] != ""

    # the three naming rules must agree
    dl = [f"{n}_{s}" if a == n else f"{a}_{n}_{s}" for a, n, s in zip(df.source_network, df.network, df.station_id)]
    sd = [f"{a}_{s}" for a, s in zip(df.source_network, df.station_id)]
    assert dl == sd, "downloader and splits_config folder names disagree"
    assert len(set(sd)) == len(sd), "duplicate folder names"
    df["dir_name"] = sd

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "station_splits.csv", index=False)
    print(f"{args.province}: {(~df.is_control).sum()} grid tiles + {df.is_control.sum()} control tiles "
          f"({', '.join(df[df.is_control].control_of)}) -> {out / 'station_splits.csv'}")


if __name__ == "__main__":
    main()
