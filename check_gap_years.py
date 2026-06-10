import zarr
from pathlib import Path

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SAT_ROOT = Path("/projects/prjs1968/satellite_zarr")

stations = [
    ("sm_only", "ISMN_REMEDHUS_ElCoto", 2021),
    ("sm_only", "ISMN_REMEDHUS_Zamarron", 2021),
    ("sm_only", "ISMN_USCRN_Joplin-24-N", 2016),
    ("flux_only", "AmeriFlux_US-Me2", 2018),
]

for cat, station, gap_year in stations:
    print(f"\n=== {station} (gap year {gap_year}) ===")
    path = ZARR_ROOT / cat / station
    try:
        zg = zarr.open_consolidated(str(path), mode="r")
    except Exception:
        zg = zarr.open_group(str(path), mode="r")

    for mod in ['s2', 's1_asc', 's1_desc']:
        if f"{mod}/dates" in zg:
            dates = sorted(int(d) for d in zg[f"{mod}/dates"][:])
            years = sorted(set(d // 10000 for d in dates))
            print(f"  token {mod}: {len(dates)} dates, years={years}, range=[{dates[0]}-{dates[-1]}]")
        else:
            print(f"  token {mod}: NOT PRESENT")

    sat_path = SAT_ROOT / f"{station}.zarr"
    if sat_path.exists():
        try:
            szg = zarr.open_consolidated(str(sat_path), mode="r")
        except Exception:
            szg = zarr.open_group(str(sat_path), mode="r")
        for mod in ['s2', 's1_asc', 's1_desc']:
            if f"{mod}/dates" in szg:
                dates = sorted(int(d) for d in szg[f"{mod}/dates"][:])
                gap_dates = [d for d in dates if d // 10000 == gap_year]
                near_dates = [d for d in dates if d // 10000 in (gap_year - 1, gap_year, gap_year + 1)]
                print(f"  satellite_zarr {mod}: {len(dates)} total dates, {len(gap_dates)} in {gap_year}")
                print(f"    near gap year ({gap_year-1}-{gap_year+1}): {near_dates}")
            else:
                print(f"  satellite_zarr {mod}: NOT PRESENT")
    else:
        print(f"  satellite_zarr NOT FOUND at {sat_path}")
