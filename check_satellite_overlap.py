import csv
from datetime import datetime, date

CSV_PATH = "/home/khanalp/data/soilmoisture/level1/station_metadata.csv"

CUTOFF = date(2016, 1, 1)

rows = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        if row["status"] != "saved":
            continue
        try:
            start  = datetime.strptime(row["start_date"].strip(), "%Y%m%d").date()
            end    = datetime.strptime(row["end_date"].strip(),   "%Y%m%d").date()
            n_days = int(float(row["n_days"]))
        except (ValueError, KeyError):
            continue
        rows.append({"start": start, "end": end, "n_days": n_days})

subset = [r for r in rows if r["end"] >= CUTOFF]

overlap_days = []
for r in subset:
    eff_start = max(r["start"], CUTOFF)
    od = (r["end"] - eff_start).days + 1
    if od > 0:
        overlap_days.append(od)

total_days = sum(overlap_days)
avg_days   = total_days / len(overlap_days) if overlap_days else 0
avg_years  = avg_days / 365.25

print(f"Total saved stations:      {len(rows)}")
print(f"Stations with data in/after 2016-01-01: {len(subset)}")
print(f"  Total days (overlap):    {total_days:,}")
print(f"  Avg days/station:        {avg_days:.0f}  ({avg_years:.1f} yrs)")
print(f"  Min overlap:             {min(overlap_days)} days  ({min(overlap_days)/365.25:.1f} yrs)")
print(f"  Max overlap:             {max(overlap_days)} days  ({max(overlap_days)/365.25:.1f} yrs)")
