"""
Extract soil sensor depths and measurement heights from ICOS Ecosystem Station
Labelling Report PDFs using regex (no API required).

Approach:
  1. Extract full text from each PDF using pdfplumber
  2. Parse sensor table rows with regex to find SWC/TS depth columns
  3. Parse EC system table to find measurement height
  4. Save results to a CSV for use in post-processing

Output:
  /home/khanalp/data/icos_raw/sensor_depths.csv

Usage:
  conda activate geo
  python extract_sensor_depths.py
"""

import re
from pathlib import Path

import pandas as pd
import pdfplumber

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path("/home/khanalp/data/icos_raw")
OUT_CSV = RAW_DIR / "sensor_depths.csv"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Sensor table rows end with SWC_x_x_x or TS_x_x_x preceded by 3 floats:
#   DEPTH  EASTWARD  NORTHWARD  VARIABLE_CODE
# e.g.: "SWC-DeltaT MLx -0.08 -3.9 2 SWC_1_1_1"
#       "BE-Bra_TS_1_1_1 -0.015 -35.58 -9.42 TS_1_1_1"
_SENSOR_ROW = re.compile(
    r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(SWC|TS)_\d+_\d+_\d+\s*$"
)

# Variable code only (no depth on same line) — used for window search
_VAR_CODE = re.compile(r"(SWC|TS)_\d+_\d+_\d+")

# 3-number group for depth extraction: DEPTH  EAST  NORTH
# Requires the first number to be at the start of the string or preceded by whitespace,
# avoiding false matches from embedded serial numbers like -001 in "TH-ICOS-002-001",
# and from single channel IDs like "CS65X -1" (which have fewer than 3 numbers).
_DEPTH_TRIPLE = re.compile(r"(?:(?<=\s)|^)(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)")

# EC measurement height: line like "HEIGHT (m) 32.6 32.6"
_EC_HEIGHT = re.compile(r"HEIGHT\s*\(m\)\s+([\d.]+)")

# Canopy/vegetation height mentioned in free text
_CANOPY = re.compile(
    r"(?:canopy|tree|vegetation|stand)\s+height[^0-9]*?([\d.]+)\s*m",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def extract_pdf_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append(f"--- Page {i+1} ---\n{text}")
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Regex-based extraction
# ---------------------------------------------------------------------------
def _find_depth_in_window(lines: list[str], center: int, window: int = 2) -> float | None:
    """Search lines within [center-1, center+window] for a valid soil depth (-3 to -0.001 m).

    Looks for the first number of a 3-float group (DEPTH EAST NORTH) where the first
    number is preceded by whitespace or start of string. This avoids:
      - Serial number suffixes like -001 in 'TH-ICOS-002-001'
      - Channel IDs like 'CS65X -1' (only one number, not 3)
    """
    for delta in range(-1, window + 1):
        j = center + delta
        if not (0 <= j < len(lines)):
            continue
        for m in _DEPTH_TRIPLE.finditer(lines[j]):
            d = float(m.group(1))
            if -3.0 <= d <= -0.001:
                return d
    return None


def extract_with_regex(site_id: str, text: str) -> dict:
    swc_depths = set()
    ts_depths = set()
    measurement_height = None
    canopy_height = None

    lines = text.splitlines()
    for i, line in enumerate(lines):
        # --- SWC / TS extraction ---
        m = _SENSOR_ROW.search(line)
        if m:
            # Depth and variable code on the same line (most common format)
            depth = float(m.group(1))
            var_type = m.group(4)
            if var_type == "SWC":
                swc_depths.add(depth)
            elif var_type == "TS" and depth < 0:
                ts_depths.add(depth)
        else:
            # Fallback: variable code found but depth is on a nearby line
            # (SWCTEMP sensors in DE/FI sites split table rows across lines)
            vc = _VAR_CODE.search(line)
            if vc:
                var_type = vc.group(1)
                depth = _find_depth_in_window(lines, i)
                if depth is not None:
                    if var_type == "SWC":
                        swc_depths.add(depth)
                    elif var_type == "TS":
                        ts_depths.add(depth)

        # --- Measurement height ---
        if measurement_height is None:
            m = _EC_HEIGHT.search(line)
            if m:
                measurement_height = float(m.group(1))

        # --- Canopy height ---
        if canopy_height is None:
            m = _CANOPY.search(line)
            if m:
                canopy_height = float(m.group(1))

    return {
        "SITE_ID":              site_id,
        "SWC_DEPTHS_M":         sorted(swc_depths),
        "TS_DEPTHS_M":          sorted(ts_depths),
        "MEASUREMENT_HEIGHT_M": measurement_height,
        "CANOPY_HEIGHT_M":      canopy_height,
        "ERROR":                None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pdfs = sorted(RAW_DIR.glob("*/metadata/*.pdf"))
    print(f"Found {len(pdfs)} labelling report PDFs\n")

    records = []
    for pdf_path in pdfs:
        site_id = pdf_path.parent.parent.name
        print(f"Processing {site_id}: {pdf_path.name}")

        try:
            text = extract_pdf_text(pdf_path)
            if not text.strip():
                print("  WARNING: no text extracted (scanned PDF?)")
                records.append({"SITE_ID": site_id, "ERROR": "no text extracted"})
                continue
        except Exception as e:
            print(f"  ERROR reading PDF: {e}")
            records.append({"SITE_ID": site_id, "ERROR": str(e)})
            continue

        result = extract_with_regex(site_id, text)
        print(f"  SWC depths (m):    {result['SWC_DEPTHS_M']}")
        print(f"  TS depths (m):     {result['TS_DEPTHS_M']}")
        print(f"  Measurement ht:    {result['MEASUREMENT_HEIGHT_M']} m")
        print(f"  Canopy ht:         {result['CANOPY_HEIGHT_M']} m")
        result["PDF_FILE"] = pdf_path.name
        records.append(result)
        print()

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print(f"Total:   {len(df)} stations")
    print(f"Success: {df['ERROR'].isna().sum()}")
    print(f"Errors:  {df['ERROR'].notna().sum()}")

    success = df[df["ERROR"].isna()]
    if len(success):
        print("\nPreview:")
        print(success[["SITE_ID", "SWC_DEPTHS_M", "TS_DEPTHS_M",
                        "MEASUREMENT_HEIGHT_M", "CANOPY_HEIGHT_M"]].to_string(index=False))


if __name__ == "__main__":
    main()
