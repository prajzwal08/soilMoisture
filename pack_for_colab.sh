#!/usr/bin/env bash
# Pack the 5 trial stations + ISMN labels for Google Drive upload.
# Run from any directory:  bash pack_for_colab.sh
# Output: ~/colab_trial_data.tar.gz  (~3.5 GB)

set -e

SAT_DIR="$HOME/data/satellite"
ISMN_DIR="$HOME/data/soilmoisture/level1"
OUT="$HOME/colab_trial_data.tar.gz"

STATIONS=(
    "GROW_0ednwhg5"
    "SCAN_TncFortBayou"
    "TWENTE_ITCSM-07b"
    "SMOSMANIA_Lahas"
    "SNOTEL_HappyJack"
)

echo "Packing satellite data for ${#STATIONS[@]} stations..."
PATHS=()
for s in "${STATIONS[@]}"; do
    PATHS+=("$SAT_DIR/$s")
done
PATHS+=("$ISMN_DIR")

tar -czf "$OUT" "${PATHS[@]}"

SIZE=$(du -sh "$OUT" | cut -f1)
echo ""
echo "Done: $OUT  ($SIZE)"
echo ""
echo "Next steps:"
echo "  1. Upload $OUT to Google Drive"
echo "  2. Open trial_training_colab.ipynb in Colab"
echo "  3. Run Cell 1 — it mounts Drive and extracts the archive"
