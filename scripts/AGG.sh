#!/bin/bash

# Output files
OUT_TXT="aggregated_results.txt"
OUT_CSV="aggregated_results.csv"
TMP="tmp_agg.txt"

# Clean previous runs
rm -f "$OUT_TXT" "$OUT_CSV" "$TMP" "$TMP.sorted"

# Headers
echo "zones avg_wall_time avg_gpu_time overhead cycles" > "$OUT_TXT"
echo "zones avg_wall_time avg_gpu_time overhead cycles" > "$OUT_CSV"

# ------------------------------------------------------------
# Extract data from result files
# ------------------------------------------------------------
for file in result_*.txt; do
    awk '
	/iterations, zones/ {
            getline
	    gsub(",", " ")   # handle accidental CSV formatting
            if (NF >= 7) {
                # columns:
                # $2 = zones
                # $3 = avg_wall_time
                # $4 = avg_gpu_time
                # $7 = cycles
                print $2, $3, $4, $7
	    }
        }
    ' "$file" >> "$TMP"
done

if [ ! -s "$TMP" ]; then
    echo "ERROR: No data extracted. Check input file format."
    exit 1
fi

# Sort by zones (column 1)
sort -n "$TMP" > "$TMP.sorted"

# Add overhead column (TXT)
awk '{
    overhead = $2 - $3;
    printf "%d %f %f %f %s\n", $1, $2, $3, overhead, $4;
}' "$TMP.sorted" >> "$OUT_TXT"

# Add overhead column (CSV)
awk '{
    overhead = $2 - $3;
    printf "%d,%f,%f,%f,%s\n", $1, $2, $3, overhead, $4;
}' "$TMP.sorted" >> "$OUT_CSV"

# Cleanup
rm -f "$TMP" "$TMP.sorted"

echo "----------------------------------------"
echo "Aggregation complete!"
echo "TXT  → $OUT_TXT"
echo "CSV  → $OUT_CSV"
echo "----------------------------------------"

