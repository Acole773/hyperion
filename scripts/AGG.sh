#!/bin/bash

# Output files
OUT_TXT="aggregated_results.txt"
OUT_CSV="aggregated_results.csv"
TMP="tmp_agg.txt"

# Clean previous runs
rm -f "$OUT_TXT" "$OUT_CSV" "$TMP"

# Headers
echo "zones wall_time gpu_time overhead cycles" > "$OUT_TXT"
echo "zones,wall_time,gpu_time,overhead,cycles" > "$OUT_CSV"

# Loop over all result files; extract data lines
for file in result_*.txt; do
    awk '
        /zones, wall_time, gpu_time, cycles/ {
            getline
            print
        }
    ' "$file" >> "$TMP"
done

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
