#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 file1 file2"
    exit 1
fi

file1="$1"
file2="$2"

# --- settings ---
ABS_TOL=1e-12
REL_TOL=1e-8

tmp1=$(mktemp)
tmp2=$(mktemp)

# --- extract table ---
extract_table() {
    awk '
    /Result/ {flag=1; next}
    flag && NF==2 {
        print $1, $2
        count++
        if (count==150) exit
    }' "$1"
}

extract_table "$file1" > "$tmp1"
extract_table "$file2" > "$tmp2"

# sanity check
n1=$(wc -l < "$tmp1")
n2=$(wc -l < "$tmp2")

if [ "$n1" -ne 150 ] || [ "$n2" -ne 150 ]; then
    echo "ERROR: Did not extract 150 entries from both files"
    echo "file1 entries: $n1"
    echo "file2 entries: $n2"
    exit 1
fi

# --- comparison ---
awk -v abs_tol="$ABS_TOL" -v rel_tol="$REL_TOL" '
BEGIN {
    max_abs = 0
    max_rel = 0
    failures = 0
}

NR==FNR {
    val1[$1] = $2
    next
}

{
    idx = $1
    v1 = val1[idx]
    v2 = $2

    abs_diff = (v1 > v2) ? v1 - v2 : v2 - v1
    rel_diff = (abs_diff > 0) ? abs_diff / ((v1>0)?v1:1) : 0

    if (abs_diff > max_abs) max_abs = abs_diff
    if (rel_diff > max_rel) max_rel = rel_diff

    if (abs_diff > abs_tol && rel_diff > rel_tol) {
        printf("FAIL idx=%d  v1=%e  v2=%e  abs=%e  rel=%e\n",
               idx, v1, v2, abs_diff, rel_diff)
        failures++
    }
}

END {
    printf("\nSummary:\n")
    printf("Max abs diff: %e\n", max_abs)
    printf("Max rel diff: %e\n", max_rel)
    printf("Failures: %d\n", failures)

    if (failures > 0) {
        exit 2
    } else {
        print "PASS"
    }
}
' "$tmp1" "$tmp2"

rm -f "$tmp1" "$tmp2"
