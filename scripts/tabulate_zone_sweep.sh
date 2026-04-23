#!/bin/bash
# Tabulate results produced by zone_sweep.sbatch.
#
# Usage:
#   scripts/tabulate_zone_sweep.sh                 # auto-pick latest zone_sweep-*_*.out set
#   scripts/tabulate_zone_sweep.sh <ARRAY_JOB_ID>  # tabulate a specific array-job id
#   scripts/tabulate_zone_sweep.sh file1 file2 ... # tabulate explicit list
#
# Hyperion prints one CSV-ish line after the header
#   "iterations, zones, avg_wall_time, avg_gpu_time, total_wall_time, total_gpu_time, cycles"
# This script parses that line out of each *.out file, augments with
# derived quantities (per-zone cost, throughput, zones/CU on the
# 104-CU MI250 GCD), and prints a markdown-ready table plus a CSV copy.

set -e

NUM_CUS=${NUM_CUS:-104}      # override for non-MI250x devices
REF_ZONES=${REF_ZONES:-104}  # baseline column for speedup ratios

# -------------------------------------------------------------
# Resolve input file list
# -------------------------------------------------------------
if [ $# -eq 0 ]; then
    # find most recent zone_sweep-<ARRAYID>_*.out set
    latest=$(ls -1t zone_sweep-*_*.out 2>/dev/null | head -1 || true)
    if [ -z "$latest" ]; then
        echo "No zone_sweep-*_*.out files found and no args given." >&2
        echo "Usage: $0 [ARRAY_JOB_ID | file1 file2 ...]" >&2
        exit 1
    fi
    jobid=$(echo "$latest" | sed -E 's/zone_sweep-([0-9]+)_[0-9]+\.out/\1/')
    FILES=$(ls -1 zone_sweep-${jobid}_*.out | sort -t_ -k2 -n)
    echo "# auto-detected array job id: ${jobid}"
elif [ $# -eq 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    FILES=$(ls -1 zone_sweep-$1_*.out 2>/dev/null | sort -t_ -k2 -n)
    if [ -z "$FILES" ]; then
        echo "No zone_sweep-$1_*.out files found." >&2
        exit 1
    fi
else
    FILES="$@"
fi

# -------------------------------------------------------------
# Parse each file into a tab-separated row
#   zones  iters  avg_wall  avg_gpu  total_gpu  cycles  file
# -------------------------------------------------------------
tmp=$(mktemp)
for f in $FILES; do
    line=$(grep -A1 "iterations, zones, avg_wall_time" "$f" 2>/dev/null | tail -1)
    if [ -z "$line" ]; then
        echo "WARN: no timing line in $f (job may still be running or failed)" >&2
        continue
    fi
    iters=$(echo "$line"  | awk '{print $1}')
    zones=$(echo "$line"  | awk '{print $2}')
    awall=$(echo "$line"  | awk '{print $3}')
    agpu=$(echo "$line"   | awk '{print $4}')
    tgpu=$(echo "$line"   | awk '{print $6}')
    cycles=$(echo "$line" | awk '{print $7}')
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$zones" "$iters" "$awall" "$agpu" "$tgpu" "$cycles" "$f" >> $tmp
done

if [ ! -s $tmp ]; then
    echo "No rows parsed." >&2
    rm -f $tmp
    exit 1
fi

# Sort numerically by zones
sort -n -k1,1 $tmp -o $tmp

# -------------------------------------------------------------
# Find reference gpu time for REF_ZONES (for speedup ratio col)
# -------------------------------------------------------------
ref_gpu=$(awk -v z=$REF_ZONES '$1==z {print $4; exit}' $tmp)
ref_perzone=""
if [ -n "$ref_gpu" ]; then
    ref_perzone=$(awk -v g=$ref_gpu -v z=$REF_ZONES 'BEGIN{printf "%.6f", g/z}')
fi

# -------------------------------------------------------------
# Emit markdown table
# -------------------------------------------------------------
echo
echo "## Zone-sweep results (${NUM_CUS}-CU device, reference=${REF_ZONES}z)"
echo
printf "| %-6s | %-5s | %-8s | %-8s | %-10s | %-10s | %-12s | %-10s | %-14s |\n" \
    zones iters wall_s gpu_s zones/CU us/zone speedup/${REF_ZONES}z efficiency file
printf "|%s|%s|%s|%s|%s|%s|%s|%s|%s|\n" \
    "-------:" "------:" "--------:" "--------:" "----------:" "----------:" "--------------:" "----------:" "---------------"
while IFS=$'\t' read -r zones iters awall agpu tgpu cycles f; do
    zpercu=$(awk -v z=$zones -v c=$NUM_CUS 'BEGIN{printf "%.2f", z/c}')
    us=$(awk -v g=$agpu -v z=$zones 'BEGIN{printf "%.1f", (g/z)*1.0e6}')
    if [ -n "$ref_gpu" ]; then
        # speedup relative to per-call time on REF_ZONES; zones scaled so
        # that at perfect scaling (gpu_s grows linearly with zones) we get
        # speedup == zones/REF_ZONES.
        speedup=$(awk -v g=$agpu -v rg=$ref_gpu -v z=$zones -v rz=$REF_ZONES \
                  'BEGIN{printf "%.3fx", (rg/rz)*z/g}')
        # efficiency = per-zone cost at ref / per-zone cost here
        eff=$(awk -v rpz=$ref_perzone -v g=$agpu -v z=$zones \
              'BEGIN{pz=g/z; if(pz>0) printf "%.1f%%", 100.0*rpz/pz; else printf "--"}')
    else
        speedup="--"
        eff="--"
    fi
    printf "| %6s | %5s | %8.4f | %8.4f | %10s | %10s | %14s | %10s | %-14s |\n" \
        "$zones" "$iters" "$awall" "$agpu" "$zpercu" "$us" "$speedup" "$eff" "$(basename $f)"
done < $tmp

echo
echo "_us/zone = per-zone GPU cost in microseconds."
echo "_speedup/${REF_ZONES}z = (ref_gpu / ref_zones) * zones / gpu_s; 1.00x means perfect strong scaling from the reference point._"
echo "_efficiency = (per-zone cost at ${REF_ZONES}z) / (per-zone cost here); >100% means larger batches are faster per zone (good), <100% means overhead per zone is growing (bad)._"
echo

# -------------------------------------------------------------
# Also emit a CSV copy for later re-tabulation / plotting
# -------------------------------------------------------------
csv=zone_sweep_tabulated_$(date +%Y%m%d_%H%M%S).csv
{
    echo "zones,iters,avg_wall_s,avg_gpu_s,zones_per_cu,us_per_zone,speedup_vs_${REF_ZONES}z,efficiency,total_gpu_s,cycles,file"
    while IFS=$'\t' read -r zones iters awall agpu tgpu cycles f; do
        zpercu=$(awk -v z=$zones -v c=$NUM_CUS 'BEGIN{printf "%.4f", z/c}')
        us=$(awk -v g=$agpu -v z=$zones 'BEGIN{printf "%.4f", (g/z)*1.0e6}')
        if [ -n "$ref_gpu" ]; then
            speedup=$(awk -v g=$agpu -v rg=$ref_gpu -v z=$zones -v rz=$REF_ZONES \
                      'BEGIN{printf "%.4f", (rg/rz)*z/g}')
            eff=$(awk -v rpz=$ref_perzone -v g=$agpu -v z=$zones \
                  'BEGIN{pz=g/z; if(pz>0) printf "%.4f", rpz/pz; else printf "0"}')
        else
            speedup=""
            eff=""
        fi
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$zones" "$iters" "$awall" "$agpu" "$zpercu" "$us" \
            "$speedup" "$eff" "$tgpu" "$cycles" "$(basename $f)"
    done < $tmp
} > $csv
echo "CSV written to $csv"

rm -f $tmp
