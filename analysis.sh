#!/usr/bin/env bash

set -euo pipefail

CPUSET="0-7"

SIM="./sim"
SIMD="./sim_d"

TEST_DIR="test_grids"
PLOTS_DIR="${TEST_DIR}/plots"
PERF_DIR="${TEST_DIR}/perf_outputs"

FINAL_GENS=10000
PERF_GENS=1000
PLOT_GENS=1000

mkdir -p "$PLOTS_DIR" "$PERF_DIR"

SOURCE=""
PLOT_MODE=0
CHECK_ONLY=0
SKIP_PERF=0

usage() {
    echo "Usage:"
    echo "  $0 <source.cpp> [--plots [gens]] [--check-only] [--no-perf]"
    exit 1
}

[[ $# -lt 1 ]] && usage

SOURCE="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plots)
            PLOT_MODE=1
            shift

            if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
                PLOT_GENS="$1"
                shift
            fi
            ;;
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        --no-perf)
            SKIP_PERF=1
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            usage
            ;;
    esac
done

find_grid() {
    local size="$1"

    find "$TEST_DIR" \
        -maxdepth 1 \
        -type f \
        -name "*_${size}.bin" \
        ! -name "*.expected.bin" \
        | sort | head -n1
}

find_512_grids() {
    find "$TEST_DIR" \
        -maxdepth 1 \
        -type f \
        -name "*_512.bin" \
        ! -name "*.expected.bin" \
        | sort
}

extract_event() {
    local file="$1"
    local event="$2"

    local line
    line=$(grep ",${event}," "$file" | head -n1 || true)

    [[ -z "$line" ]] && { echo "0"; return; }

    local val
    val=$(echo "$line" | cut -d, -f1 | tr -d ' ')

    case "$val" in
        "<notcounted>"|"<notsupported>"|"<not counted>"|"<not supported>"|"")
            echo "0"
            ;;
        *)
            echo "$val"
            ;;
    esac
}

median3() {
    printf "%s\n" "$@" | sort -n | awk 'NR==2 { print $1 }'
}

mean3() {
    awk -v a="$1" -v b="$2" -v c="$3" \
        'BEGIN { print (a+b+c)/3.0 }'
}

variance3() {
    awk -v a="$1" -v b="$2" -v c="$3" '
    BEGIN {
        mean=(a+b+c)/3.0
        var=((a-mean)^2 + (b-mean)^2 + (c-mean)^2)/3.0
        print var
    }'
}

echo "=== SETUP ==="

echo "Disable ASLR"
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space

echo
echo "Inspect THP"
cat /sys/kernel/mm/transparent_hugepage/enabled

echo
echo "=== COMPILE ==="

mv -f sim sim.old 2>/dev/null || true
mv -f sim_d sim_d.old 2>/dev/null || true

g++-14 "$SOURCE" \
    -std=c++23 \
    -O3 \
    -mcpu=native \
    -Wall \
    -Wextra \
    -lpthread \
    -funroll-loops \
    -o ./sim

g++-14 "$SOURCE" \
    -std=c++23 \
    -O3 \
    -mcpu=native \
    -Wall \
    -Wextra \
    -lpthread \
    -funroll-loops \
    -g \
    -fno-omit-frame-pointer \
    -o ./sim_d

echo
echo "=== CORRECTNESS ==="

FAILED=0

while IFS= read -r binfile; do

    expected="${binfile%.bin}.expected.bin"
    outfile="out_$(basename "$binfile")"

    echo
    echo "$(basename "$binfile")"

    taskset -c "$CPUSET" \
        "$SIM" \
        "$binfile" \
        "$outfile" \
        "$FINAL_GENS"

    if cmp -s "$outfile" "$expected"; then
        echo "CORRECT"
    else
        echo "WRONG"
        FAILED=1
    fi

    rm -f "$outfile"

done < <(find_512_grids)

[[ "$FAILED" -ne 0 ]] && exit 1

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo
    echo "Check-only done."
    exit 0
fi

BIN32=$(find_grid 32768)

GENERAL_STAT="${PERF_DIR}/perf_general.txt"
CACHE_STAT="${PERF_DIR}/perf_cache.txt"

CACHE_DATA="${PERF_DIR}/cache_misses.data"
CACHE_REPORT="${PERF_DIR}/cache_misses_report.txt"

CYCLES_DATA="${PERF_DIR}/cycles.data"
CYCLES_REPORT="${PERF_DIR}/cycles_report.txt"

if [[ "$SKIP_PERF" -eq 0 ]]; then
    echo
    echo "=== PERF STAT ==="
    echo
    echo "$BIN32"

    sudo perf stat \
        -e cycles,instructions,stalled-cycles-backend,stall_backend_mem \
        taskset -c "$CPUSET" \
        "$SIM" \
        "$BIN32" \
        out.bin \
        "$PERF_GENS" \
        2>&1 | tee "$GENERAL_STAT"

    echo

    sudo perf stat \
        -e cycles,instructions,L1-dcache-load-misses,l2d_cache_refill_rd,l2d_cache_wr \
        taskset -c "$CPUSET" \
        "$SIM" \
        "$BIN32" \
        out.bin \
        "$PERF_GENS" \
        2>&1 | tee "$CACHE_STAT"

    echo
    echo "=== PERF RECORD CACHE-MISSES ==="

    sudo perf record \
        -o "$CACHE_DATA" \
        -e cache-misses \
        -g \
        --call-graph dwarf \
        taskset -c "$CPUSET" \
        "$SIMD" \
        "$BIN32" \
        out.bin \
        "$PERF_GENS"

    sudo perf report \
        -i "$CACHE_DATA" \
        --stdio \
        --no-children \
        > "$CACHE_REPORT"

    head -n 20 "$CACHE_REPORT"

    echo
    echo "=== PERF RECORD CYCLES ==="

    sudo perf record \
        -o "$CYCLES_DATA" \
        -e cycles \
        -g \
        --call-graph dwarf \
        taskset -c "$CPUSET" \
        "$SIMD" \
        "$BIN32" \
        out.bin \
        "$PERF_GENS"

    sudo perf report \
        -i "$CYCLES_DATA" \
        --stdio \
        --no-children \
        > "$CYCLES_REPORT"

    head -n 20 "$CYCLES_REPORT"
fi

if [[ "$PLOT_MODE" -eq 1 ]]; then

    echo
    echo "=== PLOTS ==="

    EVENTS="cycles,instructions,cache-misses,dTLB-loads,dTLB-load-misses,stalled-cycles-backend"

    CSV="${PLOTS_DIR}/perf_${PLOT_GENS}.csv"

    echo "grid,cycles,instructions,cache_misses,dtlb_loads,dtlb_misses,backend_stalls,ipc,dtlb_rate" \
        > "$CSV"

    for SIZE in 512 2048 8192 32768; do

        GRID=$(find_grid "$SIZE")

        echo
        echo "$SIZE -> $(basename "$GRID")"

        cycles_vals=()
        instr_vals=()
        cache_vals=()
        tlb_load_vals=()
        tlb_miss_vals=()
        backend_vals=()

        for RUN in 1 2 3; do

            PERF_TMP="${PERF_DIR}/plot_${SIZE}_${RUN}.txt"

            sudo perf stat \
                -x , \
                -e "$EVENTS" \
                taskset -c "$CPUSET" \
                "$SIM" \
                "$GRID" \
                out.bin \
                "$PLOT_GENS" \
                2> "$PERF_TMP"

            cycles_vals+=("$(extract_event "$PERF_TMP" "cycles")")
            instr_vals+=("$(extract_event "$PERF_TMP" "instructions")")
            cache_vals+=("$(extract_event "$PERF_TMP" "cache-misses")")
            tlb_load_vals+=("$(extract_event "$PERF_TMP" "dTLB-loads")")
            tlb_miss_vals+=("$(extract_event "$PERF_TMP" "dTLB-load-misses")")
            backend_vals+=("$(extract_event "$PERF_TMP" "stalled-cycles-backend")")

        done

        cycles=$(median3 "${cycles_vals[@]}")
        instructions=$(median3 "${instr_vals[@]}")
        cache_misses=$(median3 "${cache_vals[@]}")
        dtlb_loads=$(median3 "${tlb_load_vals[@]}")
        dtlb_misses=$(median3 "${tlb_miss_vals[@]}")
        backend_stalls=$(median3 "${backend_vals[@]}")

        ipc=$(awk -v i="$instructions" -v c="$cycles" \
            'BEGIN { if (c==0) print 0; else print i/c }')

        dtlb_rate=$(awk -v m="$dtlb_misses" -v l="$dtlb_loads" \
            'BEGIN { if (l==0) print 0; else print m/l }')

        echo "${SIZE},${cycles},${instructions},${cache_misses},${dtlb_loads},${dtlb_misses},${backend_stalls},${ipc},${dtlb_rate}" \
            >> "$CSV"

    done

python3 plot_perf.py "$CSV" "$PLOTS_DIR"

fi

echo
echo "=== FINAL 10K RUNS ==="

echo
echo "=== FINAL 10K RUNS ==="

FINAL_TIMES=()

for RUN in 1 2 3; do

    echo
    echo "Run $RUN"

    RUN_LOG="${PERF_DIR}/final_run_${RUN}.txt"

    taskset -c "$CPUSET" \
        "$SIM" \
        "$BIN32" \
        out.bin \
        "$FINAL_GENS" \
        2>&1 | tee "$RUN_LOG"

    t=$(grep -Eo '[0-9]+\.[0-9]+ ms' "$RUN_LOG" | tail -n1 | awk '{print $1}')

    if [[ -z "$t" ]]; then
        echo "Failed to extract runtime from output."
        exit 1
    fi

    FINAL_TIMES+=("$t")

    echo "Captured: ${t} ms"

done

FINAL_MEDIAN=$(median3 "${FINAL_TIMES[@]}")
FINAL_MEAN=$(mean3 "${FINAL_TIMES[@]}")
FINAL_VARIANCE=$(variance3 "${FINAL_TIMES[@]}")

echo
echo "=== FINAL RUN STATS ==="

echo "Times:"
printf '  %s ms\n' "${FINAL_TIMES[@]}"

echo
echo "Median:   ${FINAL_MEDIAN} ms"
echo "Mean:     ${FINAL_MEAN} ms"
echo "Variance: ${FINAL_VARIANCE}"

echo
echo "=== SAVED OUTPUTS ==="

if [[ "$SKIP_PERF" -eq 0 ]]; then
    echo
    echo "Perf stat:"
    echo "  $GENERAL_STAT"
    echo "  $CACHE_STAT"

    echo
    echo "Perf record data:"
    echo "  $CACHE_DATA"
    echo "  $CYCLES_DATA"

    echo
    echo "Perf reports:"
    echo "  $CACHE_REPORT"
    echo "  $CYCLES_REPORT"

    echo
    echo "Useful annotate commands:"
    echo "  sudo perf annotate -i $CACHE_DATA"
    echo "  sudo perf annotate -i $CYCLES_DATA"
fi

echo
echo "Final timing logs:"
printf '  %s/final_run_%d.txt\n' "$PERF_DIR" 1 2 3

echo
echo "Plots:"
echo "  $PLOTS_DIR"


echo
echo "Done."
