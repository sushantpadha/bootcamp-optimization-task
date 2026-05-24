#!/usr/bin/env bash
# harness/run.sh — Grading harness for the Monster Spawning Grid assignment.
#
# Runs a participant's spawn_sim binary against a test grid, verifies correctness,
# and reports simulation timing statistics.
#
# Usage:
#   bash harness/run.sh [OPTIONS] <binary> <input.bin> <expected.bin>
#
# Options:
#   -n <N>        Number of iterations (default: 10)
#   -c <CPULIST>  CPU list for taskset (default: 0-7)
#   -o <file>     Save the binary output of the last run here (default: /tmp/spawn_out.bin)
#   -h            Print this help and exit
#
# Environment controls that affect stability:
#   ASLR:          Disabled via 'setarch -R' (no root required).
#   CPU affinity:  Set via 'taskset -c' (no root required).
#   FS caches:     Dropped via 'echo 3 > /proc/sys/vm/drop_caches' (requires root).
#                  The harness will attempt this and print a warning if it fails.
#   CPU frequency: Cannot be reliably controlled on cloud VMs. Confirm that the
#                  c8g.2xlarge instance has frequency scaling disabled at OS level
#                  (check: cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor).
#
# The binary is expected to:
#   - Accept:  spawn_sim <input.bin> <output.bin>
#   - Print:   a single line "<N.NNN> ms" to stdout (simulation time only, not I/O)
#   - Exit 0 on success.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY="$SCRIPT_DIR/verify.py"

# ---- Defaults ---------------------------------------------------------------
ITERATIONS=10
CPULIST="0-7"
OUTPUT_FILE="/tmp/spawn_out_$$.bin"
CLEANUP_OUTPUT=true

# ---- Argument parsing -------------------------------------------------------
usage() {
    sed -n '/^# Usage/,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 1
}

while getopts "n:c:o:h" opt; do
    case "$opt" in
        n) ITERATIONS="$OPTARG" ;;
        c) CPULIST="$OPTARG"    ;;
        o) OUTPUT_FILE="$OPTARG"; CLEANUP_OUTPUT=false ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -ne 3 ]]; then
    echo "Error: expected 3 positional arguments: <binary> <input.bin> <expected.bin>" >&2
    usage
fi

BINARY="$1"
INPUT="$2"
EXPECTED="$3"

for f in "$BINARY" "$INPUT" "$EXPECTED"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: file not found: $f" >&2
        exit 1
    fi
done

if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary not executable: $BINARY" >&2
    exit 1
fi

# ---- Print environment summary ----------------------------------------------
echo "============================================================"
echo " Monster Spawning Grid — Grading Harness"
echo "============================================================"
echo "Date       : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Host       : $(hostname)"
echo "OS         : $(uname -srm)"
if [[ -f /etc/os-release ]]; then
    echo "Distro     : $(. /etc/os-release && echo "$PRETTY_NAME")"
fi
echo "Binary     : $BINARY"
echo "Input      : $INPUT ($(du -sh "$INPUT" 2>/dev/null | cut -f1))"
echo "Expected   : $EXPECTED"
echo "Iterations : $ITERATIONS"
echo "CPU list   : $CPULIST"

# Detect governor
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    echo "CPU governor: $GOV"
    if [[ "$GOV" != "performance" ]]; then
        echo "WARNING: CPU governor is '$GOV', not 'performance'." \
             "Timing may be unstable." >&2
    fi
else
    echo "CPU governor: (not readable — may be fixed on this VM)"
fi

# Attempt to drop filesystem caches
if echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
    DROP_CACHES=true
    echo "FS caches  : will drop between runs"
else
    DROP_CACHES=false
    echo "FS caches  : cannot drop (no root) — first run may be warmer than subsequent"
fi

echo "ASLR       : disabled via setarch -R"
echo "============================================================"
echo ""

# ---- Run iterations ---------------------------------------------------------
declare -a SIM_TIMES_MS

for ((i = 1; i <= ITERATIONS; i++)); do

    # Drop FS caches before each run so the input file is not in the page cache.
    if [[ "$DROP_CACHES" == "true" ]]; then
        echo 3 > /proc/sys/vm/drop_caches
    fi

    # Run the binary under taskset (CPU affinity) and setarch -R (disable ASLR).
    # Capture stdout (the "N.NNN ms" timing line) and let stderr pass through.
    RAW_OUTPUT=$(taskset -c "$CPULIST" setarch "$(uname -m)" -R \
        "$BINARY" "$INPUT" "$OUTPUT_FILE" 2>&1 || true)

    # Extract the simulation time reported by the binary.
    # Expected format: a single line matching /[0-9]+(\.[0-9]+)? ms/
    SIM_MS=$(echo "$RAW_OUTPUT" | grep -oP '[0-9]+(\.[0-9]+)?' | head -1)
    if [[ -z "$SIM_MS" ]]; then
        echo "Error: binary produced no timing output on iteration $i." >&2
        echo "  stdout/stderr was: $RAW_OUTPUT" >&2
        exit 1
    fi

    SIM_TIMES_MS+=("$SIM_MS")
    printf "  Run %2d / %d : %s ms\n" "$i" "$ITERATIONS" "$SIM_MS"
done

echo ""

# ---- Correctness check ------------------------------------------------------
echo -n "Correctness : "
python3 "$VERIFY" "$EXPECTED" "$OUTPUT_FILE"
VERIFY_EXIT=$?

if [[ "$CLEANUP_OUTPUT" == "true" ]]; then
    rm -f "$OUTPUT_FILE"
fi

if [[ $VERIFY_EXIT -ne 0 ]]; then
    echo ""
    echo "GRADING RESULT: FAIL (incorrect output)"
    exit 1
fi

# ---- Statistics (median, min, max, stddev) ----------------------------------
echo ""
echo "Timing statistics (simulation only, ms):"

python3 - "${SIM_TIMES_MS[@]}" <<'PYEOF'
import sys
import math

vals = [float(x) for x in sys.argv[1:]]
n = len(vals)
vals_sorted = sorted(vals)

# median
if n % 2 == 1:
    median = vals_sorted[n // 2]
else:
    median = (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2.0

mn  = vals_sorted[0]
mx  = vals_sorted[-1]
avg = sum(vals) / n
var = sum((v - avg) ** 2 for v in vals) / n
sd  = math.sqrt(var)
cv  = (sd / avg * 100) if avg > 0 else 0.0

print(f"  n          : {n}")
print(f"  median     : {median:.3f} ms")
print(f"  min        : {mn:.3f} ms")
print(f"  max        : {mx:.3f} ms")
print(f"  mean       : {avg:.3f} ms")
print(f"  std dev    : {sd:.3f} ms")
print(f"  CV (sd/μ)  : {cv:.1f}%")
if cv > 5.0:
    print("  WARNING: high run-to-run variance (CV > 5%). Check for background load.")
PYEOF

echo ""
echo "============================================================"
echo " GRADING RESULT: PASS"
echo "============================================================"
