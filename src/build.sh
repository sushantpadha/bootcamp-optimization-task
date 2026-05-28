#!/usr/bin/env bash
# reference/build.sh — Builds the reference implementation.
# Run from the spawn-grid project root.
#
# Override the compiler with the CXX environment variable:
#   CXX=clang++-18 bash reference/build.sh

set -euo pipefail

CXX="${CXX:-g++-14}"
# CXXFLAGS="-std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra"
CXXFLAGS="-std=c++23 -O3 -mcpu=native -Wall -Wextra"
OUTPUT="${OUTPUT:-spawn_sim}"
INPUT="${INPUT:-src/spawn_sim_neon_ptrstep_eor3_dualpartial_rowstep_inplace.cpp}"

echo "Building with $CXX from $INPUT ..."
"$CXX" $CXXFLAGS "$INPUT" -o "$OUTPUT" -lpthread
echo "Done: $OUTPUT"
