#!/usr/bin/env bash
# reference/build.sh — Builds the reference implementation.
# Run from the spawn-grid project root.
#
# Override the compiler with the CXX environment variable:
#   CXX=clang++-18 bash reference/build.sh

set -euo pipefail

CXX="${CXX:-g++-14}"
CXXFLAGS="-std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra"
OUTPUT="${OUTPUT:-spawn_sim}"

echo "Building reference implementation with $CXX ..."
"$CXX" $CXXFLAGS reference/spawn_sim.cpp -o "$OUTPUT" -lpthread
echo "Done: $OUTPUT"
