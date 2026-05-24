#!/usr/bin/env python3
"""
smoke_test.py — Correctness tests, timing benchmarks, and TUI visualizer for spawn_sim.

Usage:
    python3 smoke_test.py                        # run correctness + timing tests
    python3 smoke_test.py --visualize            # animate 128×128 grid for 200 generations
    python3 smoke_test.py --visualize --vis-size 64 --vis-gens 500 --vis-delay 0.03

Small test cases (≤256 cells/side) are verified against a Python oracle.
Large test cases (≥2048 cells/side) are verified by running the binary twice and
comparing outputs — the Python oracle would take hours at those sizes.
"""

import argparse
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EMPTY, EGG, JUVENILE, ADULT = 0, 1, 2, 3

# (grid_size, generations, seed)
SMALL_CASES = [
    (32,   3,  12345),
    (64,   3,  12345),
    (64,   5,  99999),
    (128,  2,  42),
    (256,  1,  7),
]

LARGE_CASES = [
    (2048, 1000, 42),
    (4096,  100, 42),
]


# ── Python oracle (small cases only) ─────────────────────────────────────────

def count_adults(grid, cx, cy):
    n = len(grid)
    count = 0
    for dy in range(-2, 3):
        ny = (cy + dy) % n
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            nx = (cx + dx) % n
            if grid[ny][nx] == ADULT:
                count += 1
    return count

def step_python(grid):
    n = len(grid)
    new = [[0] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            A = count_adults(grid, x, y)
            cell = grid[y][x]
            if cell == EMPTY:
                new[y][x] = EGG if 3 <= A <= 5 else EMPTY
            elif cell == EGG:
                new[y][x] = JUVENILE
            elif cell == JUVENILE:
                new[y][x] = ADULT
            elif cell == ADULT:
                new[y][x] = ADULT if 4 <= A <= 9 else EMPTY
    return new

def simulate_python(grid, generations):
    for _ in range(generations):
        grid = step_python(grid)
    return grid


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_grid_file(path, grid):
    n = len(grid)
    with open(path, 'wb') as f:
        f.write(struct.pack('<QQ', n, n))
        for row in grid:
            f.write(bytes(row))

def write_flat_grid_file(path, n, flat):
    with open(path, 'wb') as f:
        f.write(struct.pack('<QQ', n, n))
        f.write(bytes(flat))

def read_grid_file(path, n):
    with open(path, 'rb') as f:
        w, h = struct.unpack('<QQ', f.read(16))
        assert w == n and h == n, f"unexpected dimensions {w}×{h}"
        cells = list(f.read(n * n))
    return [[cells[y * n + x] for x in range(n)] for y in range(n)]

def files_identical(path_a, path_b):
    with open(path_a, 'rb') as a, open(path_b, 'rb') as b:
        return a.read() == b.read()


# ── Build ─────────────────────────────────────────────────────────────────────

def build_binary(out_path: Path) -> None:
    build_sh = Path(__file__).parent / 'reference' / 'build.sh'
    result = subprocess.run(
        ['bash', str(build_sh)],
        env={**os.environ, 'OUTPUT': str(out_path)},
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("FAIL: build.sh failed")
        print(result.stderr)
        sys.exit(1)
    print(f"  {result.stdout.strip()}")


# ── Run helpers ───────────────────────────────────────────────────────────────

def run_binary(binary, input_bin, output_bin, generations):
    result = subprocess.run(
        [str(binary), str(input_bin), str(output_bin), str(generations)],
        capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# ── Small case: verify against Python oracle ──────────────────────────────────

def run_small_case(binary, grid_size, generations, seed, tmp):
    rng = random.Random(seed)
    init_grid = [
        [rng.choice([EMPTY, ADULT]) for _ in range(grid_size)]
        for _ in range(grid_size)
    ]
    expected = simulate_python(init_grid, generations)

    input_bin  = tmp / f'input_{grid_size}_{generations}_{seed}.bin'
    output_bin = tmp / f'output_{grid_size}_{generations}_{seed}.bin'
    write_grid_file(input_bin, init_grid)

    rc, timing, stderr = run_binary(binary, input_bin, output_bin, generations)
    if rc != 0:
        print(f"FAIL (exit {rc}): {stderr}")
        return False

    actual = read_grid_file(output_bin, grid_size)
    diffs = [
        (y, x, expected[y][x], actual[y][x])
        for y in range(grid_size)
        for x in range(grid_size)
        if expected[y][x] != actual[y][x]
    ]

    if diffs:
        print(f"FAIL — {len(diffs)} cell(s) differ  [{timing}]")
        for y, x, exp, act in diffs[:5]:
            print(f"    [{y:3d}][{x:3d}]  expected {exp}  got {act}")
        if len(diffs) > 5:
            print(f"    ... and {len(diffs) - 5} more")
        return False

    print(f"PASS  [{timing}]")
    return True


# ── Large case: verify binary against itself, report timing ───────────────────

def run_large_case(binary, grid_size, generations, seed, tmp):
    rng = random.Random(seed)
    flat = [rng.choice([EMPTY, ADULT]) for _ in range(grid_size * grid_size)]

    input_bin   = tmp / f'input_{grid_size}_{generations}_{seed}.bin'
    output_bin1 = tmp / f'output1_{grid_size}_{generations}_{seed}.bin'
    output_bin2 = tmp / f'output2_{grid_size}_{generations}_{seed}.bin'
    write_flat_grid_file(input_bin, grid_size, flat)

    rc, timing, stderr = run_binary(binary, input_bin, output_bin1, generations)
    if rc != 0:
        print(f"FAIL (exit {rc}): {stderr}")
        return False

    rc2, _, stderr2 = run_binary(binary, input_bin, output_bin2, generations)
    if rc2 != 0:
        print(f"FAIL on second run (exit {rc2}): {stderr2}")
        return False

    if not files_identical(output_bin1, output_bin2):
        print(f"FAIL — non-deterministic output  [{timing}]")
        return False

    print(f"PASS  [{timing}]")
    return True


# ── Visualizer ────────────────────────────────────────────────────────────────

# ANSI-colored characters for each cell state
CELL_DISPLAY = {
    EMPTY:    '\033[38;5;236m·\033[0m',  # dark gray dot
    EGG:      '\033[33m·\033[0m',         # yellow dot
    JUVENILE: '\033[36mo\033[0m',          # cyan o
    ADULT:    '\033[32m█\033[0m',          # green block
}

def run_visualizer(binary: Path, grid_size: int = 64, generations: int = 200,
                   seed: int = 42, delay: float = 0.05) -> None:
    rng = random.Random(seed)
    flat = bytes([rng.choice([EMPTY, ADULT]) for _ in range(grid_size * grid_size)])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cur_bin  = tmp / 'cur.bin'
        next_bin = tmp / 'next.bin'

        with open(cur_bin, 'wb') as f:
            f.write(struct.pack('<QQ', grid_size, grid_size))
            f.write(flat)

        sys.stdout.write('\033[?25l\033[2J')  # hide cursor, clear screen
        sys.stdout.flush()

        adult_counts = []

        try:
            for gen in range(generations):
                with open(cur_bin, 'rb') as f:
                    f.read(16)
                    data = f.read(grid_size * grid_size)

                n_adult    = data.count(ADULT)
                n_juvenile = data.count(JUVENILE)
                n_egg      = data.count(EGG)
                n_empty    = data.count(EMPTY)
                total      = grid_size * grid_size
                adult_counts.append(n_adult)

                buf = ['\033[H']  # move cursor to top-left
                buf.append(
                    f' Generation {gen+1:4d}/{generations}  '
                    f'█ Adult {n_adult:5d} ({100*n_adult/total:4.1f}%)  '
                    f'o Juvenile {n_juvenile:5d}  '
                    f'· Egg {n_egg:5d}  '
                    f'  Empty {n_empty:5d}\n'
                )
                for row in range(grid_size):
                    buf.append(
                        ''.join(CELL_DISPLAY[b]
                                for b in data[row * grid_size:(row + 1) * grid_size])
                    )
                    buf.append('\n')

                sys.stdout.write(''.join(buf))
                sys.stdout.flush()

                rc, _, stderr = run_binary(binary, cur_bin, next_bin, 1)
                if rc != 0:
                    print(f"\nError running binary: {stderr}")
                    break

                cur_bin, next_bin = next_bin, cur_bin
                time.sleep(delay)

        finally:
            sys.stdout.write('\033[?25h\n')  # restore cursor
            sys.stdout.flush()

    print(f"\nFinal state after {generations} generations:")
    print(f"  Adult:    {adult_counts[-1]:6d} / {total} ({100*adult_counts[-1]/total:.1f}%)")
    if adult_counts[-1] == 0:
        print("  WARNING: all adults dead — grid went extinct. "
              "Consider a different seed or initial density.")
    else:
        peak = max(adult_counts)
        peak_gen = adult_counts.index(peak) + 1
        print(f"  Peak adult count: {peak} at generation {peak_gen}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--visualize',  action='store_true',
                        help='Run the TUI visualizer instead of the test suite')
    parser.add_argument('--vis-size',   type=int,   default=64,
                        help='Grid size for visualizer (default: 128)')
    parser.add_argument('--vis-gens',   type=int,   default=200,
                        help='Generations to animate (default: 200)')
    parser.add_argument('--vis-seed',   type=int,   default=42,
                        help='RNG seed for visualizer (default: 42)')
    parser.add_argument('--vis-delay',  type=float, default=0.05,
                        help='Seconds between frames (default: 0.05)')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        binary = tmp / 'spawn_sim'
        build_binary(binary)

        if args.visualize:
            print()
            run_visualizer(binary, args.vis_size, args.vis_gens,
                           args.vis_seed, args.vis_delay)
            return

        all_passed = True

        print()
        print("Small cases (Python oracle):")
        for grid_size, generations, seed in SMALL_CASES:
            label = f"  {grid_size:4d}×{grid_size:<4d}  {generations:4d} gen  seed={seed}"
            print(f"{label} ... ", end='', flush=True)
            if not run_small_case(binary, grid_size, generations, seed, tmp):
                all_passed = False

        print()
        print("Large cases (determinism check + timing):")
        for grid_size, generations, seed in LARGE_CASES:
            label = f"  {grid_size:4d}×{grid_size:<4d}  {generations:4d} gen  seed={seed}"
            print(f"{label} ... ", end='', flush=True)
            if not run_large_case(binary, grid_size, generations, seed, tmp):
                all_passed = False

    print()
    total = len(SMALL_CASES) + len(LARGE_CASES)
    if all_passed:
        print(f"PASS: all {total} cases passed")
    else:
        print("FAIL: one or more cases failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
