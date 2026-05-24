#!/usr/bin/env python3
"""
generate.py — Generates input and expected output grids for the Monster Spawning Grid.

Requires: numpy (sudo apt-get install python3-numpy)

Usage:
    # Generate input .bin files only (fast):
    python3 test_grids/generate.py

    # Also generate expected outputs for small sizes (seconds to minutes):
    python3 test_grids/generate.py --generate-expected --sizes 512,2048

    # Expected outputs for medium sizes (run in screen, ~minutes each):
    python3 test_grids/generate.py --generate-expected --sizes 8192

    # Expected outputs for large sizes (run in screen, ~25h each):
    python3 test_grids/generate.py --generate-expected --sizes 32768

    # --sizes applies to both input and expected output generation.
    # Already-existing .expected.bin files are skipped unless --force is passed.
"""

import argparse
import struct
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("Error: numpy is required.  Run: sudo apt-get install python3-numpy")

EMPTY, EGG, JUVENILE, ADULT = 0, 1, 2, 3

DEFAULT_SIZES = [512, 2048, 8192, 32768]
DEFAULT_BINARY = Path(__file__).parent.parent / 'spawn_sim'


def write_grid(path: Path, n: int, cells: np.ndarray) -> None:
    assert cells.shape == (n, n)
    assert cells.dtype == np.uint8
    with open(path, 'wb') as f:
        f.write(struct.pack('<QQ', n, n))
        f.write(cells.tobytes())


# ---------------------------------------------------------------------------
# Grid generators  (each takes n = grid side length)
# ---------------------------------------------------------------------------

def gen_random_low(rng: np.random.Generator, n: int) -> np.ndarray:
    """~5% ADULT cells, randomly distributed.  Tests the common sparse case."""
    cells = np.zeros((n, n), dtype=np.uint8)
    cells[rng.random((n, n)) < 0.05] = ADULT
    return cells


def gen_random_high(rng: np.random.Generator, n: int) -> np.ndarray:
    """~50% ADULT cells, randomly distributed.  Tests dense computation."""
    cells = np.zeros((n, n), dtype=np.uint8)
    cells[rng.random((n, n)) < 0.50] = ADULT
    return cells


def gen_structured(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Repeating 3×3 blocks of ADULTs on a 10-cell pitch in both dimensions.
    Tests correctness on a perfectly regular, periodic input.
    """
    cells = np.zeros((n, n), dtype=np.uint8)
    period = 10
    block  = 3
    y_mask = np.arange(n) % period < block
    x_mask = np.arange(n) % period < block
    cells[np.outer(y_mask, x_mask)] = ADULT
    return cells


def gen_sparse_clusters(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Mostly-empty grid with dense circular clusters of ADULTs.
    Tests load imbalance in tiled parallel implementations.
    Cluster count and radius scale with grid size.
    """
    cells = np.zeros((n, n), dtype=np.uint8)
    num_clusters = max(4, int(50 * (n / 32768) ** 2))
    radius = max(4, int(20 * n / 32768))
    density = 0.80

    offsets = np.argwhere(
        (np.arange(-radius, radius + 1)[:, None] ** 2 +
         np.arange(-radius, radius + 1)[None, :] ** 2) <= radius ** 2
    ) - radius

    center_y = rng.integers(0, n, num_clusters)
    center_x = rng.integers(0, n, num_clusters)

    for cy, cx in zip(center_y, center_x):
        ny = (cy + offsets[:, 0]) % n
        nx = (cx + offsets[:, 1]) % n
        mask = rng.random(len(ny)) < density
        cells[ny[mask], nx[mask]] = ADULT

    return cells


def gen_boundary_stress(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Dense ADULT fill along all four edges (4 cells wide).
    Forces heavy use of the toroidal wrap in every participant's halo logic.
    """
    cells = np.zeros((n, n), dtype=np.uint8)
    density = 0.75
    border  = 4

    def fill_region(ys: np.ndarray, xs: np.ndarray) -> None:
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        mask = rng.random(yy.shape) < density
        cells[yy[mask], xx[mask]] = ADULT

    fill_region(np.arange(0, border),        np.arange(n))
    fill_region(np.arange(n - border, n),    np.arange(n))
    fill_region(np.arange(n),                np.arange(0, border))
    fill_region(np.arange(n),                np.arange(n - border, n))

    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PUBLIC_GRIDS = [
    ('public_1_random_low',      gen_random_low),
    ('public_2_random_high',     gen_random_high),
    ('public_3_structured',      gen_structured),
    ('public_4_sparse_clusters', gen_sparse_clusters),
    ('public_5_boundary_stress', gen_boundary_stress),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate Monster Spawning Grid input and expected output grids.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Requires:')[0].strip())
    parser.add_argument('--output-dir', default=str(Path(__file__).parent),
                        help='Directory to write grids (default: directory of this script)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed (default: 42)')
    parser.add_argument('--sizes', default=','.join(map(str, DEFAULT_SIZES)),
                        help=f'Comma-separated grid sizes (default: {",".join(map(str, DEFAULT_SIZES))})')
    parser.add_argument('--generate-expected', action='store_true',
                        help='Run the reference binary to produce .expected.bin files')
    parser.add_argument('--binary', default=str(DEFAULT_BINARY),
                        help=f'Path to spawn_sim binary (default: {DEFAULT_BINARY})')
    parser.add_argument('--generations', type=int, default=10000,
                        help='Generations to simulate for expected outputs (default: 10000)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing .expected.bin files (default: skip)')
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(',')]
    for s in sizes:
        if s <= 0 or (s & (s - 1)) != 0:
            sys.exit(f"Error: size {s} is not a power of two")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_inputs = []

    print(f"Sizes      : {sizes}")
    print(f"Output dir : {out_dir}/")
    print()

    # ── Generate input .bin files ─────────────────────────────────────────────
    print("Writing input grids ...")
    for size in sizes:
        print(f"  {size}×{size}:")
        for name, gen_fn in PUBLIC_GRIDS:
            path = out_dir / f'{name}_{size}.bin'
            print(f"    {path.name} ... ", end='', flush=True)
            cells = gen_fn(rng, size)
            write_grid(path, size, cells)
            adult_pct = float((cells == ADULT).sum()) / (size * size) * 100
            file_mib = (16 + size * size) / 1024 ** 2
            print(f"done  ({adult_pct:.1f}% adult, {file_mib:.1f} MiB)")
            all_inputs.append(path)

    # ── Generate expected outputs ─────────────────────────────────────────────
    if not args.generate_expected:
        print()
        print("Tip: run with --generate-expected --sizes 512,2048 to produce "
              ".expected.bin files for the small grids (fast).")
        return

    binary = Path(args.binary)
    if not binary.exists():
        sys.exit(f"Error: binary not found at '{binary}'. "
                 f"Run 'bash reference/build.sh' first.")

    print()
    print(f"Generating expected outputs ({args.generations} generations) ...")
    print(f"Binary     : {binary}")
    print()

    skipped = 0
    for path in all_inputs:
        expected = path.with_suffix('.expected.bin')
        if expected.exists() and not args.force:
            print(f"  {expected.name} ... skipped (already exists)")
            skipped += 1
            continue

        size = int(path.stem.split('_')[-1])
        est_hours = (size / 32768) ** 2 * 25 * (args.generations / 10000)
        est_str = (f"{est_hours:.0f}h" if est_hours >= 1
                   else f"{est_hours * 60:.0f}m" if est_hours * 60 >= 1
                   else f"{est_hours * 3600:.0f}s")

        print(f"  {expected.name}  (est. ~{est_str}) ... ", end='', flush=True)
        result = subprocess.run(
            [str(binary), str(path), str(expected), str(args.generations)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"FAILED\n    {result.stderr.strip()}")
        else:
            timing = result.stdout.strip()
            print(f"done  [{timing}]")

    if skipped:
        print(f"\n({skipped} file(s) skipped — use --force to overwrite)")


if __name__ == '__main__':
    main()
