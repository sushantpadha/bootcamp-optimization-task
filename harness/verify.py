#!/usr/bin/env python3
"""
verify.py — Bit-identical comparison of two Monster Spawning Grid output files.

Usage:
    python3 verify.py <expected.bin> <actual.bin>

Exit codes:
    0  files are bit-identical (PASS)
    1  files differ or are unreadable (FAIL)
"""

import struct
import sys


HEADER_BYTES = 16  # two uint64_t values


def parse_header(data: bytes, path: str) -> tuple[int, int]:
    if len(data) < HEADER_BYTES:
        print(f"FAIL: '{path}' is too short to contain a header ({len(data)} bytes).",
              file=sys.stderr)
        sys.exit(1)
    width, height = struct.unpack_from('<QQ', data, 0)
    return int(width), int(height)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <expected.bin> <actual.bin>", file=sys.stderr)
        sys.exit(1)

    expected_path = sys.argv[1]
    actual_path   = sys.argv[2]

    try:
        with open(expected_path, 'rb') as f:
            expected = f.read()
    except OSError as e:
        print(f"FAIL: cannot read expected file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(actual_path, 'rb') as f:
            actual = f.read()
    except OSError as e:
        print(f"FAIL: cannot read actual file: {e}", file=sys.stderr)
        sys.exit(1)

    if len(expected) != len(actual):
        exp_w, exp_h = parse_header(expected, expected_path)
        act_w, act_h = parse_header(actual,   actual_path)
        print(f"FAIL: size mismatch. "
              f"Expected {len(expected)} bytes ({exp_w}×{exp_h}), "
              f"got {len(actual)} bytes ({act_w}×{act_h}).")
        sys.exit(1)

    if expected == actual:
        print("PASS")
        sys.exit(0)

    # Find and report the first differing byte.
    exp_w, _ = parse_header(expected, expected_path)
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            if i < HEADER_BYTES:
                print(f"FAIL: first difference at header byte {i}: "
                      f"expected 0x{e:02x}, got 0x{a:02x}.")
            else:
                offset = i - HEADER_BYTES
                row = offset // exp_w
                col = offset % exp_w
                print(f"FAIL: first difference at byte {i} "
                      f"(cell [{row}][{col}]): "
                      f"expected {e}, got {a}.")
            sys.exit(1)

    # Unreachable if the files differ, but guard anyway.
    print("FAIL: files differ (no differing byte found — unexpected)")
    sys.exit(1)


if __name__ == '__main__':
    main()
