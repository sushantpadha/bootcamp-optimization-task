# Monster Spawning Grid

You will implement a simulator for a cellular automaton called the **Monster Spawning Grid** — a multi-state, range-2 variant of Conway's Game of Life. Your goal is to compute generation 10,000 of a large grid as fast as possible, correctly.

This is a pure systems-performance challenge. The algorithm is fixed and given to you; the implementation is yours. You will be graded on wall-clock time to compute a correct result, and you will present your implementation to a panel of senior software engineers who will ask you to explain your design choices.

## The Rules

The grid is a square of up to 32,768 × 32,768 cells. Each cell is in one of four states:

- `EMPTY` (0)
- `EGG` (1)
- `JUVENILE` (2)
- `ADULT` (3)

The **neighbourhood** of a cell is the 24 cells in the 5×5 block centred on it (range-2 Moore neighbourhood), excluding the cell itself.

Let `A` be the number of `ADULT` cells in a given cell's neighbourhood. The state transition from generation `g` to `g+1` is:

| Current state | Transition                 |
|---------------|----------------------------|
| `EMPTY`       | becomes `EGG` if `3 ≤ A ≤ 5`, otherwise stays `EMPTY` |
| `EGG`         | becomes `JUVENILE` unconditionally |
| `JUVENILE`    | becomes `ADULT` unconditionally |
| `ADULT`       | becomes `EMPTY` if `A < 4` or `A > 9`, otherwise stays `ADULT` |

Only `ADULT` cells count toward the neighbour count `A`. `EGG` and `JUVENILE` cells do not.

**Boundaries are toroidal**: the grid wraps around in both dimensions. The neighbourhood of a cell at the edge includes cells from the opposite edge.

All cells transition simultaneously based on the state of the previous generation.

## What You Must Build

A C++ program that:

1. Reads an initial grid configuration from a binary input file.
2. Simulates exactly 10,000 generations.
3. Writes the final grid to a binary output file.
4. Prints the wall-clock time of the simulation (excluding I/O) to stdout in milliseconds.

## Input/Output Format

Both input and output files use the same binary format:

- 8 bytes: little-endian `uint64_t` width
- 8 bytes: little-endian `uint64_t` height (must equal width; only square grids are valid)
- width × height bytes of cell data, row-major. Each byte is either 0, 1, 2, or 3, representing the state of the cell

Width and height are always equal and always a power of two, up to 32,768.

You may use any internal representation you wish during simulation. The on-disk format is fixed.

## Reference Hardware

Competitition will be performed on an AWS `c8g.2xlarge` instance: 8 vCPUs on AWS Graviton4 (ARM Neoverse-V2, ARMv9.0-a), 16 GiB RAM, Ubuntu 24.04 LTS. Each participant will be provided with access to an identically-spec'd dev instance during the assignment window.

Relevant architectural notes:

- This is **ARM, not x86.** Use NEON (128-bit) and SVE2 (variable-width, typically 128-bit on Graviton4) for SIMD. `<immintrin.h>` and AVX/SSE intrinsics will not compile.
- Recommended compiler flag: `-mcpu=neoverse-v2`.
- 8 vCPUs, no SMT (one thread per core).
- ~64 KiB L1d, 2 MiB L2 per core, large shared L3.

## Constraints

You **must**:

- Write standalone C++23 (or C++26) code. `-std=c++23` or `-std=c++2c`.
- Compile with the provided build script. We will use `g++-14` 
- Use only the C++ standard library, ARM intrinsics headers (`<arm_neon.h>`, `<arm_sve.h>`) if you want explicit SIMD, and OS threading primitives via the standard library. No external libraries (no Boost, no TBB, no folly, no OpenMP pragmas).
- Run on a single machine, CPU only. No GPU offload, no distributed computing.
- Produce bit-identical output to the reference implementation on all test cases.

You **must not**:

- Use any algorithm that is asymptotically better than O(generations × cells). Specifically: no memoisation of grid states across generations, no pattern-recognition shortcuts, no HashLife or similar. This is a constant-factor engineering challenge — every correct implementation does the same amount of work; what differs is how efficiently the hardware executes it. The win must come from SIMD, threading, cache behaviour, and memory layout, not from a smarter algorithm. We will inspect your code and ask about this in the code review.
- Hardcode the answer or use precomputation that depends on the specific input. Your code must work correctly on any input matching the format, not just the provided test cases.

You **should**:

- Use `std::execution`, `std::thread`, atomics, SIMD, and any other standard C++ facilities aggressively.
- Care about cache, memory bandwidth, and false sharing.
- Write a benchmark harness that produces stable, repeatable measurements.

## AI Usage Policy

You are encouraged to use AI coding assistants (Claude, Copilot, Cursor, etc.) throughout this assignment. This is how modern performance engineering is done.

For maximum learning, use AI as an implementer, not a thinker. The most effective pattern is: you identify the strategy, you decide what to try, then you ask AI to help you implement it or track down a bug. "I want to tile the grid into 256×256 blocks and process each tile on a separate thread — help me structure the loop" will teach you far more than "rewrite this code to be faster and explain what you did." The first approach puts the engineering judgement in your hands; the second outsources it.

**You are responsible for understanding every line of your submission.** The code review will probe specific implementation choices — why your tile size is 256 and not 128, why you chose NEON over SVE2 at a particular site, why your adder network has the structure it does. If you cannot defend a design choice, it does not matter who or what wrote it.

## Deliverables

1. **Source code**, in a single repository, with a `build.sh` script that produces an executable named `spawn_sim`.
2. **A design document (max 3 pages)** covering:
   - Your chosen internal cell representation and why
   - Your parallelisation strategy
   - Your SIMD strategy (if any)
   - Your memory layout and tiling strategy
   - What you tried that didn't work, and why
   - What you would do with another week
3. **Your benchmark methodology**: how you measured, what you controlled for, what your variance looked like.

## Provided Materials

```
spawn-grid/
├── README.md                  # This file
├── reference/
│   ├── spawn_sim.cpp          # Reference implementation (correctness oracle)
│   └── build.sh               # Build script (copy and modify for your submission)
├── harness/
│   ├── run.sh                 # Test harness — use this to self-grade
│   └── verify.py              # Output verification script
├── test_grids/
│   ├── public_1_random_low_512.bin         # ~5% adults, random
│   ├── public_1_random_low_512.expected.bin
│   ├── public_1_random_low_2048.bin
│   ├── public_1_random_low_2048.expected.bin
│   ├── public_1_random_low_8192.bin
│   ├── public_1_random_low_8192.expected.bin
│   ├── public_1_random_low_32768.bin
│   ├── public_1_random_low_32768.expected.bin
│   ├── public_2_random_high_{512,2048,8192,32768}.bin   # ~50% adults, random
│   ├── public_2_random_high_{512,2048,8192,32768}.expected.bin
│   ├── public_3_structured_{512,2048,8192,32768}.bin    # Repeating block pattern
│   ├── public_3_structured_{512,2048,8192,32768}.expected.bin
│   ├── public_4_sparse_clusters_{512,2048,8192,32768}.bin   # Dense clusters in sparse field
│   ├── public_4_sparse_clusters_{512,2048,8192,32768}.expected.bin
│   ├── public_5_boundary_stress_{512,2048,8192,32768}.bin   # Patterns at toroidal boundaries
│   └── public_5_boundary_stress_{512,2048,8192,32768}.expected.bin
├── design_doc_template.md     # Template for your write-up
└── DEV_ENVIRONMENT.md         # How to access your c8g.2xlarge instance
```

## Quick Start

Build the reference implementation:

```bash
bash reference/build.sh
```

Run the reference on a test grid:

```bash
./spawn_sim test_grids/public_1_random_low_2048.bin /tmp/my_output.bin
```

Self-grade against expected output (10 runs):

```bash
bash harness/run.sh -n 10 ./spawn_sim \
    test_grids/public_1_random_low_2048.bin \
    test_grids/public_1_random_low_2048.expected.bin
```
