# Design Document — Monster Spawning Grid

**Name:** ___________________
**Date:** ___________________
**Final median time (10 runs, public\_1):** ___________________ ms
**Reference median time (10 runs, public\_1):** ___________________ ms
**Speedup:** ___________________×

---

## 1. Cell Representation

The grid is stored as three separate bit-planes: `adults`, `juveniles`, `eggs`. Each is a flat row-major array of `uint64_t` words with 64 cells per word.

**Why three planes instead of a packed 2-bit encoding?**
Neighbour counting only ever reads the adult plane. Keeping it separate means the stencil reads exactly `N²/8` bytes with no masking or decoding. A packed 2-bit encoding would require extracting the adult bit from every word on every stencil read.

**Why not one byte per cell?**
At `N = 32 768`, a byte-per-cell grid is 1 GiB per buffer. Three bit-planes at 128 MiB each, with four planes active at once, totals 512 MiB — a 4× smaller working set and proportionally less memory bandwidth per generation.

**In-place adult write (`inplace_adults`).**
The next-adult mask is simply `(current_adults & survive_mask) | current_juveniles`: surviving adults plus every juvenile, which matures unconditionally. The kernel writes this directly into `current_juveniles` rather than allocating a separate scratch plane. Generation rotation then costs three pointer swaps:

```
adults <- juveniles <- eggs <- next_eggs
```

No data is copied; this also eliminates one full write pass per generation.

**Self-inclusive counts.**
The horizontal partial at column `c` sums the adult bits at `{c-2, c-1, c, c+1, c+2}`, including the center cell. The full vertical 5-row sum therefore counts the entire 5x5 box including self. The transition thresholds shift by +1 to compensate: ADULT survives if `V in [5,10]` instead of `A in [4,9]`, and EGG spawns if `V in [3,5]` (unchanged, since empty cells contribute 0). This removes a center-subtraction from the inner loop.

---

## 2. Parallelisation Strategy

The simulation runs eight `std::thread` workers, one per physical core, synchronised by two `std::barrier` instances per generation (one to start, one to finish). Each worker owns a contiguous stripe of rows for the entire run, computed once by `compute_row_range`.

The stripe assignment is static. Interior rows (`[2, N-3]`) only read from source planes that no other worker writes during the same generation, so no locks or atomics are needed inside a generation. Each worker also maintains private scratch (the row-partial ring and rolling accumulator) that would be expensive to reinitialise if the partition changed each generation.

`pthread_setaffinity_np` pins each worker to a fixed CPU. Without pinning, the OS can migrate a thread mid-generation, evicting its L2-resident scratch onto a cold core.

**Why not `std::execution::par`?**
Three reasons: (1) no thread affinity control, so the backing pool can migrate workers freely, defeating cache warmth across generations; (2) no persistent per-worker scratch state across calls, meaning the five-row ring and rolling accumulator would need to be rebuilt every generation; (3) no guarantee on partition shape, so the contiguous row-stripe alignment required by the 5-row stencil halo cannot be enforced.

---

## 3. SIMD Strategy

The implementation uses SVE2. On Neoverse-V2, SVE2 and NEON have the same 128-bit register width, so width is not the argument. The hot kernel is almost entirely bitwise operations: CSA trees, a borrow-propagation subtractor, and a boolean transition network. SVE2 provides instructions that map onto these operations more directly than NEON. `SLI`/`SRI` fuse a shift and a bit-insert into one instruction; `BSL` expresses a conditional AND-OR in one instruction; `SVNOT` is cleanly predicated in-lane. Empirically, the SVE2 path ran measurably faster than an equivalent NEON implementation.

### Carry-Save Adder tree (`csa_tree`)

Five shifted adult masks are reduced to a 3-plane bit-sliced integer in `[0,5]` via two CSA stages:

```
Stage 1:  left_2 + left_1 + curr    ->  (sum_1, carry_1)
Stage 2:  right_1 + right_2 + sum_1 ->  (sum_2, carry_2)

b0 = sum_2
b1 = carry_1 XOR carry_2
b2 = carry_1 AND carry_2
```

A CSA takes three equal-weight inputs and produces a sum bit and a carry bit without propagating carry across bit lanes. Two stages reduce five inputs to a 3-bit result with no carry chains, no byte unpacking, and no scalar loop, processing all 128 cells in the tile simultaneously.

### Cross-word bit shifts (`sli_sri`)

The five neighbour views require bits from adjacent words when shifting near word boundaries. `SLI` (shift-left-and-insert) handles this in one instruction:

```
left_1 = SLI(prev >> 63, curr, #1)
       = (curr << 1) with bit 63 of prev inserted at bit 0
```

This works cleanly because the tile at even word index `2k` loads `prev = [word 2k-1, word 2k]`, so both cross-word carry bits for the two tile lanes are already in a single register.

### Tile layout (`pair2`, `aos`)

```cpp
struct alignas(16) HorizontalPartialPair2 {
    uint64_t b0[2], b1[2], b2[2];
};
```

Two adjacent words are packed into one struct so that each 2-element array is exactly one SVE register load or store. The array-of-structs layout keeps all three bit-planes of a tile contiguous in memory, so loading a tile is three sequential cache-line-aligned loads that the prefetcher handles naturally.

### Transition network (`bsl`, `sve_emit`)

The survival condition `(~b3 & b2 & (b1|b0)) | (b3 & ~b2 & ~(b1&b0))` is rewritten as a single `BSL` (bit-select):

```
bsl(~b2 & ~(b1&b0),   b2 & (b1|b0),   b3)
     when b3=1          when b3=0
```

`SVNOT` is used inline throughout the transition logic rather than precomputing complements into temporaries, avoiding register spills in a register-heavy section.

### Branchless subtraction (`branchless_subtract`)

Sliding the vertical window subtracts the outgoing partial from the 5-plane accumulator using a ripple-borrow network applied plane-by-plane:

```
borrow_out = (~count & (partial | borrow_in)) | (partial & borrow_in)
new_count  = count XOR partial XOR borrow_in
```

No branches. `svbic` and `eor3` keep each borrow stage at two instructions.

### Dual-stream partial computation (`dualpartial`, `fused_incoming`)

When the two-row pipeline needs incoming partials for rows `r+3` and `r+4`, `compute_two_partials_from_word_ptrs` interleaves both CSA trees. This lets the CPU overlap the first tree's shifts and carries with the second tree's loads, hiding load latency.

### Aligned hot loop and pointer stepping (`aligned_hot_loop`, `ptrstep`)

Word 1 of each row is handled by a scalar prologue so the SVE loop can start at even word index 2, which is required for the `SLI`/`SRI` cross-lane carry to work correctly. The innermost function takes raw pointers and advances them by 2 words per iteration, removing multiply-add address arithmetic from the critical path.

### Exact two-row pipeline (`exact2row`, `rowstep`)

The main loop processes two output rows per iteration. The five accumulator planes are loaded once, two emit steps and two slide steps are applied, then stored once. This halves the number of accumulator load/store operations per output row compared to a single-row loop.

---

## 4. Memory Layout and Tiling

### Working set per worker

At `N = 32 768`, the hot per-worker working set is:

| Structure | Size |
|---|---|
| 5-row horizontal partial ring (`rowcache`) | `5 x (N/64) x 3 x 8` B = **120 KiB** |
| Rolling accumulator, 5 planes x 1 row (`rolling`) | `5 x (N/64) x 8` B = **20 KiB** |
| **Total** | **~140 KiB** |

This fits comfortably in the 512 KiB L2 per Neoverse-V2 core. Source-plane reads are sequential and hardware-prefetched.

### Rolling accumulator and ring (`rolling`, `rowcache`)

Rather than re-summing five partial rows per output row, the kernel maintains a running total updated by one add and one subtract per row:

```
total[r] = total[r-1] - H[r-3] + H[r+2]
```

The five partial rows in the current window live in a circular ring of five slots indexed by `row_index % 5`. When the window slides by one row, the oldest slot is overwritten with the new incoming row.

### Timing across grid sizes

![Simulation time vs. grid size (10 000 generations)](images/time_10000.png)

*Figure 1. Wall-clock time for 10 000 generations, N in {64, ..., 32 768}. Median of 10 runs.*

### PMU counters (N = 1 000, 1 000 generations)

| IPC | Backend stalls | L1 miss rate | dTLB miss rate (before) | dTLB miss rate (after) |
|---|---|---|---|---|
| ![](test_grids/plots/ipc_1000.png) | ![](test_grids/plots/backend_stalls_1000.png) | ![](test_grids/plots/cache_misses_1000.png) | ![](test_grids/plots/dtlb_rate_1000_wo_thp.png) | ![](test_grids/plots/dtlb_rate_1000.png) |

---

## 5. What Didn't Work

### 5.1 Column-wise tiling in 128-word segments

We tiled the inner word loop into 128-word segments. Each word is a `uint64_t` processed in pairs via 128-bit SVE registers, so 128 words = 8192 cells per tile slice. The target hot set per tile was estimated as:

- 5 stencil rows x 2 KiB = 10 KiB
- 5 rolling count bit-planes x 2 KiB = 10 KiB
- Horizontal partial ring cache, incoming/outgoing partials, alignment scratch: ~12-24 KiB

Estimated total: ~32-48 KiB, targeting the 64 KiB L1-D per core.

It made things worse. We measured backend stalls using PMU counters and found that only 15-20% of backend stalls were memory-caused, meaning the kernel was already compute-bound. L1 miss data confirmed this: only ~10% of L1 misses were L2 refills, so the rolling accumulator was already largely L2-resident. There was no meaningful cache pressure to relieve. The tiling added per-tile loop overhead (branch, pointer reset, ring bookkeeping) without any offsetting gain.

### 5.2

*(reserved)*

---

## 6. What You Would Do With Another Week

*(left blank)*

---

## 7. Benchmark Methodology

We initially used a lightweight correctness/performance harness (`run_tests.py`) during early iteration, then switched to a dedicated benchmarking script (`analysis.sh`) for automated compilation, correctness testing, profiling, and repeated evaluation under fixed conditions.

The script builds optimized and debug/profile binaries, verifies correctness on all 512×512 grids for 10K generations, and profiles primarily on 32768×32768 grids.

We used `perf stat` in two main configurations:

* `cycles`, `instructions`, `stalled-cycles-backend`, `stall_backend_mem` to measure IPC, backend stalls, and dependency pressure,
* `cycles`, `instructions`, `L1-dcache-load-misses`, `l2d_cache_refill_rd`, `l2d_cache_wr` to analyze cache and memory behavior.

Scaling runs additionally monitored `cache-misses`, `dTLB-loads`, `dTLB-load-misses`, and backend stalls across multiple grid sizes.

We used `perf record`, `perf report`, and `perf annotate` in two modes:

* `perf record -e cache-misses` to study cache/memory bottlenecks,
* `perf record -e cycles` to analyze execution hotspots, dependency chains, and load-use hazard stalls.

Each scaling configuration was run 3 times and the median result was used.

We controlled for fixed CPU frequency, fixed CPU affinity (`taskset`), ASLR disabled, warm page cache after initial runs, and transparent huge page configuration inspection. Optimizations were evaluated incrementally by changing one component at a time and comparing perf counters and hotspot distributions across runs.
