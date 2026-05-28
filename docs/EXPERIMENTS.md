# Experiments — What Was Tried, What Worked, and Why

This document covers every significant algorithmic and implementation direction explored during optimisation. Each section describes the idea, the evidence, the result, and the reason it succeeded or failed.

---

## Part 1 — Algorithmic Explorations

These experiments changed the high-level structure of the computation before any SIMD or ISA work was done.

---

### Two-pass tiled approach (baseline)

**Idea:** Separate the work into two sequential passes over the grid, each with its own loop structure. Pass 1 builds an intermediate horizontal-sum buffer for all rows. Pass 2 reads that buffer, performs vertical accumulation, and applies the rules.

**Result:** Functional but slow. The two passes double the memory bandwidth consumption — the intermediate buffer is written in full during pass 1 and read in full during pass 2. At 32K × 32K, this puts enormous pressure on the memory hierarchy. This became the starting-point baseline against which all later variants were measured.

---

### Row-prefix-sum variant

**Idea:** Compute the neighbour count for each row using prefix sums: scan left-to-right accumulating a running count, then look up the difference between two prefix positions to get the horizontal 5-wide sum. Run rows in parallel across threads.

**Result:** Faster than the two-pass baseline due to better cache utilisation and lower redundant work. However, the prefix-scan approach still processes each output cell with a scalar comparison chain, limiting throughput. It was superseded by the rolling-window approach.

---

### Sliding window with rolling vertical accumulator

**Idea:** Observe that consecutive output rows share four of their five source rows. Maintain a rolling 5-row horizontal-partial accumulator. Advancing to the next output row costs only one subtract (remove top row) and one add (add bottom row), rather than summing all five rows from scratch.

**Result:** This was the fundamental algorithmic breakthrough. It reduced the vertical accumulation cost from O(5) to O(1) per output row, roughly halving the work over naive approaches. All subsequent variants build on this structure.

**Why it works:** The 5-row window has 80% overlap between consecutive output rows. The rolling update exploits this overlap maximally. Without it, every row would independently sum five full horizontal partials regardless of how much reuse existed.

---

### Fused horizontal sum (no intermediate buffer)

**Idea:** Rather than computing all horizontal partials for a row into an intermediate buffer and then reading them back, compute the partial for one word, immediately add it to the vertical total, and advance. Eliminates the intermediate write-then-read round-trip.

**Result:** A modest improvement at the time it was introduced. It reduced memory traffic to the intermediate partial buffer. The same principle was later applied more aggressively in the "fused incoming-partial" optimisation (see Part 2).

---

### Two-bit packed cell storage

**Idea:** Each cell naturally encodes two bits of state. Store four cells per byte instead of one cell per byte (the default). This compresses the working set by 4×.

**Result:** Slower at small and medium grid sizes due to pack/unpack overhead. The main benefit only appears at the very largest grid sizes (32K × 32K) where cache pressure is the bottleneck. Even then, the unpack overhead in the hot loop neutralised the cache benefit. Not adopted in the final approach.

**Why it fails:** The computational overhead of unpacking 2-bit fields into workable bit-planes on every generation exceeds the savings from the smaller working set, especially when the algorithm is already using bit-sliced representation internally.

---

### LUT-based rule application

**Idea:** Precompute a 128-entry lookup table indexed by `(cell_state << 5) | neighbour_count`. Apply rules by table lookup rather than explicit comparisons.

**Result:** Competitive with the scalar comparison approach. However, table lookups become a bottleneck when the inner loop is vectorised, because SIMD gather instructions have high latency and cannot be pipelined efficiently. Not competitive against the bitwise-predicate approach used in the final design.

**Why it doesn't scale:** LUTs work well for scalar code but the gather overhead dominates at SIMD width. The bitwise Karnaugh-minimised predicates compute the same result with 6–8 bitwise instructions that all have 1-cycle latency and can execute in parallel.

---

### Sparse grid fast-path

**Idea:** If most cells are `EMPTY` and have no adult neighbours, skip them entirely rather than computing a zero neighbour count that produces no state change.

**Result:** Depends entirely on grid density. For low-density grids, skipping empty regions can be a large win. For high-density grids (the hardest test case), the branch overhead and unpredictable scan pattern eliminates any benefit. Since the benchmark uses both high and low density cases, a density-adaptive strategy would be required, adding significant complexity. Not adopted.

---

### Column-major tiling with in-register vertical accumulator

**Idea:** Reorganise the loop to iterate over columns in the outer loop and rows in the inner loop. Keep the entire 5-wide vertical accumulator in SVE registers for a full column, eliminating the rolling-total array from memory entirely.

**Result:** Significantly slower — 203 440 ms vs 138 147 ms for the row-major equivalent at 32K × 32K. A major regression.

**Why it fails:** Rows are stored contiguously in memory. A column-major traversal accesses one word per row, strided across `words_per_row * 8` bytes between each access. At 32K, each row is 4 KB, so column access strides across 4 KB gaps. This destroys the hardware prefetcher's ability to stream cache lines and generates a flood of cache misses that completely overwhelms the register-pressure savings. Row-major streaming is not negotiable on this hardware.

---

## Part 2 — Memory Layout Experiments

These experiments changed how data is arranged in memory without altering the algorithm.

---

### Flat slab allocation

**Idea:** Use a single contiguous memory slab for all bit-planes rather than separate `std::vector` allocations. This ensures the arrays are physically adjacent in virtual memory and reduces the overhead of separate heap allocations.

**Result:** A consistent improvement. Eliminated allocation jitter and slightly improved cache behaviour for the multi-plane access pattern. Adopted in the main line.

---

### AoS horizontal partials with structured loads (`svld3/svst3`)

**Idea:** Store the three bit-planes of a horizontal partial word as a struct `{b0, b1, b2}`, one struct per word. Use SVE structured loads (`svld3_u64`) to gather all three planes into separate SVE registers in one instruction.

**Result:** Slower than plain separate loads. On the target machine, `svld3_u64` gathers from three non-contiguous 8-byte locations separated by 8 bytes each. The hardware must perform three separate cache-line lookups and then assemble the result. The "convenience" of one instruction comes at the cost of three separate memory transactions.

**Why it fails:** Structured loads `ld3/st3` have higher latency than plain `ld1/st1` because they involve a gather from non-contiguous memory. On Neoverse-V2, the cost of `ld3d` is effectively three cache-line lookups, not one. Sequential `ld1d` on aligned plane buffers is faster.

---

### Pair2 layout — storing 2-word vector tiles directly

**Idea:** Since the target has 2-lane SVE vectors (`svcntd() == 2`), the natural work unit is *two consecutive words*. Instead of storing one-word-at-a-time structs, store the 2-word pair together as three 16-byte arrays packed in a struct aligned to 16 bytes:

```
{b0[2], b1[2], b2[2]}  — 48 bytes, 16-byte aligned
```

The hot loop then reads and writes each plane with a single aligned `ld1d/st1d` rather than a structured `ld3d/st3d`.

**Result:** +5.2% speedup on the 500-generation 32K benchmark.

**Why it works:** The layout is shaped exactly to match the machine's vector tile. Each `svld1_u64` loads one full `b_n` plane for the pair in one cache-line-aligned transaction. There is no gather, no strided access, and no unnecessary data interleaving. Three sequential aligned loads replace one structured non-contiguous gather.

---

### Rolling-total plane separation (SoA)

The five rolling-total planes (`b0..b4`) are stored as separate arrays rather than interleaved. This means a single word's 5-bit total is spread across five separate memory locations.

**Kept:** Even though this seems wasteful per-word, it enables the hot SVE body to load all five planes for a 2-word tile using five sequential `ld1d` instructions with consecutive addresses, which streams well. Interleaving would require gather instructions.

---

## Part 3 — SIMD and ISA-Level Optimisations

These experiments kept the algorithm and layout fixed and focused on expressing the same computation in forms that produced better machine code.

---

### SVE2 vectorisation

**Idea:** Vectorise the hot loop over the rolling total with ARM SVE2 intrinsics, processing two 64-bit words (128 bits, two lanes) per iteration.

**Result:** Large speedup over scalar. Every arithmetic operation on the rolling total — add, subtract, emit — now processes 128 cells per instruction rather than 64. The main loop body nearly halves in iteration count.

---

### Direct range masks (Karnaugh-minimised predicates)

**Idea:** The original threshold logic computed four separate predicates (`ge3`, `ge4`, `ge6`, `ge10`) and combined them with AND/OR. Replace with K-map minimised expressions operating directly on the 5-bit `{b4..b0}` representation:

```
adult_5_to_10 = ~b4 & ((~b3 & b2 & (b1|b0)) | (b3 & ~b2 & ~(b1&b0)))
egg_3_to_5    = ~b4 & ~b3 & ((~b2 & b1&b0) | (b2 & ~b1))
```

These are the ranges pre-shifted by 1 to account for the centre being included in the total (see CURRENT_APPROACH.md).

**Result:** Reduced the emit instruction count and laid the foundation for the BSL rewrites that followed.

---

### BSL-shaped carry and borrow logic (+4.1%)

**Idea:** The full-adder carry `(a&b) | (a&c) | (b&c)` can be rewritten as:
```
carry = c ? (a | b) : (a & b)  →  bsl(a | b, a & b, c)
```

Similarly, the borrow propagation:
```
borrow_out = borrow_in ? (~b | p) : (p & ~b)  →  bsl(~b | p, p & ~b, borrow_in)
```

Both map to a single `bsl` instruction per stage.

**First attempt failed:** The first implementation compiled cleanly but produced wrong output on all five test cases. The `svbsl_u64(Zdn, Zm, Zk)` intrinsic computes `(Zdn & Zk) | (Zm & ~Zk)` — the first argument is both the destination register and one data operand, not the selector mask. After isolating the semantics with a small truth-table test and correcting the argument order, all tests passed.

**Result:** +4.1% speedup. GCC 14 was not independently discovering the `bsl` lowering from the original `and/or` chain. Reshaping the source to match `bsl`'s semantic signature gave the compiler a direct lowering path.

**Why it works:** `bsl` encodes a ternary bit-select in one instruction with one-cycle throughput and latency. Replacing three `and/or` operations with one `bsl` reduces the critical path depth in both the adder and the subtractor, which execute thousands of times per generation.

---

### Aligned hot-loop peel (+0.5%)

**Idea:** The hot SVE loop started from word index 1 (odd). Since each row's base address is 64-byte aligned but the loop starts at word 1 (offset 8 bytes), the 16-byte SVE loads start at misaligned offsets: 8, 24, 40, 56 bytes from the cache-line boundary. Every fourth load crosses a 64-byte cache-line boundary.

Fix: process word index 1 in scalar, then start the SVE loop from word index 2 (even, 16-byte aligned from the row base). Each hot `ld1d` now starts at offsets 16, 32, 48 — all within single cache lines.

**Result:** +0.5% speedup, consistent across multiple run configurations.

**Why it works:** Split-line loads have higher latency on out-of-order processors because the load unit must wait for both cache lines to be available before forwarding data. Eliminating the periodic split-line event in the main streaming path reduces stall cycles.

---

### Fused incoming-partial production (+7.2%)

**Idea:** The rolling-update loop had a producer/consumer separation: first build the entire incoming row's partial into a scratch buffer, then in the main word loop reload it for the add operation:

```
build incoming row → [scratch buffer] → reload from scratch buffer → add to total
```

The fused version computes each incoming partial word *on demand* directly from the adult bit-plane, uses it immediately in the add, and stores it once into the ring slot just vacated by the outgoing row. The scratch buffer is eliminated.

```
compute partial on the fly → add to total → store into ring slot
```

**Result:** +7.2% — the largest single gain in the ISA-tuning phase. The benchmark improvement was consistent across three independent alternating runs.

**Prior attempt failed:** An earlier "vertical microkernel" approach had tried to reduce memory traffic by changing the traversal order to accumulate vertically in inner loops. It reduced accumulator loads/stores but destroyed the row-major streaming pattern that the hardware prefetcher relies on, and regressed significantly. The fused-incoming approach is more conservative — it preserves the exact row-major access order and only eliminates the unnecessary scratch buffer round-trip within that order.

**Why it works:** The scratch buffer forced each incoming partial word through two extra memory transactions (write + read) before it was used. Fusing eliminates those transactions without touching any other aspect of the algorithm. The L1 data cache pressure drops, and the load-store unit can retire work faster.

---

### Exact two-row blocking (+0.5%)

**Idea:** The rolling-update loop loaded and stored the 5-bit rolling total once per output row. Load it, use it for one row, store it back. The exact-2-row variant loads the total once, processes two consecutive output rows, then stores — halving load/store traffic for the accumulator.

**Result:** +0.5%. Real but modest.

**Why small:** The optimisation reduces memory traffic from the rolling-total arrays, but it also increases the number of live values in registers simultaneously (all 5 planes must remain live for two rows instead of being freed and reloaded). On Neoverse-V2, the register file is large enough to absorb this, but the extended live ranges slightly increase register pressure and scheduling constraints, partially cancelling the traffic savings.

---

### Pure-SVE emit path (+2.9%)

**Idea:** The emit function (which converts the rolling total into next-generation bits) had been implemented using NEON intrinsics even though the rest of the hot loop used SVE. This created a mixed-ISA bridge where SVE registers were spilled to stack, reloaded as NEON registers, processed with NEON arithmetic, and the results stored back.

The pure-SVE rewrite keeps the count data in SVE registers throughout, implementing the rule predicates directly with `svbic`, `svorr`, `svbsl`.

Additionally, the Boolean expressions were restructured into BSL-friendly ternary selects:
```
adult_5_to_10 = bic(svbsl(b1|b0, b2, ~b3) | svbsl(~(b1&b0), b3, b2), b4)
```

**Result:** +2.9% over the mixed-ISA baseline. The pure-SVE path removes the spill/reload round-trip and lets the compiler see the entire hot loop as a single register-resident computation.

**Why it works:** Register spills to stack are expensive — they occupy load-store bandwidth and introduce pipeline stalls. Eliminating the SVE→NEON bridge removes several stores and loads per tile from the hottest path.

---

### Pointer-stepped loop body (+1.6%)

**Idea:** The aligned SVE inner loop carried an integer `word_index` as its induction variable and computed all array addresses by adding `word_index` to multiple base pointers each iteration:

```
adults + word_index
juveniles + word_index
rolling_total_b0 + word_index
...
```

The pointer-stepped version maintains five explicit pointers and advances each by the SVE stride at the bottom of every iteration, removing the per-iteration multiply-or-shift-add in address generation.

Additionally, moving the aligned vector body into a dedicated `__attribute__((noinline))` function reduces the amount of code GCC must handle at once, enabling better instruction scheduling for that specific loop.

**Result:** +1.6%.

**Why it works:** Index-based addressing requires an integer multiply or shift-add to form each address, adding integer ALU operations to the critical path alongside the SVE arithmetic. Pointer arithmetic amortises that cost — each pointer is incremented once by a constant stride. Extracting into a separate function also gives the register allocator a cleaner scope with fewer live values competing for registers.

---

### `svnot` and `eor3` cleanup (+5.0%)

**Idea:** Two source patterns in the hot helpers were more indirect than necessary:

1. Bitwise NOT was expressed as `x ^ svdup_u64(~0ULL)`. This materialises an all-ones constant, consuming a register and an ALU slot.
2. Three-input XOR was expressed as `(a ^ b) ^ c` — a 2-instruction chain with a data dependency between the two XORs.

Replacing with direct SVE2 intrinsics:
- `svnot_x(pg, x)` — a dedicated NOT instruction with no constant
- `sveor3(a, b, c)` — a true 3-input XOR in one instruction, available in SVE2

The hot paths where these appear: the `carry_save_add` sum computation and the rolling-total subtract path. Both execute on every partial word every generation.

**Result:** +5.0% on 1000-generation benchmark. Assembly confirmed the change: the baseline had multiple `mov z?.b, #-1` (all-ones materialisation) sites in the hot code; the candidate had none. The `eor3` pattern appeared in the expected locations.

**Why it works:** Removing the all-ones register materialisation frees a register slot and eliminates one instruction from the hot XOR path. Fusing the two-XOR chain into `eor3` shortens the critical path depth by one cycle. Both effects compound across the tight CSA and subtract loops that execute billions of times per benchmark run.

---

### SLI/SRI for neighbour-column shifts (+1.1%)

**Idea:** Building the four horizontal neighbour columns requires four shift-and-OR operations:
```
left_2  = (curr << 2) | (prev >> 62)   // SHL + ORR
left_1  = (curr << 1) | (prev >> 63)   // SHL + ORR
right_1 = (curr >> 1) | (next << 63)   // SHR + ORR
right_2 = (curr >> 2) | (next << 62)   // SHR + ORR
```

The SVE2 (and NEON) shift-and-insert instructions fold the OR into the shift:
- `SRI Vd, Vm, #n`: shifts `Vm` right by `n`, inserts result into the bottom `n` bits of `Vd` (keeping the top `64-n` bits of `Vd`). Since `curr<<k` already has zero low bits, this is equivalent to the ORR but in one instruction.
- `SLI Vd, Vm, #n`: analogously for left shift.

```
left_2  = SRI(curr << 2, prev, 62)   // SHL + SRI  (no ORR)
right_1 = SLI(curr >> 1, next, 63)   // SHR + SLI  (no ORR)
```

**Result:** +1.1%. Assembly confirmed `sli/sri` instructions in the hot partial-generation path; the baseline had none.

**Why it works:** Four ORR instructions are removed from the horizontal-partial builder, which runs once per row per word. On a 32K grid with 512 words per row and 32K rows per generation, this is 10K calls to the partial builder per generation, each saving four instructions.

---

### Dual-partial overlap (+0.9%)

**Idea:** Reorganise the inner loop body so the horizontal partial computation for the next word position overlaps with the subtract/add/emit computation for the current word position. This gives the out-of-order execution unit more independent work to schedule in parallel.

**Result:** +0.9%. The independent streams of partial computation and total update execute simultaneously on separate execution ports.

---

### `bic` instruction across all NOT patterns

**Idea:** Systematically replace every `~b & x` pattern with `BIC(x, b)` — the ARM NEON/SVE `a & ~b` instruction. Apply this to:
- Subtract borrow: `borrow = partial & ~total` → `vbicq(partial, total)`
- Predicate logic: `~b4 & result` → `vbicq(result, b4)`
- Output: `~occupied & egg_3_to_5` → `vbicq(egg_3_to_5, occupied)`

This eliminates all explicit NOT temporaries (`not_b4`, `not_b3`, `not_b2`, `not_b1`) from the hot path.

**Result:** Consistent improvement. Combined with BSL rewriting this produces the final predicate form documented in CURRENT_APPROACH.md. The specific savings depend on how many NOT temporaries the compiler had already eliminated — the explicit rewrite guarantees zero `vmvn/svnot` instructions in the pattern positions targeted.

---

### `vbslq` for next-adult output word

**Idea:** The next-adult computation is:
```
next_adult = (adult_word & adult_5_to_10) | juvenile_word
```

Since `adult_word & juvenile_word = 0` (mutually exclusive planes), `juvenile & ~adult = juvenile` exactly. Therefore:
```
vbslq(adult_word, adult_5_to_10, juvenile_word)
  = (adult_5_to_10 & adult_word) | (juvenile_word & ~adult_word)
  = (adult_word & adult_5_to_10) | juvenile_word   ✓
```

One BSL instruction replaces one AND and one OR.

**Result:** One instruction removed from the emit path per tile.

---

### Non-temporal stores

**Idea:** Write the output planes (`next_adults`, `next_eggs`) with non-temporal store hints (`svstnt1_u64`), bypassing the cache on the write path. This reserves L1/L2 capacity for the rolling-total planes and the adult bit-plane reads, which are reused repeatedly.

**Result:** A small but consistent improvement on large grids where the output planes compete with active working-set data for cache space. On small grids, the effect is negligible.

---

### Explicit prefetch directives

**Idea:** Insert software prefetch instructions a fixed number of iterations ahead to pre-warm cache lines for the adult bit-plane and partial-ring reads.

**Result:** Neutral to slightly negative on Neoverse-V2. The hardware prefetcher on this core handles sequential access patterns well. Adding software prefetches introduced instruction overhead and sometimes interfered with the hardware prefetcher's stride-detection logic, producing slightly worse results. Not adopted.

**Why it fails here:** Software prefetch helps most when the hardware prefetcher cannot detect the access pattern (e.g., gather accesses, irregular strides). The hot loop accesses all arrays with stride-1, which the hardware prefetcher handles optimally. Explicit prefetch instructions add overhead without improving the cache hit rate.

---

### Thread-count experiments (4, 6, 8 threads)

**Idea:** Vary the number of worker threads from 4 to 8 to find the optimal for the target machine.

**Result:** 8 threads was consistently fastest on the Graviton4, which has 8 cores available. Below 8, the computation is limited by CPU throughput. Above 8, threads would share cores and add synchronisation overhead. Thread pinning (each thread fixed to one core) was important — without it, OS migrations between generations added jitter.

---

## Summary: Why Things Worked or Didn't

### What consistently worked

1. **Row-major streaming.** Any access pattern that allows the hardware prefetcher to stream 64-byte cache lines was fast. Any deviation from this (column-major, gather, structured loads) was slow. The prefetcher on Neoverse-V2 is powerful for stride-1 streams.

2. **ISA matching.** Expressing the same Boolean logic in forms that GCC could directly lower to specific instructions (`bsl`, `eor3`, `sri`, `sli`, `bic`) produced real gains. GCC did not discover these forms automatically from natural-looking `and/or` source code.

3. **Reducing register materialisations.** Eliminating all-ones constants and NOT temporaries freed register slots and instruction slots in a register-pressure-sensitive loop.

4. **Eliminating round-trips.** Any time data was computed, stored to memory, then immediately reloaded (the scratch-buffer pattern, the mixed-ISA spill/reload), removing the round-trip was a win.

5. **Layout matching hardware granularity.** Storing data in units that match the actual vector width (Pair2 layout) produced better code than structures sized for abstract convenience.

### What consistently failed

1. **Changing access order.** Column-major tiling, vertical microkernel — any reorganisation that made row access strided produced major regressions from cache misses.

2. **Software prefetch.** The hardware prefetcher outperforms software hints on sequential access patterns. Adding hints added overhead and sometimes hurt.

3. **AoS structured loads.** `ld3/st3` style loads that gather from interleaved struct fields are slower than three sequential aligned `ld1` loads on separate plane arrays.

4. **Packed cell formats.** The overhead of unpacking 2-bit or 4-bit packed cells on every generation exceeded the cache savings at the grid sizes that mattered.

5. **Wrong intrinsic semantics.** The BSL argument order mistake showed that assuming NEON-style `(mask, true, false)` semantics for SVE's `svbsl_u64(Zdn, Zm, Zk)` is wrong. BSL is a destructive instruction where the first operand is simultaneously the destination and one data input. Always verify with an isolated truth-table test before benchmarking.

### The compounding pattern

No single optimisation was transformative after the initial algorithmic work (rolling window, bitsliced representation, vectorisation). The remaining gains were small individually:

| Optimisation | Gain |
|---|---|
| BSL add/sub | +4.1% |
| Pair2 layout | +5.2% |
| Fused incoming | +7.2% |
| SVnot + eor3 | +5.0% |
| Pure-SVE emit | +2.9% |
| Ptrstep | +1.6% |
| SLI/SRI | +1.1% |
| Exact 2-row blocking | +0.5% |
| Aligned loop peel | +0.5% |
| Dual partial | +0.9% |

Applied sequentially, these compound into a total reduction of roughly 28% over the post-vectorisation baseline. The lesson: late-stage optimisation is about accumulating many small wins, each requiring specific hardware knowledge and careful measurement.
