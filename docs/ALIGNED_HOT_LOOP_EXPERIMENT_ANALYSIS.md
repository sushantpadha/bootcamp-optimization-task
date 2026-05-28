# Aligned Hot-Loop Experiment Analysis

This note explains the aligned-hot-loop rewrite that produced a small but repeatable win over the `bsl_addsub` baseline.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop.cpp`

## 1. Starting point

The starting baseline was already strong:

- SVE2 hot loop
- direct range masks
- `bsl`-based add/sub rewrite
- forced inline on `compute_horizontal_partial_row`

The remaining question was narrower: can the main hot loop be reshaped so its frequent `ld1d` loads stop starting from odd word indices?

This mattered because the hot SIMD loop in `run_one_generation_hot_rows(...)` still started from:

```cpp
const int hot_word_begin = 1;
...
int word_index = hot_word_begin;
for (; word_index < simd_word_end; word_index += sve_words) {
    svuint64_t b0 = svld1_u64(pg, total_b0_ptr + word_index);
    ...
    svld1_u64(pg, adults + flat_word_index),
    svld1_u64(pg, juveniles + flat_word_index),
    svld1_u64(pg, eggs + flat_word_index),
```

On the remote machine, `svcntd() == 2`, so each `svld1_u64` is a 16-byte load.

## 2. What was wrong with the old access pattern?

The row bases are 64-byte aligned, but the hot SIMD loop started from `word_index = 1`.

That means the 16-byte loads started at byte offsets:

- `8`
- `24`
- `40`
- `56`

relative to the aligned row base.

So the loads were not 16-byte aligned, and every fourth one crossed a 64-byte cache-line boundary:

- offset `8`: stays within one line
- offset `24`: stays within one line
- offset `40`: stays within one line
- offset `56`: spans the end of one line and the start of the next

This is not a catastrophic miss problem, but it does create unnecessary load-side overhead in the hottest loop:

- periodic split-line loads
- extra load/store-unit work
- slightly worse memory-side efficiency than an aligned stream

## 3. Fix: peel the boundary words and align the SIMD body

The core idea was simple:

1. Handle the first hot word with the existing scalar logic.
2. Start the SIMD loop at `word_index = 2`, which is 16-byte aligned for `uint64_t` data when `svcntd() == 2`.
3. Leave the scalar tail to handle the last hot word and any leftover middle word.

That was applied to:

- the main rolling-update hot loop
- the final-row result-only loop

The new variant does this only when:

```cpp
const bool use_aligned_main_hot_loop =
    (sve_words == 2) && (hot_word_begin + 1 < hot_word_end);
```

So the change is explicitly targeted at the actual remote machine configuration.

## 4. What changed in the code?

The new version peels the first hot word in scalar form:

```cpp
int word_index = hot_word_begin;
if (use_aligned_main_hot_loop) {
    ...
    store_hot_result_from_total_with_center(...);
    subtract_horizontal_partial_from_total(...);
    add_horizontal_partial_to_total(...);
    ...
    ++word_index;
```

Then the SIMD middle starts from an even word index:

```cpp
const int aligned_simd_word_end =
    word_index + ((aligned_hot_word_end - word_index) / sve_words) * sve_words;
for (; word_index < aligned_simd_word_end; word_index += sve_words) {
    svuint64_t vb0 = svld1_u64(pg, total_b0_ptr + word_index);
    ...
    svld1_u64(pg, adults + aligned_flat_word_index),
    svld1_u64(pg, juveniles + aligned_flat_word_index),
    svld1_u64(pg, eggs + aligned_flat_word_index),
```

Important scope note:

- algorithm unchanged
- boolean logic unchanged
- horizontal partial generation unchanged
- only the hot-loop iteration structure changed

## 5. Why this helps

After the peel, the SIMD loop starts from `word_index = 2`, then advances by 2:

- `2`
- `4`
- `6`
- `8`

Those correspond to byte offsets:

- `16`
- `32`
- `48`
- `64`

So the main `ld1d` streams become 16-byte aligned, and the periodic split-line case goes away for those loads.

This improves:

- `b0..b4` loads
- `adults/juveniles/eggs` loads
- corresponding `st1d` stores

The AoS partial `ld3d` traffic is still structurally more awkward than plain SoA loads, but starting from an even word index also improves its address pattern relative to the previous odd-start loop.

## 6. Correctness result

Remote validation command:

```bash
python3 -u run_tests.py \
  reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop.cpp \
  512 \
  reference/build_spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop.sh
```

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

## 7. Benchmark result

Initial 500-iteration comparison on the large grid:

| Variant | Program time | Time per iteration |
| --- | ---: | ---: |
| `bsl_addsub` baseline | `7972.847 ms` | `15.945694 ms` |
| `aligned_hot_loop` | `7925.681 ms` | `15.851362 ms` |

Delta vs `bsl_addsub`:

- `-47.166 ms` over 500 iterations
- `-0.094332 ms` per iteration
- `0.59%` speedup

Longer 1000-iteration A/B checks were then used to see whether that was just noise.

Three remote runs gave:

- `bsl_addsub`: `15913.733 ms`, `15959.330 ms`, `15996.979 ms`
- `aligned_hot_loop`: `15879.559 ms`, `15859.974 ms`, `15878.493 ms`

Average over the 3 runs:

| Variant | Average program time | Time per iteration |
| --- | ---: | ---: |
| `bsl_addsub` baseline | `15956.681 ms` | `15.956681 ms` |
| `aligned_hot_loop` | `15872.675 ms` | `15.872675 ms` |

Average delta vs `bsl_addsub`:

- `-84.005 ms` over 1000 iterations
- `-0.084005 ms` per iteration
- `0.53%` speedup

Wall-clock averages showed the same direction:

- `bsl_addsub`: `21.550 s`
- `aligned_hot_loop`: `21.460 s`

## 8. Why this optimization worked

This was not a new algorithm and not a new boolean trick.

It worked because the existing hot-loop structure was leaving a small amount of memory-side efficiency on the table by starting the main SIMD body from an odd word index. Peeling the boundary word moved the bulk of the work onto aligned vector accesses without increasing algorithmic complexity.

The gain is modest, but it appears to be real:

- it won the initial 500-iteration comparison
- it won all 3 longer 1000-iteration runs
- the improvement survived a reversed run order

## Takeaway

This was a small structural cleanup of the hottest loop:

1. Identify a real but narrow alignment inefficiency in the main SIMD body.
2. Fix it with a cheap scalar peel instead of a larger rewrite.
3. Verify correctness remotely.
4. Re-run long benchmarks to check whether the effect survives noise.

In short: peeling the first hot word and running the SIMD middle on aligned addresses produced a repeatable `~0.5%` speedup over the `bsl_addsub` baseline.
