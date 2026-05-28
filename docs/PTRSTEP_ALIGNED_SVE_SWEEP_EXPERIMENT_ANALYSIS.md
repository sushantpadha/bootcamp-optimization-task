# Ptr-Stepped Aligned SVE Sweep Experiment Analysis

This note explains the pointer-stepped aligned-SVE rewrite that improved the current pure-SVE `sve_emit_bsl` baseline.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep.cpp`

## 1. Why this comparison matters

This is an apple-to-apple comparison against the current best pure-SVE version.

The new candidate keeps:

- the same algorithm
- the same `pair2` partial layout
- the same exact-two-row schedule
- the same fused-incoming update
- the same in-place-next-adult arrangement
- the same pure-SVE emit `bsl` rewrite

So the only intended optimization here is narrower:

- keep the aligned pair2 SVE body doing the same work
- present that work to GCC in a form with less hot-loop address rebuilding

That makes this the same general flavor as the earlier `bsl` wins:

- not a new algorithm
- not a traversal change
- a source-shape cleanup inside the hottest SVE path

## 2. Starting point

The current best baseline was already strong:

- `bsl`-based add/sub logic
- `bsl`-shaped SVE emit logic
- aligned hot-loop peel
- `pair2` packed partial layout
- exact-two-row blocked hot pipeline

The remaining question was whether the aligned SVE middle was still spending too much of its instruction budget on local bookkeeping.

In the baseline file, the aligned vector body still looked like:

```cpp
for (; word_index < aligned_simd_word_end; word_index += context.sve_words) {
    const size_t aligned_flat_word_index0 =
        row0_base + static_cast<size_t>(word_index);
    const size_t aligned_flat_word_index1 =
        row1_base + static_cast<size_t>(word_index);
    const size_t partial_pair_index =
        horizontal_partial_pair_index(word_index);
    ...
    vector_backend.load_counts(context, word_index, ...);
    vector_backend.emit_next_state(context, aligned_flat_word_index0, ...);
    ...
    vector_backend.store_counts(context, word_index, ...);
    vector_backend.store_partial(
        recycled_partial_pairs0[partial_pair_index], ...);
    vector_backend.store_partial(
        recycled_partial_pairs1[partial_pair_index], ...);
}
```

That is correct, but it means the hottest aligned pair2 path still asks GCC to repeatedly carry:

- row-base plus `word_index` addressing
- count-plane plus `word_index` addressing
- partial-pair indexing from `word_index >> 1`

inside the same loop body that already contains the real SVE arithmetic.

## 3. What was the inefficiency?

The old aligned path was already fixed for memory alignment.

So this was **not** another “unaligned load” experiment.

The remaining inefficiency was instruction-stream shape:

- the aligned vector body was still inlined into `run_one_generation_hot_rows(...)`
- it mixed the real arithmetic with repeated base+index address formation
- the same induction variable fed many unrelated arrays and pair-index expressions

That tends to create extra integer-address work around the actual SVE logic.

The important point is that the kernel was not doing extra algorithmic work.
It was doing extra *per-iteration bookkeeping work* in the hottest aligned pair2 loop.

## 4. Fix: extract a pointer-stepped aligned pair2 helper

The new candidate moves that aligned vector body into a dedicated helper:

```cpp
__attribute__((noinline)) static void run_two_hot_rows_pair2_sve_ptrstep(...)
```

The caller now computes the starting pointers once:

```cpp
if (word_index < aligned_simd_word_end) {
    const int pair_count =
        (aligned_simd_word_end - word_index) / context.sve_words;
    const size_t aligned_flat_word_index0 =
        row0_base + static_cast<size_t>(word_index);
    const size_t aligned_flat_word_index1 =
        row1_base + static_cast<size_t>(word_index);
    const size_t partial_pair_index =
        horizontal_partial_pair_index(word_index);
    run_two_hot_rows_pair2_sve_ptrstep(
        pair_count,
        context.adults + aligned_flat_word_index0,
        context.juveniles + aligned_flat_word_index0,
        context.eggs + aligned_flat_word_index0,
        context.adults + aligned_flat_word_index1,
        context.juveniles + aligned_flat_word_index1,
        context.eggs + aligned_flat_word_index1,
        incoming_adult_row0 + word_index,
        incoming_adult_row1 + word_index,
        context.next_eggs + aligned_flat_word_index0,
        context.next_eggs + aligned_flat_word_index1,
        context.total_b0 + word_index,
        context.total_b1 + word_index,
        context.total_b2 + word_index,
        context.total_b3 + word_index,
        context.total_b4 + word_index,
        recycled_partial_pairs0 + partial_pair_index,
        recycled_partial_pairs1 + partial_pair_index);
    word_index = aligned_simd_word_end;
}
```

Then the helper advances raw pointers directly:

```cpp
for (int pair_index = 0; pair_index < pair_count; ++pair_index) {
    ...
    row0_adults_ptr += pair2_words;
    row0_juveniles_ptr += pair2_words;
    row0_eggs_ptr += pair2_words;
    row1_adults_ptr += pair2_words;
    row1_juveniles_ptr += pair2_words;
    row1_eggs_ptr += pair2_words;
    incoming_adult_word_ptr0 += pair2_words;
    incoming_adult_word_ptr1 += pair2_words;
    row0_next_eggs_ptr += pair2_words;
    row1_next_eggs_ptr += pair2_words;
    total_b0_ptr += pair2_words;
    total_b1_ptr += pair2_words;
    total_b2_ptr += pair2_words;
    total_b3_ptr += pair2_words;
    total_b4_ptr += pair2_words;
    ++partial_pairs0;
    ++partial_pairs1;
}
```

Important scope note:

- algorithm unchanged
- exact counts unchanged
- emit logic unchanged
- threading unchanged
- edge handling unchanged

Only the aligned pair2 SVE sweep shape changed.

## 5. What changed in source

Three small source-level additions made this possible:

1. pointer-based helpers inside `Pair2SVEHotWordBackend`

```cpp
compute_partial_from_word_ptr(...)
load_counts_from_ptrs(...)
store_counts_to_ptrs(...)
emit_next_state_from_word_ptrs(...)
```

2. the extracted helper:

```cpp
run_two_hot_rows_pair2_sve_ptrstep(...)
```

3. replacement of the old inline aligned-loop body with one helper call

So the new candidate still performs the same logical operations:

- load outgoing partials
- compute incoming partials
- load rolling counts
- emit row 0
- slide window
- emit row 1
- slide window
- store counts
- recycle partial rows

But the address setup is now hoisted out of the inner aligned pair2 sweep.

## 6. Why this helps

The best explanation is:

1. the caller computes the aligned starting addresses once
2. the extracted helper walks those arrays with simple pointer increments
3. the hottest aligned pair2 path contains less mixed address/index noise
4. GCC gets a cleaner vector loop to schedule

This is the same type of win as the earlier `bsl` emit rewrite:

- same truth table
- same dataflow
- better code shape for the compiler

It is especially plausible here because the previous best version had already removed bigger bottlenecks.

Once the algorithm and boolean network are already tight, small address-generation cleanups in the hottest loop can matter.

## 7. Generated-code evidence

I checked the generated hot symbols on the remote machine.

Baseline:

```text
run_one_generation_hot_rows(...): 0x1a8c
```

New candidate:

```text
run_two_hot_rows_pair2_sve_ptrstep(...): 0x03a4
run_one_generation_hot_rows(...):        0x1808
```

That does **not** mean the entire hot path vanished.
It means the old inline aligned pair2 body was pulled into a compact dedicated helper, and the main hot-row driver became smaller by `0x284` bytes.

That is exactly the direction I wanted:

- less monolithic hot-row function shape
- a smaller top-level hot dispatcher
- a dedicated helper for the dominant aligned pair2 case

I also checked the helper assembly on the remote machine with `-fverbose-asm -S`.
The aligned pair2 path still contains the expected SVE `bsl` instructions from the existing add/sub and emit rewrites, so this change did not undo the previous code-shape wins.

## 8. Correctness result

Remote validation command:

```bash
python3 -u run_tests.py \
  reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep.cpp \
  512 \
  reference/build_spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep.sh
```

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

I also ran a larger exact-output compare against the current best baseline on:

```text
test_grids/public_1_random_low_8192.bin
```

for `500` generations.

Result:

```text
MATCH
```

## 9. Benchmark result against the current best baseline

Comparison target:

- old: `...refactored_inplace_adults_sve_emit_bsl.cpp`
- new: `...refactored_inplace_adults_sve_emit_bsl_ptrstep.cpp`

Large-grid test:

```bash
./<binary> test_grids/public_1_random_low_32768.bin /dev/null 500
```

Setup:

- remote host: `jump-bootcamp`
- compiler: `g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra -pthread`
- warmup: `50` generations each
- measured: `3` paired runs each

Program-reported times:

- old: `6249.427 ms`, `6240.866 ms`, `6246.785 ms`
- new: `6146.169 ms`, `6144.235 ms`, `6140.527 ms`

Averages:

- old: `6245.693 ms`
- new: `6143.644 ms`

Delta vs old:

- `-102.049 ms`
- `1.634%` speedup

Wall-clock times moved in the same direction:

- old: `12.87 s`, `12.50 s`, `12.08 s`
- new: `12.51 s`, `12.45 s`, `11.96 s`

Wall-clock averages:

- old: `12.483 s`
- new: `12.307 s`

Wall delta vs old:

- `-0.177 s`
- `1.415%` speedup

So this is a real win over the actual current-best baseline, not just over an older branch of the experiment tree.

## 10. Why the new version is faster

The short answer is:

- same algorithm
- same SVE backend
- same emit logic
- cleaner aligned pair2 loop shape
- less hot-loop address/index bookkeeping

More concretely:

- the old version rebuilt multiple base+index addresses inside the aligned SVE middle
- the new version computes starting addresses once and advances them directly
- the dominant aligned pair2 path now lives in a compact dedicated helper
- the top-level hot function becomes smaller
- the compiler has a cleaner inner loop to schedule around the real SVE math

That is why this result is believable:

- the source change is narrow
- correctness stayed intact
- the large exact compare still matches
- the benchmark improved on the required remote target
- the generated hot-row shape changed in the expected direction

## Takeaway

This optimization worked for the same reason the earlier good experiments worked:

- find a real hot-path inefficiency
- keep the change narrow
- express the same work in a form the compiler can lower more cleanly
- verify remotely

In short: the new `ptrstep` candidate is faster than the current pure-SVE `sve_emit_bsl` baseline because the aligned pair2 SVE sweep was rewritten from an index-heavy inline body into a pointer-stepped dedicated helper, giving GCC a cleaner hot loop and producing a measured `~1.63%` speedup on the remote large-grid benchmark.
