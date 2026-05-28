# SLI/SRI Ptrstep Experiment Analysis

This note explains the hot-SVE shift-shape cleanup that improved the current `svnot_eor3` `ptrstep` baseline.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep_svnot_eor3.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep_svnot_eor3_sli_sri_orn.cpp`

## 1. Why this comparison matters

This is an apple-to-apple comparison against the current best `svnot_eor3` `ptrstep` pure-SVE version.

The new candidate keeps:

- the same algorithm
- the same `pair2` partial layout
- the same exact-two-row blocked hot schedule
- the same pointer-stepped aligned SVE sweep
- the same `svnot_x` / `sveor3` boolean cleanup
- the same pure-SVE emit logic
- the same threading model
- the same data representation

So the intended optimization here is narrower:

- keep the hot helper structure exactly as it is
- change only how cross-word neighbour shifts are spelled in the hottest SVE path
- make the compiler emit direct SVE2 shift-insert instructions for those neighbour words

This is the same style as the earlier boolean-shape cleanup:

- no algorithm change
- no schedule change
- no layout change
- just a tighter instruction shape for work the kernel already does

## 2. Starting point

The `svnot_eor3` baseline was already strong:

- pointer-stepped aligned pair2 SVE sweep
- direct `svnot_x(...)` for hot NOTs
- direct `sveor3(...)` for hot 3-input XORs
- pure-SVE exact-count add/sub helpers
- pure-SVE emit logic

At that point, one remaining hot shape still stood out in the helper block:

- cross-word neighbour generation still used two shifts plus an OR

That happens in the packed two-word path that computes:

- `left_2`
- `left_1`
- `right_1`
- `right_2`

for every hot tile.

## 3. The remaining inefficiency

The old helper code still expressed incoming-bit neighbour formation like this:

```cpp
template <int Shift>
ALWAYS_INLINE svuint64_t shift_left_with_incoming(
    svuint64_t curr,
    svuint64_t prev) const
{
    return or_words(shift_left<Shift>(curr), shift_right<64 - Shift>(prev));
}

template <int Shift>
ALWAYS_INLINE svuint64_t shift_right_with_incoming(
    svuint64_t curr,
    svuint64_t next) const
{
    return or_words(shift_right<Shift>(curr), shift_left<64 - Shift>(next));
}
```

That is correct, but it means each neighbour word is still built from:

- one shift of the current word
- one shift of the adjacent word
- one OR to merge them

So the hot partial-generation path is paying for that 3-op shape four times per tile.

## 4. Strategy

The strategy was to keep the helper API and truth table the same, but rewrite the neighbour-word helpers in a form that directly maps to SVE2 shift-insert instructions:

1. seed the carry-in bits with the already-needed adjacent-word shift
2. inject the current-word bits with `sli` or `sri`
3. keep everything else unchanged

Important scope note:

- no algorithm change
- no count-accumulation rewrite
- no emit-logic rewrite
- no scalar-path change
- no NEON-path change

Only the cross-word shift helpers changed.

## 5. The key rewrite

The new candidate changed `shift_left_with_incoming(...)` from:

```cpp
return or_words(shift_left<Shift>(curr), shift_right<64 - Shift>(prev));
```

to:

```cpp
svuint64_t result = shift_right<64 - Shift>(prev);
if constexpr (Shift == 1) {
    asm("sli %0.d, %1.d, #1" : "+w"(result) : "w"(curr));
} else {
    asm("sli %0.d, %1.d, #2" : "+w"(result) : "w"(curr));
}
return result;
```

And it changed `shift_right_with_incoming(...)` from:

```cpp
return or_words(shift_right<Shift>(curr), shift_left<64 - Shift>(next));
```

to:

```cpp
svuint64_t result = shift_left<64 - Shift>(next);
if constexpr (Shift == 1) {
    asm("sri %0.d, %1.d, #1" : "+w"(result) : "w"(curr));
} else {
    asm("sri %0.d, %1.d, #2" : "+w"(result) : "w"(curr));
}
return result;
```

So the logical result stays identical:

- the carry-in bits still come from the adjacent word
- the main body bits still come from the current word

But the final merge is now expressed as a direct SVE2 shift-insert instead of a separate OR.

## 6. Which hot helpers changed

The optimization intentionally touched only these two packed-tile SVE helpers:

- `shift_left_with_incoming<1/2>(...)`
- `shift_right_with_incoming<1/2>(...)`

Everything else in the hot helper block stayed as it was in the `svnot_eor3` winner.

That matters because this experiment is trying to answer a narrow question:

- does a better instruction shape for neighbour-bit formation help inside the already-good `svnot_eor3` kernel?

## 7. Why this works

The key idea is that the old source shape still forced the compiler to see an explicit boolean merge after the two shifts.

The new form helps because:

- `sli` and `sri` are built for this exact lane-local bit-insertion pattern
- they preserve one side and inject the other side in a single instruction shape
- they reduce the amount of explicit boolean glue in the hot neighbour-generation path

This matters because those neighbour helpers are used repeatedly when generating each horizontal partial.

Once the larger issues were already removed by earlier passes, that remaining per-tile shift/merge shape became worth tightening.

## 8. Why the change is still exactly correct

The new helper is just a different spelling of the same packed-lane identity:

- `curr << Shift` with incoming top bits from `prev >> (64 - Shift)`
- `curr >> Shift` with incoming low bits from `next << (64 - Shift)`

The candidate still computes exactly those values.

So it does not change:

- neighbour semantics
- partial-count semantics
- exact-count updates
- next-state truth tables
- boundary handling

It only changes how the same bit pattern is formed in the hot SVE path.

## 9. Codegen check

I checked generated assembly on the remote machine.

The baseline assembly had no `sli` or `sri` matches in the hot code excerpts.

The candidate assembly did show explicit SVE2 shift-insert instructions, for example:

```asm
sli z25.d, z27.d, #2
sli z28.d, z27.d, #1
sri z24.d, z27.d, #1
sri z26.d, z27.d, #2
```

and later:

```asm
sli z2.d, z30.d, #2
sli z1.d, z30.d, #1
sri z27.d, z30.d, #1
sri z0.d, z30.d, #2
```

That confirms the source rewrite did what it was meant to do:

- make the neighbour helpers lower to direct `sli` / `sri` instructions in the real generated hot path

## 10. Correctness result

Remote validation command:

```bash
python3 -u run_tests.py \
  reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep_svnot_eor3_sli_sri_orn.cpp \
  512 \
  reference/build_spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep_svnot_eor3_sli_sri_orn.sh
```

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

I also ran an exact remote comparison against the `svnot_eor3` baseline on:

- `test_grids/public_1_random_low_8192.bin`
- `1000` generations

Result:

```text
exact_match=true
```

The baseline and candidate output files had identical SHA-256 hashes.

## 11. Benchmark result against the current best

Comparison target:

- baseline: `..._sve_emit_bsl_ptrstep_svnot_eor3.cpp`
- candidate: `..._sve_emit_bsl_ptrstep_svnot_eor3_sli_sri_orn.cpp`

Large-grid benchmark:

```bash
./<binary> test_grids/public_1_random_low_32768.bin /dev/null 1000
```

Setup:

- remote host: `jump-bootcamp`
- compiler: `g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra -pthread`
- warmup: `100` generations each
- measured: `3` alternating runs each

Program-reported times:

- baseline: `11975.595 ms`, `12000.038 ms`, `12009.008 ms`
- candidate: `11875.223 ms`, `11849.692 ms`, `11873.293 ms`

Averages:

- baseline: `11994.880 ms`
- candidate: `11866.069 ms`

Delta vs baseline:

- `-128.811 ms`
- `1.074%` speedup

Wall-clock times:

- baseline: `17.670 s`, `17.700 s`, `17.700 s`
- candidate: `17.580 s`, `17.520 s`, `17.560 s`

Wall-clock averages:

- baseline: `17.690 s`
- candidate: `17.553 s`

Wall-clock delta vs baseline:

- `-0.137 s`
- `0.773%` speedup

So this was a real win over the current `svnot_eor3` best version, not just a code-shape curiosity.

## 12. Why this optimization was worth doing

This pass is a good example of what remains after the larger boolean and structural cleanups are already done.

Earlier passes had already improved:

- the boolean NOT shape
- the 3-input XOR shape
- the pointer-stepped aligned hot sweep

Once those were in place, the repeated neighbour-word formation in the horizontal partial path became a visible target.

This pass won because it cleaned up exactly that remaining source shape:

- direct SVE2 shift-insert instructions instead of shift-plus-OR merges

inside one of the hottest helper paths in the kernel.

## Takeaway

The best move here was not another broad rewrite.

The best move was:

1. keep the current `svnot_eor3` winner intact
2. identify the remaining repeated shift/merge pattern in the hot SVE helper path
3. rewrite that pattern directly as `sli` / `sri`

That preserved correctness, kept the patch small, and produced a measurable speedup over the current best version.
