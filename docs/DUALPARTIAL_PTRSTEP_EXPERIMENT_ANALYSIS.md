# Dual-Partial Ptrstep Experiment Analysis

This note explains the dual-stream partial-builder rewrite that improved the current `svnot_eor3_sli_sri` `ptrstep` baseline.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep_svnot_eor3_sli_sri_orn.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep_svnot_eor3_sli_sri_dualpartial.cpp`

## 1. Why this comparison matters

This is an apple-to-apple comparison against the current best `svnot_eor3_sli_sri` pure-SVE `ptrstep` version.

The new candidate keeps:

- the same algorithm
- the same `pair2` partial layout
- the same exact-two-row blocked hot schedule
- the same pointer-stepped aligned SVE sweep
- the same `svnot_x(...)` / `sveor3(...)` cleanup
- the same `sli` / `sri` neighbour-shift cleanup
- the same pure-SVE emit logic
- the same threading model
- the same data representation

So the intended optimization here is narrower:

- keep the same per-row partial truth table
- keep the same count update and emit logic
- change only how the two incoming row partials are grouped in the hottest pair2 loop

This is the same style as the earlier wins:

- no algorithm change
- no schedule change at the grid level
- no layout change
- just a better source shape inside the hottest SVE path

## 2. Starting point

The `svnot_eor3_sli_sri` baseline was already strong:

- pointer-stepped aligned pair2 SVE sweep
- direct `svnot_x(...)` for hot NOTs
- direct `sveor3(...)` for hot 3-input XORs
- direct `sli` / `sri` neighbour shaping
- pure-SVE exact-count add/sub helpers
- pure-SVE emit logic

At that point, one remaining hot path still looked slightly awkward in source:

- row 0 partial generation was expressed as one helper call
- row 1 partial generation was expressed as another helper call
- then the count/update path started

That means GCC had to schedule two separate partial-builder subgraphs around the rest of the hot loop.

## 3. The remaining inefficiency

In the baseline pair loop, the incoming partials were built like this:

```cpp
vector_backend.load_partial(... row0 ...);
vector_backend.compute_partial_from_word_ptr(
    incoming_adult_word_ptr0,
    incoming_row0_b0,
    incoming_row0_b1,
    incoming_row0_b2);
vector_backend.load_partial(... row1 ...);
vector_backend.compute_partial_from_word_ptr(
    incoming_adult_word_ptr1,
    incoming_row1_b0,
    incoming_row1_b1,
    incoming_row1_b2);
```

That is correct, but it leaves GCC freedom to interleave:

- row 0 partial generation
- row 1 partial generation
- count-plane loads
- later emit/slide work

in a way that can lengthen live ranges and mix the two partial-builder clusters with unrelated work.

The underlying issue was not extra algorithmic work.
It was source shape in the hottest pair2 loop.

## 4. Strategy

The strategy was:

1. keep the existing single-row helper untouched for general use
2. add one new dual-stream helper specialized for the aligned pair2 hot loop
3. compute both incoming row partials inside that helper before the loop body continues into the count/update path

Important scope note:

- no loop-carried reuse
- no change to the number of logical partials produced
- no change to the rolling-window math
- no change to emit logic
- no scalar or NEON path change

Only the hot pair2 partial-builder grouping changed.

## 5. The key rewrite

The new candidate adds:

```cpp
ALWAYS_INLINE void compute_two_partials_from_word_ptrs(...)
```

That helper:

- loads `prev`, `curr`, `next` for row 0 and row 1
- builds `left_2` / `left_1` for both rows
- runs the first CSA for both rows
- builds `right_1` / `right_2` for both rows
- runs the second CSA for both rows
- finishes `partial_b0`, `partial_b1`, `partial_b2` for both rows

So instead of two separate helper invocations in the loop body, the hot loop now does:

```cpp
vector_backend.load_partial(... row0 ...);
vector_backend.load_partial(... row1 ...);
vector_backend.compute_two_partials_from_word_ptrs(
    incoming_adult_word_ptr0,
    incoming_adult_word_ptr1,
    incoming_row0_b0,
    incoming_row0_b1,
    incoming_row0_b2,
    incoming_row1_b0,
    incoming_row1_b1,
    incoming_row1_b2);
```

The truth table is unchanged.
Only the grouping and scheduling opportunity changed.

## 6. What changed in source

Two source-level additions made this possible:

1. a new dual-stream helper:

```cpp
compute_two_partials_from_word_ptrs(...)
```

2. one pair-loop call-site rewrite to use that helper

Everything else in the hot backend stayed as it was in the `svnot_eor3_sli_sri` winner.

That matters because this experiment is trying to answer a narrow question:

- can the compiler do slightly better if both incoming row partials are presented as one clustered hot subgraph?

## 7. Why this helps

The best explanation is:

1. the two incoming rows already have identical partial-generation structure
2. the new helper presents both of those structures together
3. GCC can schedule the left-side neighbour formation and first CSA stage more tightly across both rows
4. the hottest partial-builder region ends up slightly more clustered before the rest of the loop consumes more registers

This does not eliminate any major logical stage.
It is a smaller win than the earlier `svnot/eor3` and `sli/sri` cleanups.

But once the larger issues were already fixed, that remaining source-shape cleanup was still worth a measurable amount.

More concretely, the expected register-pressure benefit comes from live-range shape.

In this loop, live vector values include:

- `prev`, `curr`, `next`
- `left_*`, `right_*`
- `sum_*`, `carry_*`
- outgoing partial planes
- count planes
- state words used by emit/slide

With the old source shape, GCC could partially build row 0, then start bringing in unrelated count/state work, then partially build row 1 before all row-0 temporaries had died.
That raises the number of simultaneously live vector values.

The dual-stream helper encourages a tighter subgraph:

1. load both rows
2. build both rows' left-side neighbours
3. run both first CSA steps
4. build both rows' right-side neighbours
5. finish both second CSA steps
6. materialize both rows' `partial_b0`, `partial_b1`, `partial_b2`

When that cluster stays tighter, intermediates such as:

- `left0_2`, `left0_1`
- `sum0_1`, `carry0_1`
- `left1_2`, `left1_1`
- `sum1_1`, `carry1_1`

can die sooner instead of remaining live while unrelated count loads and later emit work are being mixed into the same region.

So the benefit is not that a true dependency disappeared.
The benefit is that the compiler gets a better-shaped local scheduling region where temporary vectors can have shorter live ranges, which reduces the chance of extra moves, rematerialization, or spills.

## 8. Codegen check

I checked the generated hot assembly on the remote machine with `-fverbose-asm`.

The intended effect only happened partially:

- the candidate does show the dual-stream left-side neighbour build and first CSA stage grouped more tightly
- but GCC still hoists some count loads into the early part of the hot block

So the source-level goal was not realized perfectly.

Still, the candidate hot excerpt shows the compiler processing:

- row 1 and row 0 `prev/curr` loads
- left-side shift seeds
- paired `sli` forms
- paired first-CSA `eor3` forms

earlier as one clustered block before moving further into the right-side stage and later work.

That partial clustering was enough to produce a small but real win.

## 9. Correctness result

Remote validation used the same candidate source and build script, invoked through short symlink names only to avoid a Linux path-length issue in `run_tests.py` log naming.

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

Remote exact compare on `test_grids/public_1_random_low_8192.bin` for `1000` generations also matched exactly:

```text
exact_match=true
```

## 10. Benchmark result against the previous best

Comparison target:

- previous best: `...svnot_eor3_sli_sri_orn.cpp`
- candidate: `...svnot_eor3_sli_sri_dualpartial.cpp`

Large-grid test:

```bash
./<binary> test_grids/public_1_random_low_32768.bin /dev/null 1000
```

Setup:

- remote host: `jump-bootcamp`
- compiler: `g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra -pthread`
- warmup: `100` generations each
- measured: `2` paired runs each

Program-reported times:

- previous best: `11957.098 ms`, `11676.261 ms`
- candidate: `11784.196 ms`, `11630.733 ms`

Averages:

- previous best: `11816.680 ms`
- candidate: `11707.465 ms`

Delta vs previous best:

- `-109.215 ms`
- `0.924%` speedup

Wall times:

- previous best: `17.680 s`, `17.330 s`
- candidate: `17.480 s`, `17.280 s`

Averages:

- previous best: `17.505 s`
- candidate: `17.380 s`

Delta vs previous best:

- `-0.125 s`
- `0.714%` speedup

So this was a real win over the current `svnot_eor3_sli_sri` baseline, although it is a modest one.

## 11. Why this optimization worked

This pass won because it gave the compiler a slightly better hot-subgraph shape for the two incoming row partials:

1. both rows’ partial-generation work is now presented together
2. the early neighbour-build plus first-CSA region is more tightly clustered
3. the hottest partial-builder state seems to stay a bit cleaner before the later count/update work takes over

The improvement is not dramatic because the baseline was already highly optimized.
But it is still a real, correctness-preserving hot-path improvement in the current best kernel.
