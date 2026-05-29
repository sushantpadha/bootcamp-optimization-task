# In-Place Adults Experiment Analysis

This note explains the in-place-next-adult rewrite that removes the separate `next_adults` grid from the exact-two-row `pair2` kernel.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults.cpp`

## 1. Starting point

The source baseline for this experiment was already a strong exact-two-row `pair2` variant:

- `pair2` horizontal-partial layout
- aligned SVE hot loop
- fused incoming-partial production
- exact two-row blocked rolling-total schedule

That meant the next change had to be structural but still conservative.

The important recurrence for the cell-state planes is:

```text
A(t+1) = J(t) OR surviving_adults(A(t), count)
J(t+1) = E(t)
E(t+1) = births(A(t), J(t), E(t), count)
```

The baseline stored those results in 5 full-grid buffers:

- `current_adults`
- `current_juveniles`
- `current_eggs`
- `next_adults`
- `next_eggs`

But `J(t+1)` is already just a rename of `E(t)`, so the remaining question was narrower:

- can `A(t+1)` be written directly into the old juvenile buffer and remove `next_adults` entirely?

## 2. What the refactored baseline was still doing

In the refactored exact-two-row baseline, the hot kernel still treated `next_adults` as a separate destination:

```text
read old adults / juveniles / eggs
compute next adult bits
store them into next_adults
compute next egg bits
store them into next_eggs
```

Then the generation loop rotated like this:

```text
current_adults = next_adults
current_juveniles = current_eggs
current_eggs = next_eggs
```

So even though the logical state machine already has only one genuinely new plane per generation beyond `A(t+1)`, the implementation still carried a dedicated full-grid `next_adults` buffer.

## 3. Why this was a good target

This was not an attempt to change the algorithm or the neighbour-count machinery.

The sliding-window logic only depends on adult bits:

- horizontal partials are built from adult rows only
- vertical totals count adult neighbours only
- juveniles and eggs affect only the center cell's transition

That means the old juvenile plane is only needed as an input to the current row's transition rule, not as part of the sliding window itself.

So the tempting rewrite is:

```text
use old juvenile bits once
then overwrite that same location with A(t+1)
```

The real correctness question is not arithmetic. It is schedule safety:

- are any later passes in the same generation still going to read old juvenile bits from the locations we overwrite?

## 4. New idea: write next adults in place

The new formulation keeps 4 full-grid buffers:

- `current_adults`
- `current_juveniles`
- `current_eggs`
- `next_eggs`

and changes the destination rules to:

```text
current_juveniles <- A(t+1)
next_eggs         <- E(t+1)
```

Then the end-of-generation rotation becomes:

```text
current_adults.swap(current_juveniles)
current_juveniles.swap(current_eggs)
current_eggs.swap(next_eggs)
```

Conceptually:

```text
new A = old J buffer after overwrite with A(t+1)
new J = old E
new E = old next_eggs
```

So this is a true 5-buffer to 4-buffer reduction.

## 5. The main hazard and how it was handled

The hot interior overwrite is only safe because the overwritten region does not overlap the later fallback reads of old juvenile bits.

The hot kernel writes only:

```text
rows  2 .. grid_size-3
words 1 .. words_per_row-2
```

The later fallback code reads old juvenile bits only from:

- top wrapping rows
- bottom wrapping rows
- first word of interior rows
- last word of interior rows

So the hot overwrite does not clobber any juvenile bits that those later passes still need.

The edge paths were the harder part. In the old code they relied on a prefilled `next_adults` word and then OR-ed adult-survivor bits into it.

That no longer works if the destination is the old juvenile word, because the edge word also contains wrap columns that the wrapping fallback still has to read later in the same generation.

So the edge-word update had to change from:

```text
prefill destination
OR in valid next-adult bits
```

to:

```text
preserve the invalid columns
replace only the valid columns with final next-adult bits
```

That is why the new edge helper writes:

```text
next_adult_word = (next_adult_word & ~valid_mask) | next_adult_valid
```

instead of just OR-ing into the destination.

## 6. Before and after

### Refactored baseline shape

```text
for each generation:
  clear / prefill next_adults and next_eggs as needed

  for each owned row chunk:
    hot interior writes A(t+1) to next_adults
    edge/wrapping paths finish A(t+1) in next_adults
    all paths write E(t+1) to next_eggs

  rotate:
    A <- next_adults
    J <- old eggs
    E <- next_eggs
```

### In-place-adults shape

```text
for each generation:
  clear only next_eggs where needed

  for each owned row chunk:
    hot interior reads old juvenile bits once,
    then overwrites those hot cells with A(t+1) in current_juveniles

    edge/wrapping paths finish the non-hot cells of A(t+1)
    in current_juveniles without destroying still-needed wrap columns

    all paths write E(t+1) to next_eggs

  rotate:
    A <- overwritten juvenile buffer
    J <- old eggs
    E <- next_eggs
```

So the practical change is:

```text
delete next_adults
retarget all next-adult writes into current_juveniles
adjust edge masking and final buffer rotation
```

## 7. Exact code changes

The important structural changes in the new file are:

- `HotRowKernelContext` now exposes `juveniles` as a writable destination and no longer carries `next_adults`
- scalar hot-state emission writes `A(t+1)` directly to `context.juveniles[...]`
- SVE hot-state emission does the same
- `initialize_next_generation_rows(...)` now clears only `next_eggs`
- `finalize_masked_edge_word(...)` now computes full valid-lane adult results instead of OR-ing into a prefilled destination
- all edge/wrapping paths write adult results into `current_juveniles`
- the generation loop now uses the 4-buffer rotation

Important scope note:

- algorithm unchanged
- rolling exact-count representation unchanged
- hot-row traversal unchanged
- threading unchanged
- only next-state storage and generation rotation changed

## 8. Correctness result

Remote validation command:

```bash
python3 -u run_tests.py \
  reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults.cpp \
  512 \
  reference/build_spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults.sh
```

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

Large-grid exact-output check against the refactored exact-two-row baseline also matched:

```text
PASS
```

Additional 8192 / 500 timing from that exact-compare run:

- baseline stdout: `510.542 ms`
- candidate stdout: `510.427 ms`

## 9. Benchmark result against the current best refactored baseline

Large-grid test target:

```bash
test_grids/public_1_random_low_32768.bin
```

Generations:

```text
500
```

Alternating remote runs:

- refactored exact-two-row baseline:
  - `6780.228 ms`
  - `6779.715 ms`

- in-place-adults experiment:
  - `6605.800 ms`
  - `6607.503 ms`

Average:

| Variant | Average time |
| --- | ---: |
| `refactored_exact2row` baseline | `6779.971 ms` |
| `inplace_adults` experiment | `6606.651 ms` |

Delta vs baseline:

- `-173.320 ms` over 500 generations
- `-0.346640 ms` per iteration
- `2.556%` speedup

Wall-clock averages showed the same direction:

- baseline: `12.525 s`
- candidate: `12.440 s`

## 10. Comparison with an earlier serious contender

For context, the earlier `temporal2stripe` experiment was also rebuilt and rerun in the same alternating remote session.

Remote runs:

- `temporal2stripe`:
  - `7746.260 ms`
  - `7679.033 ms`

Average:

| Variant | Average time |
| --- | ---: |
| `temporal2stripe` | `7712.647 ms` |
| `inplace_adults` | `6606.651 ms` |

Delta vs `temporal2stripe`:

- `-1105.995 ms` over 500 generations
- `14.340%` faster

## 11. Why this optimization worked

This win did not come from changing the neighbour-count arithmetic.

It worked because the old implementation was carrying one extra full-grid destination plane for `A(t+1)` even though:

- the state recurrence already lets juveniles become adults directly
- the neighbour-count pipeline does not depend on juvenile bits
- the overwrite can be scheduled safely in the hot interior

So the new version removes one large write stream from the steady-state generation update while preserving the same sliding-window algorithm.

The gain is large enough to matter and was stable across both benchmark runs:

- correctness still holds remotely
- the 32768 / 500 benchmark improved in both runs
- wall-clock and program-stdout timings agree on the direction

## Takeaway

This rewrite was a state-layout optimization, not a counting optimization:

1. identify that `next_adults` is structurally redundant
2. prove that hot-cell overwrites do not interfere with later fallback reads
3. fix the edge-word paths so they preserve still-needed wrap columns
4. rotate the 4 surviving buffers correctly at generation boundaries
5. verify remotely and benchmark on the large grid

In short: writing `A(t+1)` directly into the old juvenile buffer successfully removed one full-grid array and produced a repeatable `~2.6%` speedup over the current refactored exact-two-row baseline.
