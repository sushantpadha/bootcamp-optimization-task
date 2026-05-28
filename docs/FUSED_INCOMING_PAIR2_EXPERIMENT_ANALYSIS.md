# Fused Incoming Pair2 Experiment Analysis

This note explains the fused incoming-partial rewrite that improved the current `aligned_hot_loop_pair2` baseline.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming.cpp`

## 1. Starting point

The source baseline for this experiment was already one of the strongest variants in the tree:

- `bsl`-based add/sub logic was already in place
- the aligned hot-loop peel was already in place
- the `pair2` horizontal-partial layout was already in place

That meant the next change had to be narrow and evidence-driven.

The hot row-major loop was still doing two separate things for the incoming row:

1. build the full horizontal-partial row for `row_index + 3`
2. later reload those just-built partials inside the main rolling-update loop

So even after the `pair2` layout win, the code still had a local memory round-trip in the hottest path.

## 2. What the current baseline was still doing

In the baseline `pair2` file, each hot-row iteration begins by building the entire incoming row:

```cpp
compute_horizontal_partial_row(
    row_index + 3,
    words_per_row,
    current_adults,
    hot_row_scratch.incoming_partial);
```

Only after that producer pass finishes does the main word loop start consuming those partials:

```cpp
load_horizontal_partial_pair2_u64_sve(
    pg,
    incoming_partial_pairs[partial_pair_index],
    incoming_partial_b0,
    incoming_partial_b1,
    incoming_partial_b2);
```

So the real per-row structure was:

```text
for row r:
  build incoming_partial[all word blocks] for row r+3
  then loop over all word blocks and consume incoming_partial[...]
```

That means the incoming partial for a given block follows this path:

```text
compute -> store to incoming_partial buffer -> reload from incoming_partial buffer -> use once
```

The code was correct and still streamed rows well, but the just-computed incoming row was being materialized into memory before it was used.

## 3. Why that was a good target

This was not a speculative rewrite. It followed directly from the profiling work on the current best `pair2` baseline.

The profile showed that a large amount of time was still going into:

- horizontal window formation
- partial-pair buffer materialization and reload
- rolling-total loads/stores

The important point was not that the algorithm was wrong. The point was that the code was still paying for an unnecessary local round-trip in the hot loop:

```text
incoming partial row is built
incoming partial row is stored
incoming partial row is reloaded almost immediately
```

That suggested a cleaner row-major formulation:

- keep the good streaming loop order
- compute each incoming partial on demand
- use it immediately
- store it only once into the recycled ring row so it can become the outgoing row 5 iterations later

## 4. New idea: fuse production and consumption

The goal was not to change the rolling algorithm.

The goal was only to change the local dataflow from:

```text
produce whole incoming row first
then consume it later
```

to:

```text
for each word block:
  load outgoing partial
  compute incoming partial directly from adult row r+3
  use incoming partial immediately in the total update
  store it into the recycled ring slot for future outgoing reuse
```

That preserves the important properties of the baseline:

- still row-major
- still same 5-row ring idea
- still same rolling-total update
- still same `pair2` partial layout

So this was a local fusion, not a traversal rewrite.

## 5. Implementation strategy

The new variant added two small helpers:

- scalar helper for one word:

```cpp
compute_horizontal_partial_scalar_from_adult_row(...)
```

- SVE helper for one `pair2` tile:

```cpp
compute_horizontal_partial_pair2_u64_sve_from_adult_row(...)
```

Those helpers compute the exact same three horizontal-partial bitplanes as the old producer path, but they return them directly in registers.

Then the hot row-major loop was rewritten so that the outgoing ring slot is recycled in place:

```cpp
HorizontalPartialRow& recycled_row =
    hot_row_scratch.row_partials[ring_head];
```

That recycled row serves two roles in the same iteration:

- before overwrite: it holds the outgoing partials for the row that is leaving the 5-row window
- after overwrite: it becomes the storage for the new incoming partials from `row_index + 3`

This avoids the separate `incoming_partial` producer buffer in the hot path.

## 6. Before and after

### Baseline shape

The old baseline was effectively:

```text
for row r:
  build incoming partial row for r+3

  for each word block j:
    load outgoing partial
    load incoming partial from scratch buffer
    load rolling total
    compute next state
    total = total - outgoing + incoming
    store rolling total

  swap outgoing-row slot with incoming buffer
```

### New fused shape

The new fused variant is effectively:

```text
for row r:
  recycled_row = ring[head]
  incoming_adult_row = adults[r+3]

  for each word block j:
    load outgoing partial from recycled_row
    compute incoming partial directly from incoming_adult_row
    load rolling total
    compute next state
    total = total - outgoing + incoming
    store rolling total
    overwrite recycled_row[j] with the new incoming partial

  advance ring head
```

So the incoming-partial dataflow becomes:

```text
compute -> use immediately -> store once for future reuse
```

instead of:

```text
compute -> store -> reload -> use
```

## 7. Exact code changes

The most important structural changes in the new file are:

- helper addition for scalar incoming-partial computation
- helper addition for SVE `pair2` incoming-partial computation
- removal of the hot-loop call that prebuilds `hot_row_scratch.incoming_partial`
- in-place recycling of `hot_row_scratch.row_partials[ring_head]`
- immediate storeback of the newly computed incoming partial into that recycled slot

In practical terms, the baseline hot path changed from:

```cpp
compute_horizontal_partial_row(..., hot_row_scratch.incoming_partial);
load incoming from hot_row_scratch.incoming_partial;
```

to:

```cpp
compute incoming directly from incoming_adult_row;
use it immediately;
store it into recycled_row;
```

Important scope note:

- algorithm unchanged
- threading unchanged
- edge handling unchanged
- `pair2` layout unchanged
- rolling-total math unchanged
- only the incoming-partial production/consumption schedule changed

## 8. Correctness result

Remote validation command:

```bash
python3 -u run_tests.py \
  reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming.cpp \
  512 \
  reference/build_spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming.sh
```

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

Large-grid exact-output check against the current best `pair2` baseline also matched:

```text
exact_output_compare=MATCH
```

## 9. Benchmark result against the current best pair2 baseline

Large-grid test:

```bash
./<binary> test_grids/public_1_random_low_32768.bin /dev/null 500
```

Exact-output comparison run on the same remote host:

| Variant | Program time |
| --- | ---: |
| `aligned_hot_loop_pair2` baseline | `7539.465 ms` |
| `aligned_hot_loop_pair2_fused_incoming` experiment | `7004.269 ms` |

Alternating warm runs:

| Run | Baseline | Fused |
| --- | ---: | ---: |
| 1 | `7540.366 ms` | `7005.790 ms` |
| 2 | `7500.680 ms` | `6999.138 ms` |
| 3 | `7592.665 ms` | `6998.190 ms` |

Averages:

| Variant | Average program time |
| --- | ---: |
| `aligned_hot_loop_pair2` baseline | `7544.570 ms` |
| `aligned_hot_loop_pair2_fused_incoming` experiment | `7001.039 ms` |

Delta vs baseline:

- `-543.531 ms`
- `7.20%` speedup by program-reported time

This was a clear and repeatable win, not a one-off run.

## 10. Why this optimization worked

This win came from removing redundant hot-path traffic without damaging the baseline’s good row-wise streaming behavior.

That matters because a previous "vertical microkernel" style idea had already shown the opposite failure mode:

- it removed some state traffic
- but it damaged the machine-friendly streaming order badly enough to lose overall

The fused incoming rewrite succeeds because it is much more conservative:

- no block-major traversal
- no prefetch-heavy rewrite
- no larger working set
- no change to the rolling recurrence

Instead it fixes one specific inefficiency:

```text
build incoming row
store it
reload it
use it
```

becomes:

```text
build incoming partial
use it immediately
store it once for later outgoing reuse
```

So the kernel keeps the good parts of the baseline and removes one unnecessary buffer round-trip.

## Takeaway

This experiment worked because it attacked a real hot-path inefficiency while preserving the existing row-major structure that the machine already liked.

In short: the fused incoming-partial rewrite kept the strong `pair2` row-major kernel intact, removed the immediate store/reload cycle for the incoming row, and produced a real `~7.2%` speedup over the current best `aligned_hot_loop_pair2` baseline.
