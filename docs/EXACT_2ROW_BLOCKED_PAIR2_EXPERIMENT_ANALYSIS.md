# Exact 2-Row Blocked Pair2 Experiment Analysis

This note explains the exact two-row blocked rewrite that produced a small win over the current `fused_incoming_pair2` baseline.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row.cpp`

## 1. Starting point

The source baseline for this experiment was already the fastest known variant in the tree:

- `pair2` horizontal-partial layout
- aligned SVE hot loop
- `bsl`-based add/sub logic
- fused incoming-partial production

That meant the next change had to be very narrow. The fused-incoming rewrite had already removed one local round-trip:

```text
compute incoming partial
-> store to temporary row buffer
-> reload immediately
-> use once
```

But the hot loop still updated the vertical total one row at a time.

## 2. What the fused baseline was still doing

In the fused-incoming baseline, each hot-row iteration does this for every word block:

1. load exact `rolling_total` bits `b0..b4`
2. emit output for row `r`
3. subtract outgoing partial for row `r-2`
4. add incoming partial for row `r+3`
5. store exact `rolling_total` back

Then on the very next iteration, for row `r+1`, it reloads that same exact total and repeats the process.

So the per-row structure is effectively:

```text
load T
emit row r
update T -> T1
store T1

load T1
emit row r+1
update T1 -> T2
store T2
```

The important point is that the representation is already exact and correct. The remaining inefficiency is the immediate reload/store handoff between two adjacent rows.

## 3. Why this was a good target

This idea follows the same pattern as the successful fused-incoming rewrite:

- do not change the algorithm
- do not change traversal order
- do not damage row-major locality
- only remove a local state round-trip

The previous positive/negative experiment lost because it added extra normalization arithmetic. The vertical/block-major experiment lost because it damaged locality. So the next plausible move was simpler:

- keep the exact representation
- keep the row-major traversal
- keep the fused incoming-partial production
- just keep the exact total live across two successive hot rows

## 4. New idea: exact two-row blocking

The goal was not to invent a new accumulator. The goal was only to process two successive hot rows while the exact total remains live in registers.

Conceptually:

```text
load T
emit row r
update T -> T1
emit row r+1
update T1 -> T2
store T2
```

This removes one `rolling_total` load/store round-trip per 2 hot rows without changing the arithmetic itself.

## 5. Before and after

### Fused-incoming baseline shape

```text
for each hot row r:
  for each word block j:
    load outgoing partial for row r-2
    compute incoming partial for row r+3
    load exact total T[j]
    emit output for row r
    T[j] = T[j] - outgoing + incoming
    store exact total T[j]
    recycle outgoing slot with the new incoming partial
```

### Exact two-row blocked shape

```text
for hot rows r, r+1 in pairs:
  for each word block j:
    load outgoing0 for row r-2
    compute incoming0 for row r+3
    load outgoing1 for row r-1
    compute incoming1 for row r+4

    load exact total T[j]

    emit output for row r
    T[j] = T[j] - outgoing0 + incoming0

    emit output for row r+1
    T[j] = T[j] - outgoing1 + incoming1

    store exact total T[j] once
    recycle both outgoing slots with incoming0/incoming1
```

So the core change is:

```text
two output rows per total load/store
```

instead of:

```text
one output row per total load/store
```

## 6. Implementation strategy

The implementation stayed deliberately conservative.

What changed:

- the hot loop processes 2 hot rows at a time
- it reads two outgoing ring rows
- it computes two incoming rows on the fly
- it keeps one exact `b0..b4` accumulator live across both rows
- it stores that accumulator only after the second row update

What did not change:

- exact accumulator representation
- add/sub helpers
- `pair2` layout
- row-major traversal
- fused incoming-partial helpers
- final-row emit path

Important detail:

- a single-row fallback remains for any leftover row before the final emit-only row

So this was not a representation rewrite. It was only a scheduling rewrite inside the hot path.

## 7. Exact code changes

The new experiment file rewrites only the hot-row update schedule.

At a high level:

- baseline hot loop:

```cpp
load total;
emit row r;
update total with outgoing/incoming for row r;
store total;
```

- exact-2-row hot loop:

```cpp
load total;
emit row r;
update total with outgoing0/incoming0;
emit row r+1;
update total with outgoing1/incoming1;
store total;
```

The scalar and SVE pair2 paths both follow that same structure.

## 8. Correctness result

Remote validation command:

```bash
python3 -u run_tests.py \
  reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row.cpp \
  512 \
  reference/build_spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row.sh
```

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

Large-grid exact-output check against the fused-incoming baseline also matched:

```text
MATCH
```

## 9. Benchmark result against the current fastest baseline

Large-grid test target:

```bash
test_grids/public_1_random_low_32768.bin
```

Generations:

```text
500
```

Alternating warm runs on the remote machine:

- fused-incoming baseline:
  - `7001.161 ms`
  - `7011.212 ms`
  - `7000.011 ms`
  - `6966.280 ms`
  - `6959.604 ms`
  - `6964.423 ms`

- exact-2-row experiment:
  - `6968.602 ms`
  - `6966.777 ms`
  - `6968.718 ms`
  - `6936.154 ms`
  - `6920.811 ms`
  - `6929.821 ms`

Average:

| Variant | Average time |
| --- | ---: |
| `fused_incoming_pair2` baseline | `6983.782 ms` |
| `fused_incoming_exact2row` experiment | `6948.481 ms` |

Delta vs baseline:

- `-35.301 ms`
- `0.51%` speedup

The signal was small, but it was consistent across all 6 alternating pairs.

## 10. Why the win is real but small

This optimization worked for the same reason the fused-incoming rewrite worked:

- it removed a local hot-path round-trip
- it preserved the good row-major streaming structure
- it did not add a new representation or a decode cost

But the upside is limited because the savings are also narrow.

The new version still has to:

- load two outgoing partial rows
- compute two incoming partial rows
- perform two exact subtract/add updates
- keep the exact accumulator live for longer

So the optimization reduces state traffic, but it also slightly increases live ranges and instruction scheduling pressure. That is why the measured gain is only about half a percent.

## 11. Takeaway

This experiment is worth keeping because it is:

- correct
- simple
- localized
- modestly faster

It also reinforces the pattern that has worked best on this kernel:

1. preserve row-major streaming locality
2. avoid speculative arithmetic redesigns
3. look for local hot-path round-trips that can be removed cleanly

In short: the exact two-row blocked rewrite is not a major breakthrough, but it is a clean incremental improvement over the current fastest fused-incoming baseline.
