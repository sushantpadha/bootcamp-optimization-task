# Pair2 Partial Layout Experiment Analysis

This note explains the horizontal-partial layout rewrite that improved the current `aligned_hot_loop` variant.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2.cpp`

## 1. Starting point

The source baseline for this experiment was already doing two useful things:

- the `bsl` add/sub rewrite was in place
- the main hot loop had already been peeled so the hot vector body starts from an aligned word index

That still left one awkward subsystem: horizontal partial storage.

The baseline stores one partial as:

```cpp
struct HorizontalPartialWord {
    uint64_t b0;
    uint64_t b1;
    uint64_t b2;
};
```

and moves vector partials with `svst3_u64` / `svld3_u64`.

## 2. What was the issue?

On the remote machine:

- `svcntd() == 2`
- one hot vector iteration naturally produces and consumes **2 words** of horizontal partials

But the storage unit was still **1 word = 24 bytes**.

That means one vector tile was really two adjacent 24-byte records, i.e. a 48-byte AoS chunk. The code was correct, but the memory shape was awkward:

- producer wrote with `svst3_u64`
- consumer read with `svld3_u64`
- hot structured loads/stores frequently spanned cache-line boundaries

So the machine’s natural work unit was 2 words, but the layout was still indexed as 1-word records.

## 3. New idea: store the vector tile directly

Instead of storing one partial word at a time, the new variant stores the 2-word tile directly:

```cpp
struct alignas(16) HorizontalPartialPair2 {
    uint64_t b0[2];
    uint64_t b1[2];
    uint64_t b2[2];
};
```

This still stores the same useful 48 bytes, but in a form that matches the real vector tile:

- 16 bytes for `b0`
- 16 bytes for `b1`
- 16 bytes for `b2`

So the producer now writes:

```cpp
svst1_u64(pg, pair.b0, partial_b0);
svst1_u64(pg, pair.b1, partial_b1);
svst1_u64(pg, pair.b2, partial_b2);
```

and the consumer reads:

```cpp
svuint64_t partial_b0 = svld1_u64(pg, pair.b0);
svuint64_t partial_b1 = svld1_u64(pg, pair.b1);
svuint64_t partial_b2 = svld1_u64(pg, pair.b2);
```

Important scope note:

- algorithm unchanged
- rolling update unchanged
- result-generation unchanged
- only the horizontal partial layout and its producer/consumer access pattern changed

## 4. Why this layout is better

The improvement is not "fewer useful bytes". It is "better-shaped useful bytes".

The old AoS layout moved one 48-byte vector tile through structured `ld3/st3`.

The new pair2 layout moves the same tile as:

- one aligned 16-byte `b0` load/store
- one aligned 16-byte `b1` load/store
- one aligned 16-byte `b2` load/store

That matches the fixed remote VL better:

- each vector register is exactly 2 x `uint64_t`
- each partial plane for a 2-word tile is exactly one `svuint64_t`
- each hot memory op is a plain aligned `ld1d/st1d`

## 5. Codegen check

Remote assembly confirmed the structural change.

Baseline `aligned_hot_loop` still contained hot:

```asm
st3d ...
ld3d ...
```

The new pair2 variant removed those structured partial moves from this path and replaced them with plain:

```asm
st1d ...
ld1d ...
```

That was the decisive codegen signal: the layout rewrite survived optimization and changed the actual hot memory instructions.

## 6. Correctness result

Remote validation command:

```bash
python3 -u run_tests.py \
  reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2.cpp \
  512 \
  reference/build_spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2.sh
```

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

## 7. Benchmark result against the current aligned-hot-loop baseline

Large-grid test:

```bash
./<binary> test_grids/public_1_random_low_32768.bin /dev/null 500
```

Remote comparison from the same session:

| Variant | Program time | Wall clock |
| --- | ---: | ---: |
| `aligned_hot_loop` baseline | `7933.918 ms` | `13.49 s` |
| `aligned_hot_loop_pair2` experiment | `7519.843 ms` | `13.08 s` |

Delta vs baseline:

- `-414.075 ms`
- `5.22%` speedup by program-reported time
- `-0.41 s` wall-clock improvement, about `3.04%`

## 8. Why this optimization worked

This win came from matching the horizontal partial layout to the machine’s real vector tile.

The earlier design was conceptually elegant, but it stored 2-word SVE work units as two separate 24-byte AoS records. The new design stores exactly the 2-word tile that the remote SVE loop already produces and consumes.

That is why the result was materially better:

- no algorithmic change
- no extra fusion
- no larger boolean logic
- just a cleaner memory layout and simpler hot vector load/store pattern

## Takeaway

This experiment worked because it followed the actual hardware granularity instead of the original scalar record shape.

In short: the pair2 layout rewrote horizontal partial traffic so the hot path now moves the same information with aligned `ld1/st1` plane accesses instead of awkward `ld3/st3` AoS traffic, and that produced a real `~5.2%` speedup over the current `aligned_hot_loop` baseline.
