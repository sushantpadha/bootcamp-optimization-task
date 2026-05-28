# SVE Emit BSL vs Previous SVE Analysis

This note compares the new pure-SVE emit rewrite directly against the previous pure-SVE `inplace_adults` version.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl.cpp`

## 1. Why this comparison matters

The mixed `neon_emit` experiment was useful, but it is not the cleanest way to understand the final SVE emit rewrite.

The real apple-to-apple comparison is:

- previous version: pure SVE hot backend
- new version: still pure SVE hot backend

That isolates the actual optimization:

- same algorithm
- same pair2 layout
- same exact-two-row schedule
- same threading model
- same hot-row pipeline
- only the packed-two-word SVE emit boolean network changed

## 2. Starting point

The previous pure-SVE version already had:

- SVE `carry_save_add(...)`
- SVE add/sub update
- SVE packed-two-word hot loop
- no mixed-ISA bridge

So this was not a “remove NEON overhead” win in this comparison.

This was specifically a code-shape win inside the SVE emit path.

## 3. What changed in source

The previous pure-SVE emit logic expressed the two key masks as nested `and`/`or` trees.

Previous `adult_5_to_10`:

```cpp
const svuint64_t adult_5_to_10 =
    and_words(
        not_b4,
        or_words(
            and_words(not_b3, and_words(count_b2, b1_or_b0)),
            and_words(
                count_b3,
                and_words(not_b2, xor_words(b1_and_b0, all_ones)))));
```

Previous `egg_3_to_5`:

```cpp
const svuint64_t egg_3_to_5 =
    and_words(
        and_words(not_b4, not_b3),
        or_words(
            and_words(not_b2, b1_and_b0),
            and_words(count_b2, not_b1)));
```

The new version rewrote those as explicit selects:

```cpp
const svuint64_t adult_when_b3_clear = and_words(count_b2, b1_or_b0);
const svuint64_t adult_when_b3_set = and_words(not_b2, not_b1_and_b0);
const svuint64_t adult_5_to_10 =
    and_words(not_b4, svbsl_u64(adult_when_b3_set, adult_when_b3_clear, count_b3));

const svuint64_t egg_when_b2_clear = b1_and_b0;
const svuint64_t egg_when_b2_set = not_b1;
const svuint64_t egg_3_to_5 = and_words(
    and_words(not_b4, not_b3),
    svbsl_u64(egg_when_b2_set, egg_when_b2_clear, count_b2));
```

This keeps the same logic, but changes how that logic is presented to GCC.

## 4. Why that source rewrite matters

The old source asks the compiler to recover a ternary choice from a deeper tree of boolean ops:

- choose one expression when `count_b3` is clear
- choose another when `count_b3` is set
- similarly for `count_b2`

The new source states that choice directly.

That matters because `bsl` is exactly the SVE instruction shape for:

```text
mask ? value_if_set : value_if_clear
```

So the new source does a better job of telling the compiler what the hardware instruction should be.

## 5. Assembly evidence

I checked both versions on the remote machine with `-fverbose-asm -S`.

Both versions do contain `bsl` somewhere in the hot code, because the packed backend already used `bsl` in the add/sub helpers.

But the important point is that the new emit rewrite also created a cleaner `bsl`-shaped boolean path in the hot function.

Example lines from the old version:

```asm
bsl z25.d, z25.d, z29.d, z23.d
bsl z23.d, z23.d, z24.d, z25.d
```

Example lines from the new version:

```asm
bsl z29.d, z29.d, z31.d, z28.d
bsl z29.d, z29.d, z14.d, z19.d
```

The exact register names are not important. What matters is:

- the new emit logic survived optimization
- it remained expressed as direct `bsl` selects in the real generated code

## 6. Hot function shape

I also compared the generated hot function size.

`run_one_generation_hot_rows(...)`:

- previous pure-SVE version: `0x1ae4`
- new pure-SVE emit rewrite: `0x1a8c`

So the new version made the hot function smaller by `0x58` bytes.

Stack frame size did **not** change in this apples-to-apples comparison:

- previous pure-SVE version: `384` bytes
- new pure-SVE emit rewrite: `384` bytes

That matters for interpretation:

- this win is not coming from a smaller stack frame
- it is coming from a tighter hot instruction stream

## 7. What likely got better

The best explanation is:

1. the emit boolean network now maps more directly onto `bsl`
2. the compiler needs fewer intermediate boolean combinations
3. the hot function becomes slightly smaller
4. the front end and scheduler have a cleaner instruction stream to work with

This matches the measured result better than any alternative explanation.

Nothing else changed that could plausibly account for the speedup:

- no layout change
- no threading change
- no range change
- no algorithm change
- no extra vector width

## 8. Correctness result

Remote validation command:

```bash
python3 -u run_tests.py \
  reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl.cpp \
  512 \
  reference/build_spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl.sh
```

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

## 9. Benchmark result against the previous pure-SVE version

Comparison target:

- old: `...refactored_inplace_adults.cpp`
- new: `...refactored_inplace_adults_sve_emit_bsl.cpp`

Large-grid test:

```bash
./<binary> test_grids/public_1_random_low_32768.bin /dev/null 500
```

Setup:

- remote host: `jump-bootcamp`
- compiler: `g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra -pthread`
- warmup: `50` generations each
- measured: `2` paired runs each

Program-reported times:

- old: `6634.339 ms`, `6606.561 ms`
- new: `6384.027 ms`, `6243.821 ms`

Averages:

- old: `6620.450 ms`
- new: `6313.924 ms`

Delta vs old:

- `-306.526 ms`
- `4.630%` speedup

So the new SVE emit rewrite is not just better than the mixed-ISA detour. It is also clearly better than the immediately previous pure-SVE version.

## 10. Why the new version is faster

The short answer is:

- same SVE backend
- same algorithm
- better boolean expression shape
- cleaner lowering to `bsl`
- smaller hot function

More concretely:

- the old version forced GCC to reconstruct ternary logic from OR/AND trees
- the new version hands GCC the ternary choice directly
- Neoverse-V2 has a native instruction shape for that choice: `bsl`
- the compiler can therefore emit a tighter sequence in the hottest part of the kernel

That is why this win is believable:

- it is rooted in the exact hotspot logic
- the source change is narrow
- the assembly changed in the expected direction
- the hot function got smaller
- correctness stayed intact
- the remote benchmark moved by a meaningful amount

## Takeaway

This optimization worked because it was precise.

It did **not** add a new idea to the algorithm.
It simply expressed the existing SVE emit logic in the form the target ISA already wants.

In short: the new pure-SVE version is faster than the previous pure-SVE version because the emit boolean network was rewritten from nested boolean trees into direct `bsl`-shaped selects, giving GCC a smaller and better-matched hot loop and producing a measured `~4.63%` speedup on the remote target.
