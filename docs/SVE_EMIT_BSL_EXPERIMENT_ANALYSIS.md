# SVE Emit BSL Experiment Analysis

This note explains the pure-SVE emit rewrite that beat the mixed SVE/NEON `neon_emit` variant.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_neon_emit.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl.cpp`

## 1. Starting point

The current best variant before this pass was the mixed-ISA `neon_emit` file:

- SVE for the packed two-word hot backend
- NEON only inside `emit_next_state(...)`
- same algorithm
- same ring-buffer layout
- same exact-two-row schedule

That version was faster than the prior pure-SVE refactor, but it still paid a bridge cost inside the hot loop.

## 2. The remaining problem

In the `neon_emit` version, `Pair2SVEHotWordBackend::emit_next_state(...)` did this for every packed two-word tile:

1. receive `count_b0..count_b4` in SVE registers
2. spill all five count planes to temporary stack arrays
3. reload those arrays as NEON vectors
4. run the NEON boolean next-state logic
5. store the result

So the likely remaining waste was:

- extra SVE stores
- extra NEON reloads
- extra stack traffic
- larger hot function
- larger stack frame

The goal was to remove that bridge completely without giving up the good instruction shape that made the NEON emit logic fast.

## 3. Strategy

The strategy was:

- keep the packed two-word hot backend fully on SVE again
- rewrite only the SVE emit boolean network
- express the key logic as explicit ternary selects so GCC could lower it to `bsl`

Important scope note:

- algorithm unchanged
- data layout unchanged
- rolling update unchanged
- no new vector width assumptions
- only `emit_next_state(...)` and its SVE helper logic changed

## 4. The key rewrite

The previous pure-SVE emit form used nested `and`/`or` trees:

```cpp
const svuint64_t adult_5_to_10 =
    and_words(
        not_b4,
        or_words(
            and_words(not_b3, and_words(count_b2, b1_or_b0)),
            and_words(
                count_b3,
                and_words(not_b2, xor_words(b1_and_b0, all_ones)))));
const svuint64_t egg_3_to_5 =
    and_words(
        and_words(not_b4, not_b3),
        or_words(
            and_words(not_b2, b1_and_b0),
            and_words(count_b2, not_b1)));
```

The new form rewrote those as two explicit selects:

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

This kept the exact same truth table, but gave the compiler a much better chance to use `bsl` directly instead of rebuilding the logic from smaller boolean ops.

## 5. What was removed

The new candidate also removed the entire SVE-to-NEON bridge from the packed backend:

- no `count_b0_words[2]`
- no `count_b1_words[2]`
- no `count_b2_words[2]`
- no `count_b3_words[2]`
- no `count_b4_words[2]`
- no spill to stack just to enter emit logic
- no NEON reloads inside `emit_next_state(...)`

So this was not only a boolean rewrite. It also restored a single-ISA hot path.

## 6. Codegen check

I compiled the candidate remotely with:

```bash
g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra -pthread -fverbose-asm -S ...
```

Two things mattered:

1. `bsl` had to show up in the generated hot code.
2. the old bridge temp symbols had to disappear.

Evidence from the remote assembly:

```asm
bsl z29.d, z29.d, z31.d, z28.d
bsl z29.d, z29.d, z14.d, z19.d
```

And the grep for the old bridge temp names returned nothing:

```text
TEMP_BRIDGE_LINES
```

That confirmed the source-level bridge was gone and the compiler did lower the emit logic to `bsl` in the real generated code.

## 7. Hot function size and stack frame

I also compared the generated hot function shape with debug-symbol builds.

`run_one_generation_hot_rows(...)` size:

- `neon_emit` best before this pass: `0x1bfc`
- new `sve_emit_bsl` candidate: `0x1a8c`

Stack frame size from the prologue:

- `neon_emit` best before this pass: `480` bytes
- new `sve_emit_bsl` candidate: `384` bytes

So the new version produced:

- a smaller hot function
- a smaller stack frame
- less obvious local spill space

That matches the intended effect of removing the mixed-ISA bridge.

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

## 9. Benchmark result against the previous best

Comparison target:

- previous best: `...refactored_inplace_adults_neon_emit.cpp`
- candidate: `...refactored_inplace_adults_sve_emit_bsl.cpp`

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

- previous best: `6432.533 ms`, `6429.898 ms`
- candidate: `6244.182 ms`, `6246.603 ms`

Averages:

- previous best: `6431.216 ms`
- candidate: `6245.392 ms`

Delta vs previous best:

- `-185.823 ms`
- `2.889%` speedup

So this was a real win over the mixed-ISA version, not just over an older baseline.

## 10. Why this optimization worked

This pass won for two reasons at once:

1. It removed the SVE-to-NEON bridge from the hot path.
2. It rewrote the SVE emit logic into a form that GCC could lower more cleanly for Neoverse-V2.

That combination is important.

The earlier `neon_emit` experiment showed that the emit boolean network mattered.
The later hybrid experiments showed that expanding NEON deeper into the packed backend did **not** help.
This final pass kept the good part:

- better boolean code shape

while removing the bad part:

- mixed-ISA bridge overhead

## Takeaway

The best answer was not "use more NEON".

The best answer was:

1. identify that the bridge itself had become overhead
2. make the hot path single-ISA again
3. keep the good boolean-expression shaping idea
4. verify the compiler actually emitted `bsl`
5. confirm the result with remote correctness and remote timing

In short: the winning change was a pure-SVE emit rewrite that removed the mixed-ISA bridge and let GCC generate a smaller, cleaner hot loop, producing a measured `~2.9%` speedup over the previous best variant.
