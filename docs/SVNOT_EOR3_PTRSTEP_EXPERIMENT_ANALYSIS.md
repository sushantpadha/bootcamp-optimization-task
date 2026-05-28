# Svnot/Eor3 Ptrstep Experiment Analysis

This note explains the hot-SVE instruction-shape cleanup that improved the current `ptrstep` pure-SVE baseline.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep_svnot_eor3.cpp`

## 1. Why this comparison matters

This is an apple-to-apple comparison against the current best `ptrstep` pure-SVE version.

The new candidate keeps:

- the same algorithm
- the same `pair2` partial layout
- the same exact-two-row blocked hot schedule
- the same pointer-stepped aligned SVE sweep
- the same pure-SVE emit `bsl` rewrite
- the same threading model
- the same data representation

So the intended optimization here is narrower:

- keep the hot SVE helper structure exactly as it is
- change only a few boolean instruction shapes inside that hot helper path
- make the compiler see direct SVE idioms for bitwise NOT and 3-input XOR

That makes this the same style of optimization as the earlier `bsl` win:

- no algorithm change
- no schedule change
- no data-layout change
- just a cleaner expression of the same hot boolean work

## 2. Starting point

The `ptrstep` baseline was already strong:

- pointer-stepped aligned pair2 SVE sweep
- pure-SVE exact-count add/sub helpers
- pure-SVE `bsl`-shaped emit logic
- in-place next-adult arrangement

But even after those larger wins, the hot SVE helper block still contained a few source shapes that were more indirect than necessary.

In particular:

- bitwise NOT was still expressed as XOR with all-ones
- some hot 3-input XOR chains were still expressed as nested 2-input XORs

Those are logically fine, but they can cost the compiler freedom in the hottest path.

## 3. The remaining inefficiency

The old helper code still had shapes like this:

```cpp
ALWAYS_INLINE svuint64_t invert_word(svuint64_t word) const
{
    return xor_words(word, svdup_u64(~0ULL));
}

ALWAYS_INLINE void carry_save_add(
    svuint64_t a,
    svuint64_t b,
    svuint64_t c,
    svuint64_t& sum,
    svuint64_t& carry) const
{
    sum = xor_words(xor_words(a, b), c);
    carry = svbsl_u64(or_words(a, b), and_words(a, b), c);
}
```

And in the subtract path:

```cpp
count_b1 = xor_words(xor_words(count_b1, partial_b1), borrow);
count_b2 = xor_words(xor_words(count_b2, partial_b2), borrow);
```

And in emit:

```cpp
const svuint64_t all_ones = svdup_u64(~0ULL);
const svuint64_t empty_word = xor_words(occupied_word, all_ones);
```

None of that changes the truth table.

The issue is just instruction shape:

- `~x` was written as `x ^ -1`
- `a ^ b ^ c` was written as `(a ^ b) ^ c`

That can force extra all-ones materialization and can make a hot 3-input XOR chain look like two separate operations instead of one combined boolean op.

## 4. Strategy

The strategy was to port only the useful instruction-shape ideas from `reference/ref_impl.cpp` into the current best `ptrstep` kernel, without cargo-culting the full helper layout.

Specifically:

1. use `svnot_x(...)` for hot packed-word NOTs
2. use `sveor3(...)` for hot 3-input XOR chains
3. leave the rest of the helper structure alone

Important scope note:

- no algorithm change
- no boolean-logic rewrite beyond equivalent instruction-shape cleanup
- no scalar-path change
- no NEON-path change
- no hot-loop scheduling change

## 5. The key rewrite

The new candidate added a tiny helper for explicit 3-input XOR:

```cpp
ALWAYS_INLINE svuint64_t xor3_words(
    svuint64_t first,
    svuint64_t second,
    svuint64_t third) const
{
    return sveor3(first, second, third);
}
```

Then it changed `invert_word(...)` to the direct SVE NOT form:

```cpp
ALWAYS_INLINE svuint64_t invert_word(svuint64_t word) const
{
    return svnot_x(predicate(), word);
}
```

Then it changed the hot sum path in `carry_save_add(...)`:

```cpp
sum = xor3_words(a, b, c);
```

instead of:

```cpp
sum = xor_words(xor_words(a, b), c);
```

The subtract helper also changed the two hot total updates:

```cpp
count_b1 = xor3_words(count_b1, partial_b1, borrow);
count_b2 = xor3_words(count_b2, partial_b2, borrow);
```

instead of:

```cpp
count_b1 = xor_words(xor_words(count_b1, partial_b1), borrow);
count_b2 = xor_words(xor_words(count_b2, partial_b2), borrow);
```

And the emit path now gets empty cells through the same direct NOT helper:

```cpp
const svuint64_t empty_word = invert_word(occupied_word);
```

instead of:

```cpp
const svuint64_t all_ones = svdup_u64(~0ULL);
const svuint64_t empty_word = xor_words(occupied_word, all_ones);
```

## 6. Which hot helpers changed

The optimization intentionally touched only the packed-tile SVE helper block:

- `invert_word(...)`
- `carry_save_add(...)`
- `subtract_partial_from_total(...)`
- `write_next_state_from_counts(...)`

The rest of the kernel stayed structurally the same.

That matters because this experiment is trying to answer a narrow question:

- does cleaner direct SVE boolean shape help inside the already-good `ptrstep` kernel?

## 7. Why this works

The key idea is that the compiler is better at using good instructions when the source already names the operation directly.

`svnot_x(...)` helps because:

- it expresses bitwise NOT directly
- it avoids explicit all-ones vector materialization in the source
- it removes some pressure from the hot boolean path

`sveor3(...)` helps because:

- it expresses a 3-input XOR directly
- it can replace a two-step XOR chain with one boolean instruction shape
- it shortens the hot dependency chain for the sum update

This is especially relevant in:

- `carry_save_add(...)`, which is on the packed count-accumulation path
- `subtract_partial_from_total(...)`, which is on the rolling-window recycle path

Those helpers sit directly inside the hottest exact-count update network, so even small instruction-shape cleanups can matter once the larger structural problems have already been removed.

## 8. Why the change is still exactly correct

Every rewrite is a strict boolean identity:

- `~x` is identical to `x ^ all_ones`
- `a ^ b ^ c` is identical to `(a ^ b) ^ c`

So the new candidate does not change:

- count semantics
- borrow semantics
- next-state truth tables
- boundary handling

It only changes how those same boolean operations are presented to the compiler.

That is why this kind of optimization is safe to apply in a mature hot path:

- same math
- same schedule
- different source shape

## 9. Codegen check

I checked generated assembly on the remote machine.

The candidate did emit `eor3` in the generated hot code, for example:

```asm
eor3    z19.d, z19.d, z28.d, z27.d
eor3    z17.d, z17.d, z26.d, z19.d
eor3    z27.d, z27.d, z3.d, z2.d
eor3    z24.d, z24.d, z1.d, z27.d
```

I also compared the old explicit all-ones materialization pattern in the `.s` files.

Baseline:

```asm
mov     z27.b, #-1
mov     z31.b, #-1
mov     z23.b, #-1
mov     z18.b, #-1
```

Candidate:

```text
no matching `mov z?.b, #-1` sites in the generated .s
```

So while GCC did not leave a literal `not` mnemonic in the final assembly dump, the candidate did remove the visible all-ones vector materializations that the old XOR-with-all-ones source shape encouraged.

That is exactly the direction we wanted.

## 10. Correctness result

Remote validation command:

```bash
python3 -u run_tests.py \
  reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep_svnot_eor3.cpp \
  512 \
  reference/build_spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub_aligned_hot_loop_pair2_fused_incoming_exact2row_refactored_inplace_adults_sve_emit_bsl_ptrstep_svnot_eor3.sh
```

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

I also ran an exact remote comparison against the baseline on:

- `test_grids/public_1_random_low_8192.bin`
- `500` generations

Result:

```text
exact_match=true
```

The baseline and candidate output files had identical SHA-256 hashes.

## 11. Benchmark result against the current best

Comparison target:

- baseline: `..._sve_emit_bsl_ptrstep.cpp`
- candidate: `..._sve_emit_bsl_ptrstep_svnot_eor3.cpp`

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

- baseline: `12438.438 ms`, `12458.800 ms`, `12610.717 ms`
- candidate: `11824.221 ms`, `11825.645 ms`, `11979.078 ms`

Averages:

- baseline: `12502.652 ms`
- candidate: `11876.315 ms`

Delta vs baseline:

- `-626.337 ms`
- `5.010%` speedup

Wall-clock times:

- baseline: `18.080 s`, `18.110 s`, `18.280 s`
- candidate: `17.480 s`, `17.480 s`, `17.680 s`

Wall-clock averages:

- baseline: `18.157 s`
- candidate: `17.547 s`

Wall-clock delta vs baseline:

- `-0.610 s`
- `3.360%` speedup

So this was not just a printed-kernel-timer win.
With output-file noise removed from the measured runs, the candidate also won on end-to-end wall time.

## 12. Why this optimization was worth doing

This pass is a good example of what becomes valuable late in an optimization sequence.

Earlier passes removed larger structural issues:

- mixed-ISA bridge overhead
- less favorable emit logic shape
- more index-heavy aligned loop structure

Once those are gone, smaller boolean instruction-shape details can become visible.

This pass won because it cleaned up exactly those remaining details:

- direct NOT instead of XOR-with-all-ones
- direct 3-input XOR instead of nested XOR chains

inside the hottest exact-count SVE helper path.

## Takeaway

The best move here was not to redesign the kernel again.

The best move was:

1. keep the current best `ptrstep` structure
2. identify a few remaining indirect SVE boolean shapes
3. rewrite those shapes directly as `svnot_x(...)` and `sveor3(...)`

That preserved correctness, kept the patch small, and produced a real speedup over the current best version.
