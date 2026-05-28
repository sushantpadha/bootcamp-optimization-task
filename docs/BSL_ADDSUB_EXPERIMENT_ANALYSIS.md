# BSL Add/Sub Experiment Analysis

This note explains the `bsl`-based add/sub rewrite that produced a real win over the current `always_inline` baseline.

Relevant files:

- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_always_inline.cpp`
- `reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub.cpp`

## 1. Starting point

The current baseline was already strong:

- SVE2 hot loop
- direct add/sub update
- direct range masks
- forced inline on `compute_horizontal_partial_row`

The remaining question was narrow: can the hot SVE boolean logic be expressed in a form that `g++-14` lowers better on Neoverse-V2?

This was worth checking because the hot path still spends time in:

- `carry_save_add_u64_sve(...)`
- `subtract_horizontal_partial_from_total_u64_sve(...)`

## 2. First check: what does the compiler already do?

Before changing source, I checked whether GCC was already collapsing the current full-adder to the best SVE2 instruction sequence.

Current carry logic:

```cpp
sum = a ^ b ^ c;
carry = (a & b) | (a & c) | (b & c);
```

Standalone remote compile with `g++-14 -O3 -mcpu=neoverse-v2 -S` showed:

```asm
eor3 ...
orr  ...
and  ...
and  ...
orr  ...
```

That meant:

- GCC already used `eor3` for the 3-input XOR
- GCC did **not** reduce the carry path to a single ternary boolean instruction

So there was still room for an ISA-shaped rewrite.

## 3. Next idea: keep the math, change the expression shape

The goal was not to invent a new adder. The goal was to express the exact same boolean function in a way that could lower to `bsl`.

Equivalent full-adder carry:

```text
carry = c ? (a | b) : (a & b)
```

Equivalent full-subtractor borrow stage:

```text
borrow_out = borrow_in ? (~total | partial) : (~total & partial)
```

If those forms lowered to `bsl` in the hot loop, they could replace multiple `and`/`orr` instructions with a smaller ternary select sequence.

## 4. First implementation failed

The first `svbsl_u64(...)` rewrite compiled cleanly but failed correctness on the remote machine:

```text
Summary: 0 passed, 5 failed, 0 skipped
```

This was useful because it showed the idea was not disproven yet; the likely problem was intrinsic semantics, not the algebra itself.

## 5. Then I stopped guessing and measured `svbsl_u64` directly

I wrote a tiny remote truth-table program to test `svbsl_u64` bit semantics.

Evidence snippet:

```text
m=0 x=1 y=0 -> 1
m=0 x=0 y=1 -> 0
m=1 x=0 y=1 -> 1
```

The important conclusion was:

```text
svbsl_u64(m, x, y) = y ? m : x   // per bit
```

That was the correction point. My first implementation had assumed normal "mask selects first/second value" semantics on the first operand.

## 6. Corrected implementation

After fixing operand order, the new variant changed only the SVE helper formulas.

New carry form:

```cpp
sum = sveor_x(pg, sveor_x(pg, a, b), c);
carry = svbsl_u64(svorr_x(pg, a, b), svand_x(pg, a, b), c);
```

New borrow stage form:

```cpp
const svuint64_t not_total_b1 = sveor_x(pg, total_b1, all_ones);
next_borrow = svbsl_u64(
    svorr_x(pg, not_total_b1, partial_b1),
    svand_x(pg, not_total_b1, partial_b1),
    borrow);
```

Important scope note:

- algorithm unchanged
- data layout unchanged
- rolling update unchanged
- only the boolean expression shape changed

## 7. Codegen check after the fix

I recompiled the real experiment file on the remote with `-fverbose-asm -S` and checked that the rewritten helpers actually lowered to `bsl` in the hot function.

Evidence snippet:

```asm
bsl z25.d, z25.d, z1.d, z28.d
bsl z31.d, z31.d, z0.d, z29.d
```

That was the decisive codegen signal: the rewrite survived optimization and produced the intended SVE2 instruction in the hot path.

## 8. Correctness result

Remote validation command:

```bash
python3 -u run_tests.py \
  reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub.cpp \
  512 \
  reference/build_spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2_direct_ranges_bsl_addsub.sh
```

Result:

```text
Summary: 5 passed, 0 failed, 0 skipped
```

## 9. Benchmark result against the current baseline

Large-grid test:

```bash
./<binary> test_grids/public_1_random_low_32768.bin /dev/null 100
```

Clean comparison from the same remote session:

| Variant | Program time | Wall clock |
| --- | ---: | ---: |
| `direct_ranges_always_inline` baseline | `1636.797 ms` | `7.19 s` |
| `direct_ranges_bsl_addsub` experiment | `1569.069 ms` | `7.12 s` |

Delta vs baseline:

- `-67.728 ms`
- `4.14%` speedup by program-reported time
- `-0.07 s` wall-clock improvement, about `0.97%`

Repeat runs showed the same pattern:

- baseline: `1640.212 ms`, `1639.031 ms`
- experiment: `1567.997 ms`, `1567.924 ms`

So this was not a one-off noisy sample.

## 10. Why this optimization worked

The win did **not** come from changing the algorithm. It came from expressing the same boolean functions in a form that GCC 14 could lower to `bsl` inside the real hot loop on Neoverse-V2.

That is why this experiment succeeded where several "smarter adder" ideas failed:

- no structural code growth
- no extra fusion
- no larger working set
- just a tighter instruction selection for logic that already dominates the hot path

## Takeaway

This was a good example of the right optimization order:

1. Start from measured hotspot evidence.
2. Check existing compiler output before rewriting source.
3. Use a tiny isolated semantic test when an intrinsic is unclear.
4. Verify codegen again after the fix.
5. Require both remote correctness and remote benchmarking before calling it a win.

In short: the `bsl` rewrite was a small source change, but it matched the target ISA better and produced a real `~4.1%` speedup over the current baseline.
