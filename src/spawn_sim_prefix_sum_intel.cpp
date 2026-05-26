// spawn_sim_x86.cpp — Monster Spawning Grid, x86-64 / AVX2 edition.
//
// Identical algorithm to the ARM NEON version:
//   Pass 1 (parallel): build padded adult-bit row → horizontal 5-wide sliding sum → row_sums[y]
//   Pass 2 (parallel): vertical sum of 5 row_sum rows → A, subtract centre adult, apply rules
//   std::barrier synchronises the two passes across NTHREADS worker threads.
//
// SIMD: Intel AVX2 via <immintrin.h>
//   Width: 256-bit → 32 × uint8_t per register (vs 16 on NEON).
//   Compile with: -mavx2  (or -march=native / -march=x86-64-v3)
//
// UNSIGNED RANGE COMPARISON TRICK
// ────────────────────────────────
// AVX2 has no unsigned byte comparison (vcmpge_epu8 etc.).
// _mm256_cmpgt_epi8 is *signed*.  For our values A ∈ [0, 24] and thresholds
// 3, 4, 5, 9 (all < 128), signed == unsigned — but only for this problem.
// A cleaner general solution uses the "subtract-and-min" idiom:
//
//   3 ≤ A ≤ 5  ⟺  (uint8)(A − 3) ≤ 2
//
// Check "(uint8)v ≤ k" with: _mm256_cmpeq_epi8(_mm256_min_epu8(v, K), v)
//   min_epu8(v, K) == v  iff  v ≤ K  (unsigned).
// This avoids any signed/unsigned confusion and works for all uint8 values.
//
// Analogously: 4 ≤ A ≤ 9  ⟺  (uint8)(A − 4) ≤ 5.
//
// MEMORY / THREADING: same as ARM version.
//   3 N² bytes total (grid_a + grid_b + row_sums), 64-byte aligned.
//   8 threads, one per vCPU; main thread runs as thread 0.

#include <immintrin.h>   // AVX2 — replaces <arm_neon.h>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static constexpr int NTHREADS = 8;

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 1 helpers
// ─────────────────────────────────────────────────────────────────────────────

// Populate ap[0..N+3] with adult bits (0 or 1) + 2-cell toroidal wrap on each end.
// ap layout: [ adult(gr[N-2]) | adult(gr[N-1]) | adult(gr[0..N-1]) | adult(gr[0]) | adult(gr[1]) ]
static inline void build_adult_padded(const uint8_t* __restrict__ gr,
                                       uint8_t* __restrict__       ap,
                                       int N)
{
    ap[0]   = (gr[N - 2] == 3u) ? 1u : 0u;
    ap[1]   = (gr[N - 1] == 3u) ? 1u : 0u;
    ap[N+2] = (gr[0]     == 3u) ? 1u : 0u;
    ap[N+3] = (gr[1]     == 3u) ? 1u : 0u;

    const __m256i v3 = _mm256_set1_epi8(3);
    const __m256i v1 = _mm256_set1_epi8(1);
    int x = 0;
    for (; x + 32 <= N; x += 32) {
        __m256i cells = _mm256_loadu_si256((const __m256i*)(gr + x));
        // adult bit: 0xFF where cell==3, AND 0x01 → 0 or 1
        __m256i adult = _mm256_and_si256(_mm256_cmpeq_epi8(cells, v3), v1);
        _mm256_storeu_si256((__m256i*)(ap + 2 + x), adult);
    }
    for (; x < N; ++x)
        ap[2 + x] = (gr[x] == 3u) ? 1u : 0u;
}

// Horizontal 5-wide sliding window sum.
// rs[x] = ap[x]+ap[x+1]+ap[x+2]+ap[x+3]+ap[x+4]  for x ∈ [0, N).
// ap is N+4 bytes (padded); result fits in uint8_t (max 5).
static inline void sliding_sum5(const uint8_t* __restrict__ ap,
                                  uint8_t* __restrict__       rs,
                                  int N)
{
    int x = 0;
    for (; x + 32 <= N; x += 32) {
        __m256i s =
            _mm256_add_epi8(
                _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(ap + x)),
                                _mm256_loadu_si256((const __m256i*)(ap + x + 1))),
                _mm256_add_epi8(
                    _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(ap + x + 2)),
                                    _mm256_loadu_si256((const __m256i*)(ap + x + 3))),
                    _mm256_loadu_si256((const __m256i*)(ap + x + 4))));
        _mm256_storeu_si256((__m256i*)(rs + x), s);
    }
    for (; x < N; ++x)
        rs[x] = ap[x] + ap[x+1] + ap[x+2] + ap[x+3] + ap[x+4];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 2: vertical sum → A, subtract centre adult, apply transition rules
// ─────────────────────────────────────────────────────────────────────────────

// Unsigned range helper: returns 0xFF lanes where (uint8)(v - base) ≤ width.
// Equivalent to: base ≤ v ≤ base+width  (unsigned, wrapping subtraction).
//
// Proof:
//   v < base           → (uint8)(v-base) is a large value (wraps), > width → 0x00 ✓
//   base ≤ v ≤ base+width → (uint8)(v-base) ∈ [0,width] ≤ width            → 0xFF ✓
//   v > base+width     → (uint8)(v-base) > width                             → 0x00 ✓
//
// The check "(uint8)d ≤ w" is done as:  min_epu8(d, W) == d  (d ≤ W iff min clamps to d)
static inline __m256i in_range_u8(const __m256i v,
                                   const __m256i base,
                                   const __m256i width)
{
    __m256i d = _mm256_sub_epi8(v, base);                 // wrapping uint8 subtract
    return _mm256_cmpeq_epi8(_mm256_min_epu8(d, width), d); // 0xFF where d ≤ width
}

static inline void apply_rules(
    const uint8_t* __restrict__ cur_row,
    const uint8_t* __restrict__ rs0,    // row_sums[y-2]
    const uint8_t* __restrict__ rs1,    // row_sums[y-1]
    const uint8_t* __restrict__ rs2,    // row_sums[y  ]
    const uint8_t* __restrict__ rs3,    // row_sums[y+1]
    const uint8_t* __restrict__ rs4,    // row_sums[y+2]
    uint8_t* __restrict__       nxt_row,
    int N)
{
    const __m256i k0 = _mm256_setzero_si256();     // EMPTY value
    const __m256i k1 = _mm256_set1_epi8(1);        // EGG value      / adult mask clamp
    const __m256i k2 = _mm256_set1_epi8(2);        // JUVENILE value
    const __m256i k3 = _mm256_set1_epi8(3);        // ADULT value

    // Range constants for in_range_u8():
    //   3 ≤ A ≤ 5  ↔  in_range(A, base=3, width=2)   [5-3=2]
    //   4 ≤ A ≤ 9  ↔  in_range(A, base=4, width=5)   [9-4=5]
    const __m256i base3 = _mm256_set1_epi8(3), width_3_5 = _mm256_set1_epi8(2);
    const __m256i base4 = _mm256_set1_epi8(4), width_4_9 = _mm256_set1_epi8(5);

    int x = 0;
    for (; x + 32 <= N; x += 32) {

        // ── Step 1: sum five row_sum rows (vertical neighbourhood) ──
        __m256i A =
            _mm256_add_epi8(
                _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(rs0 + x)),
                                _mm256_loadu_si256((const __m256i*)(rs1 + x))),
                _mm256_add_epi8(
                    _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(rs2 + x)),
                                    _mm256_loadu_si256((const __m256i*)(rs3 + x))),
                    _mm256_loadu_si256((const __m256i*)(rs4 + x))));

        // ── Step 2: subtract centre adult (A includes the centre cell) ──
        __m256i c = _mm256_loadu_si256((const __m256i*)(cur_row + x));
        A = _mm256_sub_epi8(A, _mm256_and_si256(_mm256_cmpeq_epi8(c, k3), k1));

        // ── Step 3: one-hot state masks (0xFF or 0x00) ──
        __m256i is_empty    = _mm256_cmpeq_epi8(c, k0);
        __m256i is_egg      = _mm256_cmpeq_epi8(c, k1);
        __m256i is_juvenile = _mm256_cmpeq_epi8(c, k2);
        __m256i is_adult    = _mm256_cmpeq_epi8(c, k3);

        // ── Step 4: compute next-state contributions ──

        // EMPTY → EGG(1)  iff  3 ≤ A ≤ 5
        __m256i spawn = in_range_u8(A, base3, width_3_5);
        __m256i n0    = _mm256_and_si256(_mm256_and_si256(is_empty, spawn), k1);

        // EGG → JUVENILE(2)  always
        __m256i n1    = _mm256_and_si256(is_egg, k2);

        // JUVENILE → ADULT(3)  always
        __m256i n2    = _mm256_and_si256(is_juvenile, k3);

        // ADULT → ADULT(3)  iff  4 ≤ A ≤ 9
        __m256i surv  = in_range_u8(A, base4, width_4_9);
        __m256i n3    = _mm256_and_si256(_mm256_and_si256(is_adult, surv), k3);

        // Combine: exactly one of n0/n1/n2/n3 is non-zero per lane → OR is correct
        _mm256_storeu_si256((__m256i*)(nxt_row + x),
            _mm256_or_si256(_mm256_or_si256(n0, n1),
                            _mm256_or_si256(n2, n3)));
    }

    // Scalar tail (never reached for power-of-2 N ≥ 512 since 512 = 16 × 32)
    for (; x < N; ++x) {
        int A = (int)rs0[x] + (int)rs1[x] + (int)rs2[x]
                            + (int)rs3[x] + (int)rs4[x];
        const uint8_t cell = cur_row[x];
        A -= (cell == 3u);
        uint8_t next;
        switch (cell) {
            case 0:  next = (A >= 3 && A <= 5) ? 1u : 0u; break;
            case 1:  next = 2u;                            break;
            case 2:  next = 3u;                            break;
            case 3:  next = (A >= 4 && A <= 9) ? 3u : 0u; break;
            default: next = 0u;
        }
        nxt_row[x] = next;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Worker thread  (identical structure to ARM version)
// ─────────────────────────────────────────────────────────────────────────────

struct WorkCtx {
    uint8_t* grids[2];
    uint8_t* row_sums;
    int      N;
    int      generations;
};

static void worker(int tid, int nthreads, WorkCtx* wc, std::barrier<>* bar)
{
    const int N = wc->N;

    const int base    = N / nthreads;
    const int rem     = N % nthreads;
    const int y_start = tid * base + std::min(tid, rem);
    const int y_end   = y_start + base + (tid < rem ? 1 : 0);

    std::vector<uint8_t> ap(static_cast<size_t>(N) + 4);  // thread-local padded row

    for (int gen = 0; gen < wc->generations; ++gen) {
        const uint8_t* cur = wc->grids[ gen      & 1];
        uint8_t*       nxt = wc->grids[(gen + 1) & 1];
        uint8_t*       rs  = wc->row_sums;

        // ── Pass 1: horizontal row sums ──────────────────────────────────
        for (int y = y_start; y < y_end; ++y) {
            const uint8_t* gr = cur + static_cast<size_t>(y) * N;
            build_adult_padded(gr, ap.data(), N);
            sliding_sum5(ap.data(), rs + static_cast<size_t>(y) * N, N);
        }

        bar->arrive_and_wait();   // all row_sums ready

        // ── Pass 2: vertical sum + rules ─────────────────────────────────
        for (int y = y_start; y < y_end; ++y) {
            const int y0 = (y - 2 + N) % N;
            const int y1 = (y - 1 + N) % N;
            const int y3 = (y + 1) % N;
            const int y4 = (y + 2) % N;

            apply_rules(
                cur + static_cast<size_t>(y)  * N,
                rs  + static_cast<size_t>(y0) * N,
                rs  + static_cast<size_t>(y1) * N,
                rs  + static_cast<size_t>(y)  * N,
                rs  + static_cast<size_t>(y3) * N,
                rs  + static_cast<size_t>(y4) * N,
                nxt + static_cast<size_t>(y)  * N,
                N);
        }

        bar->arrive_and_wait();   // all of nxt written
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
            "Usage: %s <input.bin> <output.bin> [generations]\n", argv[0]);
        return 1;
    }

    int generations = 10000;
    if (argc == 4) {
        char* end;
        long g = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || g <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        generations = static_cast<int>(g);
    }

    // ── Read input ───────────────────────────────────────────────────────────
    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::fprintf(stderr, "Error: cannot open input '%s'\n", argv[1]);
        return 2;
    }

    uint64_t width = 0, height = 0;
    if (std::fread(&width,  sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, fin) != 1) {
        std::fprintf(stderr, "Error: cannot read header\n");
        std::fclose(fin); return 3;
    }
    if (width == 0 || width != height) {
        std::fprintf(stderr,
            "Error: grid must be square and non-empty, got "
            "%" PRIu64 " × %" PRIu64 "\n", width, height);
        std::fclose(fin); return 3;
    }

    const int    N     = static_cast<int>(width);
    const size_t CELLS = static_cast<size_t>(N) * N;

    auto alloc_aligned = [](size_t bytes) -> uint8_t* {
        void* p = nullptr;
        if (posix_memalign(&p, 64, bytes) != 0 || p == nullptr) {
            std::fprintf(stderr, "Fatal: allocation failed (%zu bytes)\n", bytes);
            std::exit(1);
        }
        return static_cast<uint8_t*>(p);
    };

    uint8_t* grid_a   = alloc_aligned(CELLS);
    uint8_t* grid_b   = alloc_aligned(CELLS);
    uint8_t* row_sums = alloc_aligned(CELLS);

    if (std::fread(grid_a, 1, CELLS, fin) != CELLS) {
        std::fprintf(stderr, "Error: input file truncated\n");
        std::fclose(fin); return 4;
    }
    std::fclose(fin);

    // ── Simulate ─────────────────────────────────────────────────────────────
    WorkCtx wc{ {grid_a, grid_b}, row_sums, N, generations };
    std::barrier<> bar(NTHREADS);

    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(NTHREADS - 1);
    for (int t = 1; t < NTHREADS; ++t)
        threads.emplace_back(worker, t, NTHREADS, &wc, &bar);

    worker(0, NTHREADS, &wc, &bar);
    for (auto& th : threads) th.join();

    auto t1 = std::chrono::steady_clock::now();
    std::printf("%.3f ms\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    // ── Write output ─────────────────────────────────────────────────────────
    uint8_t* result = wc.grids[generations & 1];

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output '%s'\n", argv[2]);
        return 5;
    }

    const bool ok =
        (std::fwrite(&width,  sizeof(uint64_t), 1, fout) == 1) &&
        (std::fwrite(&height, sizeof(uint64_t), 1, fout) == 1) &&
        (std::fwrite(result,  1, CELLS, fout) == CELLS);
    std::fclose(fout);

    if (!ok) {
        std::fprintf(stderr, "Error: write error on output '%s'\n", argv[2]);
        return 6;
    }

    std::free(grid_a);
    std::free(grid_b);
    std::free(row_sums);
    return 0;
}