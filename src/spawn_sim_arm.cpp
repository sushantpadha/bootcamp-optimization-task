// spawn_sim.cpp — Optimised Monster Spawning Grid simulator.
//
// Algorithm: two-pass tiled computation.
//
//   Pass 1 (horizontal):
//     For each of TR+4 rows in the tile (TR output rows plus a 2-row halo
//     top and bottom), compute a 5-tap horizontal sum of ADULT cells into a
//     thread-local  hsum[TR+4][TC]  buffer.  Each value is in [0, 5].
//
//   Pass 2 (vertical + transition):
//     For each of the TR output rows, sum five hsum rows (±2), subtract the
//     centre cell's ADULT flag to get neighbour count A, then apply the
//     state transition rules.
//
// Cache behaviour:
//   hsum (~66 KB) lives on the stack and stays warm in L2 for the whole
//   tile.  The state reads per tile are (TR+4)×(TC+4) ≈ 68 KB, also L2-
//   resident.  Total working set per tile ≈ 200 KB — well within the 2 MB
//   L2 on Graviton4 — avoiding a full N² hsum round-trip to main memory.
//
// Threading:
//   Up to 8 persistent threads share a std::barrier.  An atomic counter
//   distributes tiles (work-stealing).  The barrier completion function
//   swaps the active grid index and resets the counter each generation.
//
// SIMD:
//   Explicit NEON on AArch64, AVX2 on x86-64.  The scalar fallback is
//   written branchless so GCC/Clang auto-vectorise it with -O3 -march=native.
//
// Memory:
//   Grids allocated with mmap(MAP_ANONYMOUS) + madvise(MADV_HUGEPAGE) to
//   use 2 MB transparent huge pages, keeping TLB pressure low on 1 GB grids.
//
// Build:
//   ARM:  g++-14 -std=c++23 -O3 -mcpu=neoverse-v2 -pthread -o spawn_sim spawn_sim.cpp
//   x86:  g++-14 -std=c++23 -O3 -march=native     -pthread -o spawn_sim spawn_sim.cpp

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifdef __linux__
#  include <sys/mman.h>
#endif
#ifdef __ARM_NEON__
#  include <arm_neon.h>
#endif
#ifdef __AVX2__
#  include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// Compile-time tuning knobs
// ---------------------------------------------------------------------------
static constexpr int TR           = 256;    // tile height (rows)
static constexpr int TC           = 256;    // tile width  (cols); must divide N
static constexpr int HALO         = 2;      // stencil radius
static constexpr int MAX_THREADS  = 8;
static constexpr int DEFAULT_GENS = 10000;

// ---------------------------------------------------------------------------
// Grid allocation: mmap + transparent huge pages on Linux, aligned_alloc elsewhere
// ---------------------------------------------------------------------------
static uint8_t* grid_alloc(size_t bytes)
{
#ifdef __linux__
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
    madvise(p, bytes, MADV_HUGEPAGE);
    return static_cast<uint8_t*>(p);
#else
    constexpr size_t A = 64;
    void* p = std::aligned_alloc(A, (bytes + A - 1) & ~(A - 1));
    if (!p) { std::fputs("OOM\n", stderr); std::exit(1); }
    return static_cast<uint8_t*>(p);
#endif
}

static void grid_free(uint8_t* p, [[maybe_unused]] size_t bytes)
{
#ifdef __linux__
    munmap(p, bytes);
#else
    std::free(p);
#endif
}

// ---------------------------------------------------------------------------
// Core: compute one TR×TC output tile.
//
//   cur  — read-only current generation (full N×N grid, row-major)
//   nxt  — write-only next generation
//   r0, c0 — tile top-left corner in grid coordinates
//   N    — grid dimension (power of 2, ≥ 2·TC)
// ---------------------------------------------------------------------------
[[gnu::hot, gnu::noinline]]
static void process_tile(
        const uint8_t* __restrict__ cur,
        uint8_t*       __restrict__ nxt,
        int r0, int c0, int N)
{
    // Thread-local stack buffers.
    // hsum[ri][ci] = 5-tap horizontal ADULT count for grid row (r0-HALO+ri), col (c0+ci).
    alignas(64) uint8_t hsum[TR + 4][TC];
    // Scratch for rows that straddle the column wrap boundary (≤ 2 of N/TC tiles).
    alignas(64) uint8_t row_buf[TC + 8];   // +8 = safe SIMD over-read margin

    const bool need_col_wrap = (c0 < HALO) | (c0 + TC + HALO > N);

    // -----------------------------------------------------------------------
    // Pass 1: horizontal 5-tap adult-count sum → hsum
    // -----------------------------------------------------------------------
    for (int ri = 0; ri < TR + 4; ++ri) {
        const int      r    = (r0 - HALO + ri + N) & (N - 1);   // toroidal row
        const uint8_t* row  = cur + (size_t)r * N;
        uint8_t*       hrow = hsum[ri];

        // src[0] = grid column (c0 - HALO), src[TC+3] = grid column (c0 + TC + 1)
        const uint8_t* src;
        if (__builtin_expect(!need_col_wrap, 1)) {
            src = row + (c0 - HALO);
        } else {
            const int base = c0 - HALO;
            for (int k = 0; k < TC + 4; ++k)
                row_buf[k] = row[(base + k + N) & (N - 1)];
            src = row_buf;
        }

        // hrow[ci] = count of src[ci..ci+4] that equal ADULT (3)
#ifdef __ARM_NEON__
        {
            const uint8x16_t v3 = vdupq_n_u8(3);
            for (int ci = 0; ci < TC; ci += 16) {
                // vceqq_u8 → 0xFF where equal; vshrq_n_u8(...,7) → 0 or 1
                uint8x16_t a0 = vshrq_n_u8(vceqq_u8(vld1q_u8(src + ci    ), v3), 7);
                uint8x16_t a1 = vshrq_n_u8(vceqq_u8(vld1q_u8(src + ci + 1), v3), 7);
                uint8x16_t a2 = vshrq_n_u8(vceqq_u8(vld1q_u8(src + ci + 2), v3), 7);
                uint8x16_t a3 = vshrq_n_u8(vceqq_u8(vld1q_u8(src + ci + 3), v3), 7);
                uint8x16_t a4 = vshrq_n_u8(vceqq_u8(vld1q_u8(src + ci + 4), v3), 7);
                vst1q_u8(hrow + ci,
                    vaddq_u8(vaddq_u8(vaddq_u8(vaddq_u8(a0, a1), a2), a3), a4));
            }
        }
#elif defined(__AVX2__)
        {
            const __m256i v3 = _mm256_set1_epi8(3);
            const __m256i v1 = _mm256_set1_epi8(1);
            for (int ci = 0; ci < TC; ci += 32) {
                // AND with 1 turns 0xFF→1 efficiently on AVX2
                __m256i a0 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(src+ci  )), v3), v1);
                __m256i a1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(src+ci+1)), v3), v1);
                __m256i a2 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(src+ci+2)), v3), v1);
                __m256i a3 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(src+ci+3)), v3), v1);
                __m256i a4 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(src+ci+4)), v3), v1);
                _mm256_storeu_si256((__m256i*)(hrow+ci),
                    _mm256_add_epi8(_mm256_add_epi8(_mm256_add_epi8(_mm256_add_epi8(a0, a1), a2), a3), a4));
            }
        }
#else
        // Branchless scalar — GCC/Clang auto-vectorise with -O3 -march=native
        for (int ci = 0; ci < TC; ++ci)
            hrow[ci] = uint8_t(  (src[ci  ] == 3) + (src[ci+1] == 3)
                                + (src[ci+2] == 3) + (src[ci+3] == 3)
                                + (src[ci+4] == 3));
#endif
    } // end Pass 1

    // -----------------------------------------------------------------------
    // Pass 2: vertical 5-tap sum + state transition
    // -----------------------------------------------------------------------
    for (int ri = HALO; ri < TR + HALO; ++ri) {
        const int      r    = (r0 - HALO + ri + N) & (N - 1);
        const uint8_t* srow = cur + (size_t)r * N + c0;   // current state, tile cols
        uint8_t*       nrow = nxt + (size_t)r * N + c0;   // output

        const uint8_t* h0 = hsum[ri - 2];
        const uint8_t* h1 = hsum[ri - 1];
        const uint8_t* h2 = hsum[ri    ];
        const uint8_t* h3 = hsum[ri + 1];
        const uint8_t* h4 = hsum[ri + 2];

#ifdef __ARM_NEON__
        {
            const uint8x16_t c0v = vdupq_n_u8(0);
            const uint8x16_t c1  = vdupq_n_u8(1),  c2 = vdupq_n_u8(2);
            const uint8x16_t c3  = vdupq_n_u8(3),  c4 = vdupq_n_u8(4);
            const uint8x16_t c5  = vdupq_n_u8(5),  c9 = vdupq_n_u8(9);

            for (int ci = 0; ci < TC; ci += 16) {
                // Sum five hsum rows (= total adults in 5×5 window including centre)
                uint8x16_t vsum = vaddq_u8(vaddq_u8(vaddq_u8(vaddq_u8(
                    vld1q_u8(h0+ci), vld1q_u8(h1+ci)),
                    vld1q_u8(h2+ci)), vld1q_u8(h3+ci)),
                    vld1q_u8(h4+ci));

                uint8x16_t state = vld1q_u8(srow + ci);
                uint8x16_t adt   = vceqq_u8(state, c3);           // 0xFF where ADULT
                // A = neighbours only (subtract centre cell from window sum)
                uint8x16_t A     = vsubq_u8(vsum, vandq_u8(adt, c1));

                // Transition masks (0xFF = condition true)
                // EMPTY births if 3 ≤ A ≤ 5
                uint8x16_t born = vandq_u8(vceqq_u8(state, c0v),
                                  vandq_u8(vcgeq_u8(A, c3), vcleq_u8(A, c5)));
                // ADULT survives if 4 ≤ A ≤ 9
                uint8x16_t surv = vandq_u8(adt,
                                  vandq_u8(vcgeq_u8(A, c4), vcleq_u8(A, c9)));

                // Next state.  Four cases are mutually exclusive → safe to OR.
                //   EGG  (1) → JUVENILE (2)
                //   JUV  (2) → ADULT    (3)
                //   SURV (3) → ADULT    (3)
                //   BORN (0) → EGG      (1)
                //   else     → EMPTY    (0)
                uint8x16_t next = vorrq_u8(
                    vorrq_u8(vandq_u8(vceqq_u8(state, c1), c2),
                             vandq_u8(vceqq_u8(state, c2), c3)),
                    vorrq_u8(vandq_u8(surv, c3),
                             vandq_u8(born, c1)));
                vst1q_u8(nrow + ci, next);
            }
        }
#elif defined(__AVX2__)
        {
            const __m256i c0v = _mm256_setzero_si256();
            const __m256i c1  = _mm256_set1_epi8(1),  c2 = _mm256_set1_epi8(2);
            const __m256i c3  = _mm256_set1_epi8(3),  c4 = _mm256_set1_epi8(4);
            const __m256i c5  = _mm256_set1_epi8(5),  c9 = _mm256_set1_epi8(9);

            for (int ci = 0; ci < TC; ci += 32) {
                __m256i vsum = _mm256_add_epi8(
                    _mm256_add_epi8(
                        _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(h0+ci)),
                                        _mm256_loadu_si256((const __m256i*)(h1+ci))),
                        _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(h2+ci)),
                                        _mm256_loadu_si256((const __m256i*)(h3+ci)))),
                    _mm256_loadu_si256((const __m256i*)(h4+ci)));

                __m256i state = _mm256_loadu_si256((const __m256i*)(srow+ci));
                __m256i adt   = _mm256_cmpeq_epi8(state, c3);
                __m256i A     = _mm256_sub_epi8(vsum, _mm256_and_si256(adt, c1));

                // AVX2 has no unsigned >=/<= directly.
                // a >= b  ↔  max_epu8(a,b) == a
                // a <= b  ↔  min_epu8(a,b) == a
                __m256i age3 = _mm256_cmpeq_epi8(_mm256_max_epu8(A, c3), A);  // A >= 3
                __m256i ale5 = _mm256_cmpeq_epi8(_mm256_min_epu8(A, c5), A);  // A <= 5
                __m256i age4 = _mm256_cmpeq_epi8(_mm256_max_epu8(A, c4), A);  // A >= 4
                __m256i ale9 = _mm256_cmpeq_epi8(_mm256_min_epu8(A, c9), A);  // A <= 9

                __m256i born = _mm256_and_si256(_mm256_cmpeq_epi8(state, c0v),
                               _mm256_and_si256(age3, ale5));
                __m256i surv = _mm256_and_si256(adt,
                               _mm256_and_si256(age4, ale9));

                __m256i next = _mm256_or_si256(
                    _mm256_or_si256(_mm256_and_si256(_mm256_cmpeq_epi8(state, c1), c2),
                                    _mm256_and_si256(_mm256_cmpeq_epi8(state, c2), c3)),
                    _mm256_or_si256(_mm256_and_si256(surv, c3),
                                    _mm256_and_si256(born, c1)));
                _mm256_storeu_si256((__m256i*)(nrow+ci), next);
            }
        }
#else
        // Branchless scalar — auto-vectorisable
        for (int ci = 0; ci < TC; ++ci) {
            const uint8_t vs   = uint8_t(h0[ci] + h1[ci] + h2[ci] + h3[ci] + h4[ci]);
            const uint8_t A    = uint8_t(vs - uint8_t(srow[ci] == 3));
            const uint8_t s    = srow[ci];
            const uint8_t ie   = uint8_t(s == 0);
            const uint8_t ieg  = uint8_t(s == 1);
            const uint8_t ij   = uint8_t(s == 2);
            const uint8_t ia   = uint8_t(s == 3);
            const uint8_t born = uint8_t(ie  & uint8_t(A >= 3) & uint8_t(A <= 5));
            const uint8_t surv = uint8_t(ia  & uint8_t(A >= 4) & uint8_t(A <= 9));
            nrow[ci] = uint8_t((ieg * 2u) | (ij * 3u) | (surv * 3u) | (born * 1u));
        }
#endif
    } // end Pass 2
}

// ---------------------------------------------------------------------------
// Simulation driver: persistent thread pool + std::barrier
// ---------------------------------------------------------------------------
// Returns a pointer to whichever of ga/gb holds the final result.
static const uint8_t* simulate(uint8_t* ga, uint8_t* gb, int N, int gens)
{
    if (gens == 0) return ga;

    const int tpr   = N / TC;          // tiles per row (= tiles per column)
    const int total = tpr * tpr;       // total tiles per generation
    const int nth   = std::max(1, std::min(MAX_THREADS,
                          static_cast<int>(std::thread::hardware_concurrency())));

    uint8_t* grids[2] = {ga, gb};
    std::atomic<int> cur_idx{0};   // index into grids[] for the current generation
    std::atomic<int> tile_ctr{0};  // work-stealing counter

    // Barrier completion: runs once per generation after all threads arrive.
    // Swaps the active grid and resets the tile counter for the next generation.
    // The barrier's happens-before guarantee means plain relaxed ops are fine here.
    auto on_gen_end = [&]() noexcept {
        cur_idx.fetch_xor(1, std::memory_order_relaxed);
        tile_ctr.store(0, std::memory_order_relaxed);
    };
    std::barrier sync{nth, on_gen_end};

    auto worker = [&] {
        for (int g = 0; g < gens; ++g) {
            // Read cur/nxt after the barrier (or at the very start).
            // The barrier guarantees we see the completion function's writes.
            const int      ci = cur_idx.load(std::memory_order_relaxed);
            const uint8_t* C  = grids[ci];
            uint8_t*       NX = grids[ci ^ 1];

            int idx;
            while ((idx = tile_ctr.fetch_add(1, std::memory_order_relaxed)) < total) {
                const int tr = idx / tpr;
                const int tc = idx % tpr;
                process_tile(C, NX, tr * TR, tc * TC, N);
            }

            sync.arrive_and_wait();
            // After this point cur_idx is updated and tile_ctr is 0.
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(nth - 1);
    for (int t = 1; t < nth; ++t)
        pool.emplace_back(worker);
    worker();   // main thread participates as well
    for (auto& t : pool) t.join();

    // cur_idx now points to the grid that was last written.
    return grids[cur_idx.load()];
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
            "Usage: %s <input.bin> <output.bin> [generations]\n", argv[0]);
        return 1;
    }

    int gens = DEFAULT_GENS;
    if (argc == 4) {
        char* end;
        long g = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || g <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        gens = static_cast<int>(g);
    }

    // -------------------------------------------------------------------------
    // Read input
    // -------------------------------------------------------------------------
    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]);
        return 2;
    }

    uint64_t width, height;
    if (std::fread(&width,  sizeof width,  1, fin) != 1 ||
        std::fread(&height, sizeof height, 1, fin) != 1) {
        std::fprintf(stderr, "Error: input file too short (cannot read header)\n");
        std::fclose(fin);
        return 3;
    }
    if (width == 0 || width != height) {
        std::fprintf(stderr,
            "Error: grid must be square and non-empty "
            "(got %" PRIu64 " x %" PRIu64 ")\n", width, height);
        std::fclose(fin);
        return 3;
    }

    const int    N = static_cast<int>(width);
    const size_t M = static_cast<size_t>(N) * N;

    uint8_t* ga = grid_alloc(M);
    if (std::fread(ga, 1, M, fin) != M) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        grid_free(ga, M);
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    uint8_t* gb = grid_alloc(M);

    // -------------------------------------------------------------------------
    // Simulate (timed)
    // -------------------------------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();

    const uint8_t* result = simulate(ga, gb, N, gens);

    auto t1 = std::chrono::steady_clock::now();
    std::printf("%.3f ms\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    // -------------------------------------------------------------------------
    // Write output
    // -------------------------------------------------------------------------
    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        grid_free(ga, M); grid_free(gb, M);
        return 5;
    }
    if (std::fwrite(&width,  sizeof width,  1, fout) != 1 ||
        std::fwrite(&height, sizeof height, 1, fout) != 1 ||
        std::fwrite(result, 1, M, fout) != M) {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(fout);
        grid_free(ga, M); grid_free(gb, M);
        return 6;
    }
    std::fclose(fout);
    grid_free(ga, M);
    grid_free(gb, M);
    return 0;
}