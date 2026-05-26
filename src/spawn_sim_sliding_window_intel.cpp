// spawn_sim_sw_x86.cpp  —  Monster Spawning Grid, Intel AVX2, sliding-window V.
//
// Identical algorithm to spawn_sim_sw_arm.cpp.  Only the SIMD layer differs:
//   ARM  NEON:  uint8x16_t,   16 cells per register  (<arm_neon.h>)
//   Intel AVX2: __m256i,      32 cells per register  (<immintrin.h>)
//
// See spawn_sim_sw_arm.cpp for the full algorithm description and correctness
// arguments.  This file contains only a brief recap.
//
// ALGORITHM RECAP
// ───────────────
// Each thread owns a stripe of rows [y_start, y_end).
// Per generation:
//
//   Init:  Compute row_sums for rows y_start-2 … y_start+2.
//          Sum them into V[x] (the 5×5 running total).
//
//   Loop (y = y_start … y_end-1):
//     a. apply_rules(cur[y], V)  →  nxt[y]          [1 load of V, not 5]
//     b. Compute row_sum for grid row (y+3) % N
//        → store in ring_buf[(y+3) & 7]              [ring slot for new row]
//     c. V[x] = V[x] − ring_buf[(y-2)&7][x]
//                     + ring_buf[(y+3)&7][x]         [slide the 5×5 window]
//
//   Barrier  (one per generation; down from two in the two-pass version)
//
// MEMORY SAVINGS vs TWO-PASS
// ──────────────────────────
//  • No global row_sums array (saves 1 GiB for N=32768).
//  • Per-generation DRAM traffic:  ~2 × N²  (was ~9 × N²).
//  • Working set per thread:  ~320 KiB  (ring_buf + V + ap) → fits in L2.
//
// UNSIGNED RANGE CHECK  (same trick as spawn_sim_x86.cpp)
// ────────────────────
//   3 ≤ A ≤ 5  ⟺  (uint8)(A − 3) ≤ 2
//   Check "(uint8)d ≤ w":  min_epu8(d, W) == d

#include <immintrin.h>
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
static constexpr int RING     = 8;
static constexpr int RMASK    = RING - 1;

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 1 helpers
// ─────────────────────────────────────────────────────────────────────────────

static inline void build_adult_padded(const uint8_t* __restrict__ gr,
                                       uint8_t* __restrict__       ap,
                                       int N)
{
    ap[0]   = (gr[N-2] == 3u) ? 1u : 0u;
    ap[1]   = (gr[N-1] == 3u) ? 1u : 0u;
    ap[N+2] = (gr[0]   == 3u) ? 1u : 0u;
    ap[N+3] = (gr[1]   == 3u) ? 1u : 0u;

    const __m256i v3 = _mm256_set1_epi8(3);
    const __m256i v1 = _mm256_set1_epi8(1);
    int x = 0;
    for (; x + 32 <= N; x += 32) {
        __m256i cells = _mm256_loadu_si256((const __m256i*)(gr + x));
        _mm256_storeu_si256((__m256i*)(ap + 2 + x),
            _mm256_and_si256(_mm256_cmpeq_epi8(cells, v3), v1));
    }
    for (; x < N; ++x)
        ap[2 + x] = (gr[x] == 3u) ? 1u : 0u;
}

static inline void sliding_sum5(const uint8_t* __restrict__ ap,
                                  uint8_t* __restrict__       rs,
                                  int N)
{
    int x = 0;
    for (; x + 32 <= N; x += 32) {
        __m256i s =
            _mm256_add_epi8(
                _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(ap+x)),
                                _mm256_loadu_si256((const __m256i*)(ap+x+1))),
                _mm256_add_epi8(
                    _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(ap+x+2)),
                                    _mm256_loadu_si256((const __m256i*)(ap+x+3))),
                    _mm256_loadu_si256((const __m256i*)(ap+x+4))));
        _mm256_storeu_si256((__m256i*)(rs + x), s);
    }
    for (; x < N; ++x)
        rs[x] = ap[x] + ap[x+1] + ap[x+2] + ap[x+3] + ap[x+4];
}

// ─────────────────────────────────────────────────────────────────────────────
//  V maintenance helpers
// ─────────────────────────────────────────────────────────────────────────────

// V[x] += rs[x]  (used 5 times during per-generation initialisation)
static inline void add_rows(uint8_t* __restrict__       V,
                             const uint8_t* __restrict__ rs, int N)
{
    int x = 0;
    for (; x + 32 <= N; x += 32)
        _mm256_storeu_si256((__m256i*)(V + x),
            _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(V  + x)),
                            _mm256_loadu_si256((const __m256i*)(rs + x))));
    for (; x < N; ++x)
        V[x] = (uint8_t)(V[x] + rs[x]);
}

// Slide V one row:  V[x] = V[x] − rs_old[x] + rs_new[x].
// V[x] ≥ rs_old[x] always (rs_old is one of the five summed rows), so no underflow.
static inline void slide_V(uint8_t* __restrict__       V,
                            const uint8_t* __restrict__ rs_new,
                            const uint8_t* __restrict__ rs_old,
                            int N)
{
    int x = 0;
    for (; x + 32 <= N; x += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(V      + x));
        __m256i n = _mm256_loadu_si256((const __m256i*)(rs_new + x));
        __m256i o = _mm256_loadu_si256((const __m256i*)(rs_old + x));
        _mm256_storeu_si256((__m256i*)(V + x),
            _mm256_add_epi8(_mm256_sub_epi8(v, o), n));
    }
    for (; x < N; ++x)
        V[x] = (uint8_t)(V[x] - rs_old[x] + rs_new[x]);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rule application
// ─────────────────────────────────────────────────────────────────────────────

// Unsigned range check helper (see spawn_sim_x86.cpp for full explanation):
//   base ≤ v ≤ base+width  ⟺  min_epu8(v−base, width) == v−base
static inline __m256i in_range_u8(const __m256i v,
                                   const __m256i base,
                                   const __m256i width)
{
    __m256i d = _mm256_sub_epi8(v, base);
    return _mm256_cmpeq_epi8(_mm256_min_epu8(d, width), d);
}

// V[x] is the 5×5 sum INCLUDING the centre cell; we subtract it here.
static inline void apply_rules(const uint8_t* __restrict__ cur_row,
                                const uint8_t* __restrict__ V,
                                uint8_t* __restrict__       nxt_row,
                                int N)
{
    const __m256i k0 = _mm256_setzero_si256();
    const __m256i k1 = _mm256_set1_epi8(1), k2 = _mm256_set1_epi8(2),
                  k3 = _mm256_set1_epi8(3);
    const __m256i base3 = _mm256_set1_epi8(3), w35 = _mm256_set1_epi8(2);  // 5-3=2
    const __m256i base4 = _mm256_set1_epi8(4), w49 = _mm256_set1_epi8(5);  // 9-4=5

    int x = 0;
    for (; x + 32 <= N; x += 32) {
        __m256i A = _mm256_loadu_si256((const __m256i*)(V       + x));
        __m256i c = _mm256_loadu_si256((const __m256i*)(cur_row + x));

        // Subtract centre adult
        A = _mm256_sub_epi8(A, _mm256_and_si256(_mm256_cmpeq_epi8(c, k3), k1));

        __m256i is0 = _mm256_cmpeq_epi8(c, k0);
        __m256i is1 = _mm256_cmpeq_epi8(c, k1);
        __m256i is2 = _mm256_cmpeq_epi8(c, k2);
        __m256i is3 = _mm256_cmpeq_epi8(c, k3);

        __m256i n0 = _mm256_and_si256(_mm256_and_si256(is0, in_range_u8(A,base3,w35)), k1);
        __m256i n1 = _mm256_and_si256(is1, k2);
        __m256i n2 = _mm256_and_si256(is2, k3);
        __m256i n3 = _mm256_and_si256(_mm256_and_si256(is3, in_range_u8(A,base4,w49)), k3);

        _mm256_storeu_si256((__m256i*)(nxt_row + x),
            _mm256_or_si256(_mm256_or_si256(n0, n1), _mm256_or_si256(n2, n3)));
    }
    for (; x < N; ++x) {
        int A = (int)V[x];
        uint8_t cell = cur_row[x];
        A -= (cell == 3u);
        uint8_t nv;
        switch (cell) {
            case 0:  nv = (A >= 3 && A <= 5) ? 1u : 0u; break;
            case 1:  nv = 2u;                            break;
            case 2:  nv = 3u;                            break;
            case 3:  nv = (A >= 4 && A <= 9) ? 3u : 0u; break;
            default: nv = 0u;
        }
        nxt_row[x] = nv;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Worker thread
// ─────────────────────────────────────────────────────────────────────────────

struct WorkCtx {
    uint8_t* grids[2];
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

    std::vector<uint8_t> ring_buf((size_t)RING * N, 0u);
    std::vector<uint8_t> V  (N, 0u);
    std::vector<uint8_t> ap (static_cast<size_t>(N) + 4, 0u);

    auto ring_slot = [&](int d) -> uint8_t* {
        return ring_buf.data() + (size_t)(d & RMASK) * N;
    };

    for (int gen = 0; gen < wc->generations; ++gen) {
        const uint8_t* cur = wc->grids[ gen      & 1];
        uint8_t*       nxt = wc->grids[(gen + 1) & 1];

        // ── Initialise V for the first window ────────────────────────────
        std::fill(V.begin(), V.end(), 0u);

        for (int d = y_start - 2; d <= y_start + 2; ++d) {
            int gr = ((d % N) + N) % N;
            build_adult_padded(cur + (size_t)gr * N, ap.data(), N);
            uint8_t* slot = ring_slot(d);
            sliding_sum5(ap.data(), slot, N);
            add_rows(V.data(), slot, N);
        }

        // ── Sliding loop ─────────────────────────────────────────────────
        for (int y = y_start; y < y_end; ++y) {

            // a. Apply rules using the current 5×5 sum V
            apply_rules(cur + (size_t)y * N, V.data(),
                        nxt + (size_t)y * N, N);

            // b. Compute row_sum for the row entering the window (y+3)
            int      new_gr = (y + 3) % N;
            uint8_t* rs_new = ring_slot(y + 3);
            build_adult_padded(cur + (size_t)new_gr * N, ap.data(), N);
            sliding_sum5(ap.data(), rs_new, N);

            // c. Slide V: drop row y-2, add row y+3
            slide_V(V.data(), rs_new, ring_slot(y - 2), N);
        }

        bar->arrive_and_wait();
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

    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) { std::fprintf(stderr, "Error: cannot open '%s'\n", argv[1]); return 2; }

    uint64_t width = 0, height = 0;
    if (std::fread(&width,  8, 1, fin) != 1 ||
        std::fread(&height, 8, 1, fin) != 1 ||
        width == 0 || width != height) {
        std::fprintf(stderr, "Error: invalid header\n");
        std::fclose(fin); return 3;
    }

    const int    N     = static_cast<int>(width);
    const size_t CELLS = static_cast<size_t>(N) * N;

    auto alloc = [](size_t n) -> uint8_t* {
        void* p = nullptr;
        if (posix_memalign(&p, 64, n) != 0 || !p) {
            std::fprintf(stderr, "Fatal: alloc failed\n"); std::exit(1);
        }
        return static_cast<uint8_t*>(p);
    };

    uint8_t* grid_a = alloc(CELLS);
    uint8_t* grid_b = alloc(CELLS);

    if (std::fread(grid_a, 1, CELLS, fin) != CELLS) {
        std::fprintf(stderr, "Error: input truncated\n");
        std::fclose(fin); return 4;
    }
    std::fclose(fin);

    WorkCtx wc{ {grid_a, grid_b}, N, generations };
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

    uint8_t* result = wc.grids[generations & 1];

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) { std::fprintf(stderr, "Error: cannot open '%s'\n", argv[2]); return 5; }

    const bool ok =
        (std::fwrite(&width,  8, 1, fout) == 1) &&
        (std::fwrite(&height, 8, 1, fout) == 1) &&
        (std::fwrite(result, 1, CELLS, fout) == CELLS);
    std::fclose(fout);
    if (!ok) { std::fprintf(stderr, "Error: write error\n"); return 6; }

    std::free(grid_a);
    std::free(grid_b);
    return 0;
}