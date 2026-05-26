// spawn_sim_sw_v2_x86.cpp  —  Sliding-window Monster Spawning Grid, Intel AVX2.
//                              Version 2: fused row-sum (no intermediate `ap`).
//
// See spawn_sim_sw_v2_arm.cpp for the full rationale.
// This file differs only in the SIMD layer: AVX2 (__m256i, 32 cells/register)
// instead of NEON (uint8x16_t, 16 cells/register).
//
// compute_row_sum() interior SIMD range for AVX2:
//   Condition:  x + 32 <= N - 2  →  x ≤ N-34
//   Widest load: gr + x + 2 + 31 = gr + N - 34 + 33 = gr + N - 1  ✓
//   After loop:  x = N-2  →  scalar remainder never reached for 2^k N ≥ 512.
//
// Unsigned range-check trick (same as spawn_sim_x86.cpp):
//   base ≤ v ≤ base+width  ⟺  min_epu8(v−base, width) == v−base

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
//  Fused row-sum
// ─────────────────────────────────────────────────────────────────────────────

static inline void compute_row_sum(const uint8_t* __restrict__ gr,
                                    uint8_t* __restrict__       rs,
                                    int N)
{
    // ── Left boundary ────────────────────────────────────────────────────
    rs[0] = (gr[N-2]==3u) + (gr[N-1]==3u) + (gr[0]==3u) + (gr[1]==3u) + (gr[2]==3u);
    rs[1] = (gr[N-1]==3u) + (gr[0]==3u)   + (gr[1]==3u) + (gr[2]==3u) + (gr[3]==3u);

    // ── SIMD interior ────────────────────────────────────────────────────
    // Load grid at 5 offsets, compare to 3, AND to 0/1, sum.
    // All loads are safe: x ∈ [2, N-34], offsets -2..+2, width 32 → [0, N-1].
    const __m256i v3 = _mm256_set1_epi8(3);
    const __m256i v1 = _mm256_set1_epi8(1);

    int x = 2;
    for (; x + 32 <= N - 2; x += 32) {
        __m256i a0 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(gr+x-2)), v3), v1);
        __m256i a1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(gr+x-1)), v3), v1);
        __m256i a2 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(gr+x  )), v3), v1);
        __m256i a3 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(gr+x+1)), v3), v1);
        __m256i a4 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(gr+x+2)), v3), v1);

        _mm256_storeu_si256((__m256i*)(rs + x),
            _mm256_add_epi8(
                _mm256_add_epi8(a0, a1),
                _mm256_add_epi8(_mm256_add_epi8(a2, a3), a4)));
    }

    // ── Scalar remainder (never reached for power-of-2 N ≥ 512) ─────────
    for (; x < N - 2; ++x)
        rs[x] = (gr[x-2]==3u) + (gr[x-1]==3u) + (gr[x]==3u)
              + (gr[x+1]==3u) + (gr[x+2]==3u);

    // ── Right boundary ───────────────────────────────────────────────────
    rs[N-2] = (gr[N-4]==3u) + (gr[N-3]==3u) + (gr[N-2]==3u) + (gr[N-1]==3u) + (gr[0]==3u);
    rs[N-1] = (gr[N-3]==3u) + (gr[N-2]==3u) + (gr[N-1]==3u) + (gr[0]==3u)   + (gr[1]==3u);
}

// ─────────────────────────────────────────────────────────────────────────────
//  V maintenance
// ─────────────────────────────────────────────────────────────────────────────

static inline void add_rows(uint8_t* __restrict__       V,
                             const uint8_t* __restrict__ rs, int N)
{
    int x = 0;
    for (; x + 32 <= N; x += 32)
        _mm256_storeu_si256((__m256i*)(V+x),
            _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(V+x)),
                            _mm256_loadu_si256((const __m256i*)(rs+x))));
    for (; x < N; ++x) V[x] = (uint8_t)(V[x] + rs[x]);
}

static inline void slide_V(uint8_t* __restrict__       V,
                            const uint8_t* __restrict__ rs_new,
                            const uint8_t* __restrict__ rs_old,
                            int N)
{
    int x = 0;
    for (; x + 32 <= N; x += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(V      +x));
        __m256i n = _mm256_loadu_si256((const __m256i*)(rs_new +x));
        __m256i o = _mm256_loadu_si256((const __m256i*)(rs_old +x));
        _mm256_storeu_si256((__m256i*)(V+x),
            _mm256_add_epi8(_mm256_sub_epi8(v, o), n));
    }
    for (; x < N; ++x)
        V[x] = (uint8_t)(V[x] - rs_old[x] + rs_new[x]);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rule application
// ─────────────────────────────────────────────────────────────────────────────

static inline __m256i in_range_u8(const __m256i v,
                                   const __m256i base,
                                   const __m256i width)
{
    __m256i d = _mm256_sub_epi8(v, base);
    return _mm256_cmpeq_epi8(_mm256_min_epu8(d, width), d);
}

static inline void apply_rules(const uint8_t* __restrict__ cur_row,
                                const uint8_t* __restrict__ V,
                                uint8_t* __restrict__       nxt_row,
                                int N)
{
    const __m256i k0    = _mm256_setzero_si256();
    const __m256i k1    = _mm256_set1_epi8(1), k2 = _mm256_set1_epi8(2),
                  k3    = _mm256_set1_epi8(3);
    const __m256i base3 = _mm256_set1_epi8(3), w35 = _mm256_set1_epi8(2);
    const __m256i base4 = _mm256_set1_epi8(4), w49 = _mm256_set1_epi8(5);

    int x = 0;
    for (; x + 32 <= N; x += 32) {
        __m256i A = _mm256_loadu_si256((const __m256i*)(V       + x));
        __m256i c = _mm256_loadu_si256((const __m256i*)(cur_row + x));

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

    // `ap` is gone.  Thread-local footprint: ring_buf (256 KiB) + V (32 KiB).
    std::vector<uint8_t> ring_buf((size_t)RING * N, 0u);
    std::vector<uint8_t> V(N, 0u);

    auto ring_slot = [&](int d) -> uint8_t* {
        return ring_buf.data() + (size_t)(d & RMASK) * N;
    };

    for (int gen = 0; gen < wc->generations; ++gen) {
        const uint8_t* cur = wc->grids[ gen      & 1];
        uint8_t*       nxt = wc->grids[(gen + 1) & 1];

        std::fill(V.begin(), V.end(), 0u);

        for (int d = y_start - 2; d <= y_start + 2; ++d) {
            int gr = ((d % N) + N) % N;
            uint8_t* slot = ring_slot(d);
            compute_row_sum(cur + (size_t)gr * N, slot, N);
            add_rows(V.data(), slot, N);
        }

        for (int y = y_start; y < y_end; ++y) {
            apply_rules(cur + (size_t)y * N, V.data(),
                        nxt + (size_t)y * N, N);

            int      new_gr = (y + 3) % N;
            uint8_t* rs_new = ring_slot(y + 3);
            compute_row_sum(cur + (size_t)new_gr * N, rs_new, N);

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