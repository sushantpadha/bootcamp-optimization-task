// spawn_sim_lut_arm_opt.cpp  —  Monster Spawning Grid, ARM NEON, LUT rules.
//
// - no packing
// - seeing IPC = 3.0x in 32K, 3.9x in 2K
// - LUT helps ig, but need to confirm once that it fits in NEON registers
// - removed ap buffer, every thread uses one ring_buf + V only
// added:
// - thread pinning - minor help
// - refactor compute_row_sum to reduce number of reads - minor help
// - noticed that THP helps a lot

#include <arm_neon.h>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <array>
#include <sys/mman.h>

// * CODE for pinning threads - see commented line in worker
#include <pthread.h>
#include <sched.h>

static void pin_thread(int tid)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(tid, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0)
        std::fprintf(stderr, "Warning: failed to pin thread %d\n", tid);
}

static constexpr int NTHREADS = 8;
static constexpr int RING     = 8;
static constexpr int RMASK    = RING - 1;

// ─────────────────────────────────────────────────────────────────────────────
//  Rule LUT
// ─────────────────────────────────────────────────────────────────────────────
//
//  key = (state << 5) | A   (state in bits 6:5, A in bits 4:0)
//  Max key = (3 << 5) | 24 = 120  →  128-entry table (two 64-byte halves)
//
//  lut_lo[0..63] :  state 0 (EMPTY, keys 0..31) and state 1 (EGG, keys 32..63)
//  lut_hi[0..63] :  state 2 (JUV, keys 0..31 after -64) and state 3 (ADULT, 32..63)

static constexpr auto build_lut_ct() {
    std::array<uint8_t, 128> t{};
    for (int s = 0; s < 4; ++s) {
        for (int a = 0; a < 32; ++a) {
            uint8_t nx;
            switch (s) {
                case 0:  nx = (a >= 3 && a <= 5) ? 1u : 0u; break;
                case 1:  nx = 2u;                            break;
                case 2:  nx = 3u;                            break;
                default: nx = (a >= 4 && a <= 9) ? 3u : 0u; break;
            }
            t[(s << 5) | a] = nx;
        }
    }
    return t;
}

alignas(64) static constexpr auto rule_lut_storage = build_lut_ct();
constexpr const uint8_t* rule_lut = rule_lut_storage.data();

// ─────────────────────────────────────────────────────────────────────────────
//  Row-sum  (fused: no ap buffer)
// ─────────────────────────────────────────────────────────────────────────────
//
//  rs[x] = number of ADULT (state==3) cells in [x-2 .. x+2], toroidal wrap.
//  Interior (x=2..N-3): all 5 window positions are within [0,N-1], no wrap
//    — load 5 overlapping 16-byte windows and sum directly.
//  Boundary (x=0,1,N-2,N-1): scalar with modular indexing.
//
//  vshrq_n_u8(vceqq_u8(v, k3), 7):
//    vceqq gives 0xFF (all-ones) for match, 0x00 otherwise.
//    Right-shift by 7 → 0x01 / 0x00.
//    Two instructions instead of three (compare + and-with-1), avoids
//    keeping a k1=1 constant in a register.

static inline void compute_row_sum(const uint8_t* __restrict__ gr,
                                    uint8_t* __restrict__       rs,
                                    int N)
{
    const uint8x16_t k3 = vdupq_n_u8(3u);
    const int mask = N - 1;

    auto a = [&](int x) -> uint8_t {
        return gr[x & mask] == 3u ? 1u : 0u;
    };

    // Boundary positions (toroidal wrap, scalar)
    rs[0]   = a(-2)+a(-1)+a(0)+a(1)+a(2);
    rs[1]   = a(-1)+a(0)+a(1)+a(2)+a(3);
    rs[N-2] = a(N-4)+a(N-3)+a(N-2)+a(N-1)+a(0);
    rs[N-1] = a(N-3)+a(N-2)+a(N-1)+a(0)+a(1);

    // adult flag: 0xFF → 0x01 via compare+shift
    auto adult = [&](uint8x16_t v) -> uint8x16_t {
        return vshrq_n_u8(vceqq_u8(v, k3), 7);
    };

    // SIMD interior: x in [2, N-2).
    // We load v0 = gr[x-2 .. x+13]  (16 bytes starting at x-2)
    //         v1 = gr[x+14 .. x+29] (16 bytes starting at x+14)
    // Then derive the 5 windows via vextq_u8:
    //   offset -2: bytes 0..15 of v0:v1  → v0            (ext by 0, i.e. v0 itself)
    //   offset -1: bytes 1..16 of v0:v1  → vextq(v0,v1,1)
    //   offset  0: bytes 2..17 of v0:v1  → vextq(v0,v1,2)
    //   offset +1: bytes 3..18 of v0:v1  → vextq(v0,v1,3)
    //   offset +2: bytes 4..19 of v0:v1  → vextq(v0,v1,4)
    //
    // Loop guard: x-2 >= 0  (x >= 2, guaranteed by x starting at 2)
    //         and x+14+15 = x+29 <= N-1  i.e. x <= N-30
    // So SIMD runs while x+16 <= N-28, i.e. the stricter bound x+30 <= N.
    // (Baseline used x+16 <= N-2, i.e. x <= N-18; new guard x <= N-30 is
    //  tighter by 12, meaning up to 12 extra scalar iterations at the tail —
    //  still at most 27 scalar iters total for power-of-2 N, negligible.)

    int x = 2;
    for (; x + 30 <= N; x += 16) {
        uint8x16_t v0 = vld1q_u8(gr + x - 2);
        uint8x16_t v1 = vld1q_u8(gr + x + 14);

        uint8x16_t s = vaddq_u8(
            vaddq_u8(adult(v0),
                     adult(vextq_u8(v0, v1, 1))),
            vaddq_u8(adult(vextq_u8(v0, v1, 2)),
            vaddq_u8(adult(vextq_u8(v0, v1, 3)),
                     adult(vextq_u8(v0, v1, 4)))));

        vst1q_u8(rs + x, s);
    }
    // Scalar tail: covers x in [loop_end, N-3] plus the 4 boundary cells
    // already handled above. At most 27 iterations for power-of-2 N.
    for (; x <= N - 3; ++x)
        rs[x] = a(x-2)+a(x-1)+a(x)+a(x+1)+a(x+2);
}

// ─────────────────────────────────────────────────────────────────────────────
//  V maintenance  (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

static inline void add_rows(uint8_t* __restrict__       V,
                             const uint8_t* __restrict__ rs, int N)
{
    int x = 0;
    for (; x + 16 <= N; x += 16)
        vst1q_u8(V+x, vaddq_u8(vld1q_u8(V+x), vld1q_u8(rs+x)));
    for (; x < N; ++x) V[x] = (uint8_t)(V[x] + rs[x]);
}

static inline void slide_V(uint8_t* __restrict__       V,
                            const uint8_t* __restrict__ rs_new,
                            const uint8_t* __restrict__ rs_old, int N)
{
    int x = 0;
    for (; x + 16 <= N; x += 16)
        vst1q_u8(V+x, vaddq_u8(
            vsubq_u8(vld1q_u8(V+x), vld1q_u8(rs_old+x)),
            vld1q_u8(rs_new+x)));
    for (; x < N; ++x) V[x] = (uint8_t)(V[x] - rs_old[x] + rs_new[x]);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rule application  (LUT-based)
// ─────────────────────────────────────────────────────────────────────────────
//
//  lut_lo and lut_hi stay live in 8 NEON registers across the entire row loop.
//  They are loaded once per generation in the worker and passed here.
//
//  Per 16 cells (~9 instructions):
//    vshrq  — centre adult flag (0 or 1)
//    vsub   — A = V - centre_flag
//    vshl   — state << 5
//    vorr   — key = (state<<5) | A
//    vqtbl4 — lookup in lut_lo (returns 0 for key >= 64)
//    vsub   — key - 64
//    vqtbl4 — lookup in lut_hi (returns 0 for key-64 >= 64, i.e. original key < 64)
//    vorr   — combine (exactly one half is non-zero per lane)
//    vst1   — store

static inline void apply_rules(
    const uint8_t* __restrict__ cur_row,
    const uint8_t* __restrict__ V,
    uint8_t* __restrict__       nxt_row,
    int N,
    uint8x16x4_t lut_lo,
    uint8x16x4_t lut_hi)
{
    const uint8x16_t k3  = vdupq_n_u8(3u);
    const uint8x16_t k64 = vdupq_n_u8(64u);

    int x = 0;
    for (; x + 16 <= N; x += 16) {
        uint8x16_t c = vld1q_u8(cur_row + x);

        // A = 5×5 adult sum − 1 if centre cell is adult (it was counted in V)
        uint8x16_t A = vsubq_u8(vld1q_u8(V + x),
                                 vshrq_n_u8(vceqq_u8(c, k3), 7));

        // key ∈ [0,127]: bits 6:5 = state, bits 4:0 = A
        uint8x16_t key = vorrq_u8(vshlq_n_u8(c, 5), A);

        // Dual-half lookup.  vqtbl4q_u8 returns 0 for index >= 64.
        vst1q_u8(nxt_row + x,
            vorrq_u8(vqtbl4q_u8(lut_lo, key),
                     vqtbl4q_u8(lut_hi, vsubq_u8(key, k64))));
    }
    // Scalar tail (rare: only if N not a multiple of 16)
    for (; x < N; ++x) {
        uint8_t c = cur_row[x];
        uint8_t A = (uint8_t)(V[x] - (c == 3u ? 1u : 0u));
        nxt_row[x] = rule_lut[(c << 5) | A];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Worker thread
// ─────────────────────────────────────────────────────────────────────────────

struct WorkCtx {
    uint8_t* grids[2];   // two N×N byte grids (alternating)
    int      N;
    int      generations;
};

static void worker(int tid, int nthreads, WorkCtx* wc, std::barrier<>* bar)
{
    pin_thread(tid); // pin thread

    const int N = wc->N;
    const int base    = N / nthreads;
    const int rem     = N % nthreads;
    const int y_start = tid * base + std::min(tid, rem);
    const int y_end   = y_start + base + (tid < rem ? 1 : 0);

    // Thread-local storage (no ap buffer any more):
    //   ring_buf : 8 × N bytes   (~256 KiB for N=32768, fits in L2)
    //   V        : N bytes       (~32  KiB for N=32768, fits in L1)
    std::vector<uint8_t> ring_buf((size_t)RING * N, 0u);
    std::vector<uint8_t> V(N, 0u);

    auto ring_slot = [&](int d) -> uint8_t* {
        return ring_buf.data() + (size_t)(d & RMASK) * N;
    };

    // Load both LUT halves into NEON registers once.
    // 8 registers total — they stay allocated by the compiler across the row loop.
    const uint8x16x4_t lut_lo = vld1q_u8_x4(rule_lut);
    const uint8x16x4_t lut_hi = vld1q_u8_x4(rule_lut + 64);

    for (int gen = 0; gen < wc->generations; ++gen) {
        const uint8_t* cur = wc->grids[ gen      & 1];
        uint8_t*       nxt = wc->grids[(gen + 1) & 1];

        const int Nmask = N - 1;   // N is power of 2
        // ── Seed V for this thread's starting row ────────────────────────
        std::fill(V.begin(), V.end(), 0u);
        for (int d = y_start - 2; d <= y_start + 2; ++d) {
            int gr = d & Nmask;    // works for negative d in two's complement
            uint8_t* slot = ring_slot(d);
            compute_row_sum(cur + (size_t)gr * N, slot, N);
            add_rows(V.data(), slot, N);
        }

        // ── Sliding window ───────────────────────────────────────────────
        for (int y = y_start; y < y_end; ++y) {
            // Apply rules for this row
            apply_rules(cur + (size_t)y * N, V.data(),
                        nxt + (size_t)y * N, N, lut_lo, lut_hi);

            // Compute row_sum for the row entering the window (y+3)
            int      new_gr = (y + 3) & Nmask;
            uint8_t* rs_new = ring_slot(y + 3);
            compute_row_sum(cur + (size_t)new_gr * N, rs_new, N);

            // Slide V: subtract outgoing row (y-2), add incoming row (y+3)
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

        constexpr size_t HPAGE = 2 * 1024 * 1024;

        if (posix_memalign(&p, HPAGE, n) != 0 || !p) {
            std::fprintf(stderr, "Fatal: alloc failed\n");
            std::exit(1);
        }

        if (madvise(p, n, MADV_HUGEPAGE) != 0) {
            std::perror("madvise");
        }

        return static_cast<uint8_t*>(p);
    };

    uint8_t* grid_a = alloc(CELLS);
    uint8_t* grid_b = alloc(CELLS);

    // touch pages
    std::memset(grid_a, 0, CELLS);
    std::memset(grid_b, 0, CELLS);

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
        (std::fwrite(&width,  8, 1,     fout) == 1) &&
        (std::fwrite(&height, 8, 1,     fout) == 1) &&
        (std::fwrite(result,  1, CELLS, fout) == CELLS);
    std::fclose(fout);
    if (!ok) { std::fprintf(stderr, "Error: write error\n"); return 6; }

    std::free(grid_a);
    std::free(grid_b);
    return 0;
}