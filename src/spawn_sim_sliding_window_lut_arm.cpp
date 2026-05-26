// spawn_sim_lut_arm.cpp  —  Monster Spawning Grid, ARM NEON, LUT rules.
//
// CHANGES FROM PACKED VERSION
// ───────────────────────────
// 1. PACKING DROPPED
//    perf stat on N=512 showed IPC=3.47 (Graviton4 peak ~4) and cache-miss
//    rate 0.08%.  The algorithm is compute-bound, not memory-bound.  The
//    pack/unpack overhead (~36% of rule-application cost) was a net negative.
//    At large N (32768) the grid exceeds L3, but sequential streaming is
//    handled by the hardware prefetcher before packing would help.
//
// 2. LUT RULE APPLICATION  (main optimisation)
//    Before: ~19 NEON instructions per 16 cells
//      (4 × vceqq, 4 × range-check pairs, 3 × vorr, 2 × vand-mask)
//    After:  ~9 NEON instructions per 16 cells
//      (vsub centre, vshl key, vorr key, 2 × vqtbl4q_u8, vsub offset, vorr result)
//
//    Key encoding:  (state << 5) | A   →   [0, 127]
//      state ∈ {0,1,2,3},  A = adult neighbours, centre excluded, [0..24]
//    128-byte table split into two 64-byte halves:
//      lut_lo  (indices 0..63 )  — covers EMPTY (state 0) and EGG (state 1)
//      lut_hi  (indices 64..127) — covers JUV   (state 2) and ADULT (state 3)
//    vqtbl4q_u8 returns 0 for any index ≥ 64, so OR of the two lookups is
//    correct: exactly one half is non-zero per lane.
//    Both halves fit in 8 NEON registers total and stay live across the row.
//
// 3. FUSED compute_row_sum  (no ap buffer)
//    Previous versions materialised an N+4 byte adult-flag array, then read
//    it back for the sliding sum — a full N-byte write+read per row.
//    Now: load 5 overlapping windows of 16 bytes directly, adult-detect with
//    vshrq_n_u8(vceqq,7)  (2 instructions instead of 3), sum immediately.
//    The ap buffer is gone; thread-local memory drops to ring_buf + V only.
//
// WHAT IS UNCHANGED
// ─────────────────
//    Ring-buffer sliding window, std::barrier threading, add_rows, slide_V.

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

alignas(64) static uint8_t rule_lut[128];

static void build_lut()
{
    for (int s = 0; s < 4; ++s) {
        for (int a = 0; a < 32; ++a) {   // 32 slots per state; only a=0..24 reached
            uint8_t nx;
            switch (s) {
                case 0:  nx = (a >= 3 && a <= 5) ? 1u : 0u;          break;  // EMPTY
                case 1:  nx = 2u;                                      break;  // EGG → JUV
                case 2:  nx = 3u;                                      break;  // JUV → ADULT
                default: nx = (a >= 4 && a <= 9) ? 3u : 0u;           break;  // ADULT
            }
            rule_lut[(s << 5) | a] = nx;
        }
    }
}

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

    // Helper: adult flag at position x with toroidal wrap.
    // N is always a power of 2, so x & (N-1) replaces ((x%N)+N)%N.
    // Works for negative x in two's complement: e.g. -2 & (N-1) == N-2.
    const int mask = N - 1;
    auto a = [&](int x) -> uint8_t {
        return gr[x & mask] == 3u ? 1u : 0u;
    };

    // Boundary positions (involve wrap-around, handled scalar)
    rs[0]   = a(-2)+a(-1)+a(0)+a(1)+a(2);
    rs[1]   = a(-1)+a(0)+a(1)+a(2)+a(3);
    rs[N-2] = a(N-4)+a(N-3)+a(N-2)+a(N-1)+a(0);
    rs[N-1] = a(N-3)+a(N-2)+a(N-1)+a(0)+a(1);

    // SIMD interior: x in [2, N-2).
    // Loop guard x+16 <= N-2 ensures gr[x-2] >= gr[0] and gr[x+17] <= gr[N-1].
    int x = 2;
    for (; x + 16 <= N - 2; x += 16) {
        // adult_at(offset) = vshrq_n_u8(vceqq_u8(load, k3), 7) → 0 or 1 per byte
        #define A16(off) vshrq_n_u8(vceqq_u8(vld1q_u8(gr+x+(off)), k3), 7)
        uint8x16_t s = vaddq_u8(
            vaddq_u8(A16(-2), A16(-1)),
            vaddq_u8(vaddq_u8(A16(0), A16(1)), A16(2)));
        #undef A16
        vst1q_u8(rs + x, s);
    }
    // Scalar tail (at most 15 iterations for power-of-2 N)
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

    build_lut();

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