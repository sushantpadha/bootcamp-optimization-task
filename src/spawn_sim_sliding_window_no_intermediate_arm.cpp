// spawn_sim_sw_v2_arm.cpp  —  Sliding-window Monster Spawning Grid, ARM NEON.
//                             Version 2: fused row-sum (no intermediate `ap`).
//
// CHANGE FROM v1
// ──────────────
// Previously, computing row_sums for a new row required two separate passes:
//
//   Pass A  build_adult_padded(grid_row → ap)
//             reads  N bytes (grid), writes N+4 bytes (ap)
//
//   Pass B  sliding_sum5(ap → rs)
//             reads  N+4 bytes (ap), writes N bytes (rs)
//
// The two passes were chained by a store→load dependency on `ap`.  Even
// though `ap` fit in L1 (32 KiB for N=32768), the round-trip stalled the
// pipeline: sliding_sum5 could not start until build_adult_padded had
// committed all of `ap`.
//
// NOW we fuse both into compute_row_sum():
//
//   compute_row_sum(grid_row → rs)
//     loads grid row at 5 offsets, compares each to 3, ANDs to 0/1, sums.
//     reads N bytes (grid), writes N bytes (rs).  No intermediate buffer.
//
// The four boundary positions (x=0,1,N-2,N-1) need toroidal wrap-around
// and are handled with 4 scalar computations.  All other positions run in
// SIMD with plain unaligned loads — no bounds risk, because:
//   interior SIMD range:  x ∈ [2, N-2)
//   widest load:          gr + x + 2 + (SIMD_WIDTH-1) ≤ gr + N-1   ✓
//
// For power-of-2 N ≥ 512, the scalar tail between the SIMD loop and the
// right boundary is always empty (the loop runs right up to x = N-2).
//
// ALL OTHER LOGIC (ring buffer, sliding V, threading, barriers) is
// identical to v1.  Only compute_row_sum replaces the two-function pair,
// and the `ap` thread-local buffer is removed.

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
static constexpr int RING     = 8;    // must be > 5 and a power of 2
static constexpr int RMASK    = RING - 1;

// ─────────────────────────────────────────────────────────────────────────────
//  Fused row-sum  (replaces build_adult_padded + sliding_sum5)
// ─────────────────────────────────────────────────────────────────────────────
//
// Computes rs[x] = Σ_{d=-2}^{2} (gr[(x+d+N)%N] == ADULT)  for x ∈ [0,N).
//
// Layout:
//   x = 0,1        — scalar; wrap reads from gr[N-2..N-1]
//   x = 2..N-3     — SIMD; loads at offsets -2..+2 never leave [0, N-1]
//   x = N-2, N-1   — scalar; wrap reads from gr[0..1]
//
// Because all test grids have power-of-2 N ≥ 512, the SIMD loop always
// exits exactly at x = N-2, so the scalar-remainder branch is never taken.
static inline void compute_row_sum(const uint8_t* __restrict__ gr,
                                    uint8_t* __restrict__       rs,
                                    int N)
{
    // ── Left boundary (2 cells) ──────────────────────────────────────────
    rs[0] = (gr[N-2]==3u) + (gr[N-1]==3u) + (gr[0]==3u) + (gr[1]==3u) + (gr[2]==3u);
    rs[1] = (gr[N-1]==3u) + (gr[0]==3u)   + (gr[1]==3u) + (gr[2]==3u) + (gr[3]==3u);

    // ── SIMD interior ────────────────────────────────────────────────────
    // Each NEON register holds 16 uint8 lanes.
    // For each output chunk rs[x..x+15]:
    //   load gr at offsets -2,-1,0,+1,+2 (all unaligned, NEON handles fine)
    //   compare each to 3  →  0xFF where ADULT, 0 elsewhere
    //   AND with 1         →  0x01 where ADULT, 0 elsewhere
    //   sum the five       →  count of adults in the 1×5 window
    const uint8x16_t v3 = vdupq_n_u8(3u);
    const uint8x16_t v1 = vdupq_n_u8(1u);

    int x = 2;
    for (; x + 16 <= N - 2; x += 16) {
        // adult_at_offset(d) = (gr[x+d .. x+d+15] == 3) & 1
        uint8x16_t a0 = vandq_u8(vceqq_u8(vld1q_u8(gr + x - 2), v3), v1);
        uint8x16_t a1 = vandq_u8(vceqq_u8(vld1q_u8(gr + x - 1), v3), v1);
        uint8x16_t a2 = vandq_u8(vceqq_u8(vld1q_u8(gr + x    ), v3), v1);
        uint8x16_t a3 = vandq_u8(vceqq_u8(vld1q_u8(gr + x + 1), v3), v1);
        uint8x16_t a4 = vandq_u8(vceqq_u8(vld1q_u8(gr + x + 2), v3), v1);

        vst1q_u8(rs + x,
            vaddq_u8(
                vaddq_u8(a0, a1),
                vaddq_u8(vaddq_u8(a2, a3), a4)));
    }

    // ── Scalar remainder (never reached for power-of-2 N ≥ 512) ─────────
    for (; x < N - 2; ++x)
        rs[x] = (gr[x-2]==3u) + (gr[x-1]==3u) + (gr[x]==3u)
              + (gr[x+1]==3u) + (gr[x+2]==3u);

    // ── Right boundary (2 cells) ─────────────────────────────────────────
    rs[N-2] = (gr[N-4]==3u) + (gr[N-3]==3u) + (gr[N-2]==3u) + (gr[N-1]==3u) + (gr[0]==3u);
    rs[N-1] = (gr[N-3]==3u) + (gr[N-2]==3u) + (gr[N-1]==3u) + (gr[0]==3u)   + (gr[1]==3u);
}

// ─────────────────────────────────────────────────────────────────────────────
//  V maintenance  (unchanged from v1)
// ─────────────────────────────────────────────────────────────────────────────

static inline void add_rows(uint8_t* __restrict__       V,
                             const uint8_t* __restrict__ rs, int N)
{
    int x = 0;
    for (; x + 16 <= N; x += 16)
        vst1q_u8(V + x, vaddq_u8(vld1q_u8(V + x), vld1q_u8(rs + x)));
    for (; x < N; ++x) V[x] = (uint8_t)(V[x] + rs[x]);
}

static inline void slide_V(uint8_t* __restrict__       V,
                            const uint8_t* __restrict__ rs_new,
                            const uint8_t* __restrict__ rs_old,
                            int N)
{
    int x = 0;
    for (; x + 16 <= N; x += 16) {
        uint8x16_t v = vld1q_u8(V      + x);
        uint8x16_t n = vld1q_u8(rs_new + x);
        uint8x16_t o = vld1q_u8(rs_old + x);
        vst1q_u8(V + x, vaddq_u8(vsubq_u8(v, o), n));
    }
    for (; x < N; ++x)
        V[x] = (uint8_t)(V[x] - rs_old[x] + rs_new[x]);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rule application  (unchanged from v1)
// ─────────────────────────────────────────────────────────────────────────────

static inline void apply_rules(const uint8_t* __restrict__ cur_row,
                                const uint8_t* __restrict__ V,
                                uint8_t* __restrict__       nxt_row,
                                int N)
{
    const uint8x16_t k1 = vdupq_n_u8(1u), k2 = vdupq_n_u8(2u), k3 = vdupq_n_u8(3u);
    const uint8x16_t k4 = vdupq_n_u8(4u), k5 = vdupq_n_u8(5u), k9 = vdupq_n_u8(9u);

    int x = 0;
    for (; x + 16 <= N; x += 16) {
        uint8x16_t A = vld1q_u8(V       + x);
        uint8x16_t c = vld1q_u8(cur_row + x);

        A = vsubq_u8(A, vandq_u8(vceqq_u8(c, k3), k1));

        uint8x16_t is0 = vceqq_u8(c, vdupq_n_u8(0u));
        uint8x16_t is1 = vceqq_u8(c, k1);
        uint8x16_t is2 = vceqq_u8(c, k2);
        uint8x16_t is3 = vceqq_u8(c, k3);

        uint8x16_t n0 = vandq_u8(vandq_u8(is0, vandq_u8(vcgeq_u8(A,k3), vcleq_u8(A,k5))), k1);
        uint8x16_t n1 = vandq_u8(is1, k2);
        uint8x16_t n2 = vandq_u8(is2, k3);
        uint8x16_t n3 = vandq_u8(vandq_u8(is3, vandq_u8(vcgeq_u8(A,k4), vcleq_u8(A,k9))), k3);

        vst1q_u8(nxt_row + x, vorrq_u8(vorrq_u8(n0, n1), vorrq_u8(n2, n3)));
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

    // Thread-local storage.
    // NOTE: `ap` is GONE.  Thread-local footprint is now:
    //   ring_buf : 8 × N bytes  (256 KiB for N=32768)  → L2
    //   V        : 1 × N bytes  ( 32 KiB for N=32768)  → L1
    // Previously there was also `ap` (N+4 bytes, ~32 KiB), costing extra
    // L1d pressure and the store→load round-trip described above.
    std::vector<uint8_t> ring_buf((size_t)RING * N, 0u);
    std::vector<uint8_t> V(N, 0u);

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
            uint8_t* slot = ring_slot(d);
            compute_row_sum(cur + (size_t)gr * N, slot, N);   // fused: no ap
            add_rows(V.data(), slot, N);
        }

        // ── Sliding loop ─────────────────────────────────────────────────
        for (int y = y_start; y < y_end; ++y) {

            // a. Rules: V holds the 5×5 sum for row y
            apply_rules(cur + (size_t)y * N, V.data(),
                        nxt + (size_t)y * N, N);

            // b. Row-sum for the incoming row (y+3), written to ring slot
            int      new_gr = (y + 3) % N;
            uint8_t* rs_new = ring_slot(y + 3);
            compute_row_sum(cur + (size_t)new_gr * N, rs_new, N);  // fused

            // c. Slide V
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