// spawn_sim_sw_arm.cpp  —  Monster Spawning Grid, ARM NEON, sliding-window V.
//
// KEY DIFFERENCE FROM THE TWO-PASS VERSION
// ─────────────────────────────────────────
// Previously, Pass 2 recomputed the 5×5 neighbourhood sum from scratch for
// every row by loading five separate row_sum rows:
//
//   A[x] = rs[y-2][x] + rs[y-1][x] + rs[y][x] + rs[y+1][x] + rs[y+2][x]
//
// That accessed each row_sum row 5 times, producing ~7 × N² bytes of
// memory traffic per generation — most of it missing the L2 cache.
//
// HERE we maintain an array V[x] that holds the current 5×5 sum and update
// it *incrementally* as the window slides one row at a time:
//
//   V_new[x] = V_old[x]  −  rs[y−2][x]  +  rs[y+3][x]
//              └──────┘    └──────────┘    └──────────┘
//            keep 4 rows   drop oldest      add newest
//
// The row_sums themselves live in a thread-local RING-buffer of 8 rows
// (8 × N bytes = 256 KiB for N=32768), which fits entirely in L2.
// V[x] (N bytes = 32 KiB for N=32768) stays hot in L1.
//
// MEMORY TRAFFIC (per generation, N=32768)
// ─────────────────────────────────────────
//  Previous two-pass:   ~9 × N²  bytes of DRAM traffic
//  This version:        ~2 × N²  bytes  (read cur once, write nxt once)
//
// SYNCHRONISATION
// ───────────────
// The two-pass version needed 2 barriers per generation (end of Pass 1,
// end of Pass 2).  Here each thread works independently through all rows
// of its stripe — no cross-thread dependency within a generation — so
// only 1 barrier per generation is needed.
//
// RING BUFFER DESIGN
// ──────────────────
// RING = 8  (next power-of-two above 5, enabling cheap modulo via & 7).
// Slot assignment:  row with *logical index* d  →  ring_buf[(d & 7) * N].
// "Logical index" runs from y_start-2 upward, never wrapping (just for
// indexing the ring).  The actual grid row read is ((d % N) + N) % N.
//
// At step y we write slot (y+3)&7 and read slot (y-2)&7.
// These are always 5 apart, so they never alias with RING=8.
//
// SIMD:  ARM NEON, 128-bit, 16 × uint8_t per register.

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
static constexpr int RING     = 8;   // ring-buffer size; must be > 5 and a power of 2
static constexpr int RMASK    = RING - 1;

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 1 helpers  (unchanged from two-pass version)
// ─────────────────────────────────────────────────────────────────────────────

// Build adult_padded[0..N+3]:  2 wrap-around cells + adult bits (0/1) + 2 wrap.
static inline void build_adult_padded(const uint8_t* __restrict__ gr,
                                       uint8_t* __restrict__       ap,
                                       int N)
{
    ap[0]   = (gr[N-2] == 3u) ? 1u : 0u;
    ap[1]   = (gr[N-1] == 3u) ? 1u : 0u;
    ap[N+2] = (gr[0]   == 3u) ? 1u : 0u;
    ap[N+3] = (gr[1]   == 3u) ? 1u : 0u;

    const uint8x16_t v3 = vdupq_n_u8(3u);
    const uint8x16_t v1 = vdupq_n_u8(1u);
    int x = 0;
    for (; x + 16 <= N; x += 16)
        vst1q_u8(ap + 2 + x, vandq_u8(vceqq_u8(vld1q_u8(gr + x), v3), v1));
    for (; x < N; ++x)
        ap[2 + x] = (gr[x] == 3u) ? 1u : 0u;
}

// Horizontal 5-wide sliding window sum (row_sum[x] = sum of 5 consecutive adults).
static inline void sliding_sum5(const uint8_t* __restrict__ ap,
                                  uint8_t* __restrict__       rs,
                                  int N)
{
    int x = 0;
    for (; x + 16 <= N; x += 16) {
        uint8x16_t s =
            vaddq_u8(
                vaddq_u8(vld1q_u8(ap + x), vld1q_u8(ap + x + 1)),
                vaddq_u8(
                    vaddq_u8(vld1q_u8(ap + x + 2), vld1q_u8(ap + x + 3)),
                    vld1q_u8(ap + x + 4)));
        vst1q_u8(rs + x, s);
    }
    for (; x < N; ++x)
        rs[x] = ap[x] + ap[x+1] + ap[x+2] + ap[x+3] + ap[x+4];
}

// ─────────────────────────────────────────────────────────────────────────────
//  V maintenance helpers  (NEW in this version)
// ─────────────────────────────────────────────────────────────────────────────

// V[x] += rs[x]  —  used during ring-buffer initialisation (5 calls, once per gen).
static inline void add_rows(uint8_t* __restrict__ V,
                             const uint8_t* __restrict__ rs, int N)
{
    int x = 0;
    for (; x + 16 <= N; x += 16)
        vst1q_u8(V + x, vaddq_u8(vld1q_u8(V + x), vld1q_u8(rs + x)));
    for (; x < N; ++x)
        V[x] = (uint8_t)(V[x] + rs[x]);
}

// Slide V one row:  V[x] = V[x] − rs_old[x] + rs_new[x].
//
// Correctness of subtraction:
//   V[x] = rs[y-2]+rs[y-1]+rs[y]+rs[y+1]+rs[y+2], all ∈ [0,5].
//   rs_old = rs[y-2].  V[x] ≥ rs_old[x] always (V contains rs_old as one term).
//   So V[x] − rs_old[x] ≥ 0; no uint8 underflow.
static inline void slide_V(uint8_t* __restrict__       V,
                            const uint8_t* __restrict__ rs_new,
                            const uint8_t* __restrict__ rs_old,
                            int N)
{
    int x = 0;
    for (; x + 16 <= N; x += 16) {
        uint8x16_t v = vld1q_u8(V       + x);
        uint8x16_t n = vld1q_u8(rs_new  + x);
        uint8x16_t o = vld1q_u8(rs_old  + x);
        // subtract first (safe: no underflow), then add
        vst1q_u8(V + x, vaddq_u8(vsubq_u8(v, o), n));
    }
    for (; x < N; ++x)
        V[x] = (uint8_t)(V[x] - rs_old[x] + rs_new[x]);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rule application  (takes the pre-built V directly — 1 load vs 5 before)
// ─────────────────────────────────────────────────────────────────────────────

// V[x] is the full 5×5 neighbour-count INCLUDING the centre cell.
// We subtract the centre adult flag inside this function (same as before).
static inline void apply_rules(const uint8_t* __restrict__ cur_row,
                                const uint8_t* __restrict__ V,
                                uint8_t* __restrict__       nxt_row,
                                int N)
{
    const uint8x16_t k1 = vdupq_n_u8(1u), k2 = vdupq_n_u8(2u), k3 = vdupq_n_u8(3u);
    const uint8x16_t k4 = vdupq_n_u8(4u), k5 = vdupq_n_u8(5u), k9 = vdupq_n_u8(9u);

    int x = 0;
    for (; x + 16 <= N; x += 16) {
        uint8x16_t A = vld1q_u8(V       + x);   // 5×5 sum (centre included)
        uint8x16_t c = vld1q_u8(cur_row + x);

        // Subtract centre adult: (c == 3) → 0xFF; AND 1 → 0 or 1
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
    uint8_t* grids[2];   // ping-pong buffers; grids[0] = initial data
    int      N;
    int      generations;
    // NOTE: no global row_sums array — it has been eliminated entirely.
    //       Each thread maintains its own ring buffer.
};

static void worker(int tid, int nthreads, WorkCtx* wc, std::barrier<>* bar)
{
    const int N = wc->N;

    const int base    = N / nthreads;
    const int rem     = N % nthreads;
    const int y_start = tid * base + std::min(tid, rem);
    const int y_end   = y_start + base + (tid < rem ? 1 : 0);

    // Thread-local storage.  All fits in L2 for N ≤ 32768:
    //   ring_buf: 8 × 32 KiB = 256 KiB  (8 rows of row_sums)
    //   V:        1 ×  32 KiB = 32 KiB  (current 5×5 running sum)
    //   ap:       ≈   32 KiB            (padded adult row, intermediate)
    std::vector<uint8_t> ring_buf((size_t)RING * N, 0u);
    std::vector<uint8_t> V  (N, 0u);
    std::vector<uint8_t> ap (static_cast<size_t>(N) + 4, 0u);

    // Convenience: pointer to ring slot for *logical* row index d.
    // Uses bitwise AND instead of % since RING is a power of two.
    // Negative d works correctly in C++ (two's complement): (-2) & 7 == 6.
    auto ring_slot = [&](int d) -> uint8_t* {
        return ring_buf.data() + (size_t)(d & RMASK) * N;
    };

    for (int gen = 0; gen < wc->generations; ++gen) {
        const uint8_t* cur = wc->grids[ gen      & 1];
        uint8_t*       nxt = wc->grids[(gen + 1) & 1];

        // ── Initialise ring buffer and V for the first window ────────────
        //
        // Logical rows d = y_start-2 … y_start+2 map to grid rows
        // ((d % N) + N) % N  (handles negative d for the toroidal wrap).
        std::fill(V.begin(), V.end(), 0u);

        for (int d = y_start - 2; d <= y_start + 2; ++d) {
            int gr = ((d % N) + N) % N;                  // actual grid row
            build_adult_padded(cur + (size_t)gr * N, ap.data(), N);
            uint8_t* slot = ring_slot(d);
            sliding_sum5(ap.data(), slot, N);
            add_rows(V.data(), slot, N);                  // V accumulates 5 row_sums
        }
        // V[x] now equals rs[y_start-2][x] + … + rs[y_start+2][x]

        // ── Sliding loop: process every row in the stripe ────────────────
        for (int y = y_start; y < y_end; ++y) {

            // 1. Apply rules — V is the complete 5×5 sum for row y.
            apply_rules(cur + (size_t)y * N, V.data(),
                        nxt + (size_t)y * N, N);

            // 2. Compute row_sum for the row that enters the window next
            //    (logical row y+3, grid row (y+3) % N).
            int  new_gr   = (y + 3) % N;
            uint8_t* rs_new = ring_slot(y + 3);          // overwrites slot (y+3)&7
            build_adult_padded(cur + (size_t)new_gr * N, ap.data(), N);
            sliding_sum5(ap.data(), rs_new, N);

            // 3. Slide V:  drop rs[y-2], add rs[y+3].
            //    After this, V holds rs[y-1]+…+rs[y+3], ready for row y+1.
            //
            //    Ring slot safety:  (y+3)&7  vs  (y-2)&7  always differ
            //    because 3 ≢ −2 (mod 8), so rs_new and rs_old never alias.
            slide_V(V.data(), rs_new, ring_slot(y - 2), N);
        }

        // ── One barrier per generation (down from two in the two-pass version)
        //    All threads must finish writing nxt before the next gen reads it.
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
    // No row_sums array — 1 GiB saved compared to the two-pass version.

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