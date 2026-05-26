// spawn_sim.cpp — Fast Monster Spawning Grid simulator.
//
// ALGORITHM: Two-pass row-prefix-sum
// ─────────────────────────────────
// The reference counts 24 scattered neighbours per cell.  We replace that
// with two cheap sequential passes:
//
//  Pass 1 (parallel over rows):
//    For every row y, build a padded adult-bit row:
//      adult_padded[0..1]    = (grid[y][N-2..N-1] == ADULT)  — left wrap
//      adult_padded[2..N+1]  = (grid[y][0..N-1]   == ADULT)
//      adult_padded[N+2..N+3]= (grid[y][0..1]     == ADULT)  — right wrap
//    Then compute a horizontal 5-wide sliding window sum:
//      row_sums[y][x] = Σ adult_padded[x..x+4]              (0 ≤ x < N)
//    This counts adults in columns [x-2, x+2] of row y, wrapping toroidally.
//
//  Pass 2 (parallel over rows):
//    For row y, sum five consecutive row_sums rows:
//      A[x] = row_sums[y-2][x] + … + row_sums[y+2][x]
//    That is the full 5×5-block adult count (including the centre cell).
//    Subtract 1 if the centre cell itself is ADULT, giving the correct A.
//    Apply the four-state transition rule and write to the next grid.
//
//  A std::barrier synchronises between the two passes.
//  Each generation requires exactly two barrier phases across 8 threads.
//
// MEMORY LAYOUT
// ─────────────
//  grids[2]  : two N×N uint8_t arrays, ping-pong buffered (64-byte aligned).
//  row_sums  : N×N uint8_t, reused every generation        (64-byte aligned).
//  ap        : thread-local scratch, size N+4 bytes (one padded adult row).
//
//  For N=32768: grids[0]+grids[1]+row_sums ≈ 3 GiB — fits in 16 GiB.
//
// SIMD
// ────
//  All hot loops use ARM NEON (128-bit, 16×uint8_t per register).
//  Scalar tail handles any non-multiple-of-16 remainder (all test sizes
//  are powers of 2 ≥ 512, so the tail is never exercised in practice).
//
// THREADING
// ─────────
//  8 std::threads (matching c8g.2xlarge's 8 vCPUs, no SMT).
//  Rows are divided into contiguous stripes; stripe boundaries are aligned
//  to whole rows to avoid false sharing on row_sums.
//  The main thread itself participates as thread 0.
//
// OUTPUT
// ──────
//  Prints ONLY the simulation wall-clock time (ms) to stdout.
//  All diagnostics go to stderr.

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

// ─────────────────────────────────────────────────────────────────────────────
//  Tuning knobs
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int NTHREADS = 8;   // must match vCPU count on c8g.2xlarge

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 1 helpers
// ─────────────────────────────────────────────────────────────────────────────

// Populate ap[0..N+3] with adult bits (0 or 1) for grid row `gr[0..N-1]`,
// prepending/appending 2 wrap-around cells so that the sliding window in
// sliding_sum5() needs no bounds checks.
//
//  Layout:  [ adult(gr[N-2]) | adult(gr[N-1]) | adult(gr[0..N-1]) | adult(gr[0]) | adult(gr[1]) ]
//            ╰──── ap[0] ───╯ ╰──── ap[1] ───╯ ╰───── ap[2..N+1] ─────╯ ╰─ap[N+2]─╯ ╰─ap[N+3]─╯
static inline void build_adult_padded(const uint8_t* __restrict__ gr,
                                       uint8_t* __restrict__       ap,
                                       int N)
{
    // Scalar boundary cells (only 4, always fast)
    ap[0]   = (gr[N - 2] == 3u) ? 1u : 0u;
    ap[1]   = (gr[N - 1] == 3u) ? 1u : 0u;
    ap[N+2] = (gr[0]     == 3u) ? 1u : 0u;
    ap[N+3] = (gr[1]     == 3u) ? 1u : 0u;

    // SIMD body: convert 16 cells at a time to adult bits 0/1.
    // vceqq_u8(v, 3) → 0xFF where cell==ADULT, 0 elsewhere.
    // vandq_u8(…, 1)  → 0x01 where adult.
    const uint8x16_t v3 = vdupq_n_u8(3u);
    const uint8x16_t v1 = vdupq_n_u8(1u);
    int x = 0;
    for (; x + 16 <= N; x += 16) {
        uint8x16_t cells = vld1q_u8(gr + x);
        vst1q_u8(ap + 2 + x, vandq_u8(vceqq_u8(cells, v3), v1));
    }
    for (; x < N; ++x)                            // scalar tail (never hit for power-of-2 N≥512)
        ap[2 + x] = (gr[x] == 3u) ? 1u : 0u;
}

// Compute the horizontal 5-wide sliding window sum from the padded adult array.
// rs[x] = ap[x] + ap[x+1] + ap[x+2] + ap[x+3] + ap[x+4]  for x in [0, N).
// ap has size N+4, so all accesses are in-bounds.
// Result fits in uint8_t (maximum value: 5 adults × 1 = 5).
static inline void sliding_sum5(const uint8_t* __restrict__ ap,
                                  uint8_t* __restrict__       rs,
                                  int N)
{
    int x = 0;
    for (; x + 16 <= N; x += 16) {
        // Five loads at offsets 0..4; unaligned but NEON handles this natively.
        uint8x16_t s =
            vaddq_u8(
                vaddq_u8(vld1q_u8(ap + x),     vld1q_u8(ap + x + 1)),
                vaddq_u8(
                    vaddq_u8(vld1q_u8(ap + x + 2), vld1q_u8(ap + x + 3)),
                    vld1q_u8(ap + x + 4)));
        vst1q_u8(rs + x, s);
    }
    for (; x < N; ++x)
        rs[x] = ap[x] + ap[x+1] + ap[x+2] + ap[x+3] + ap[x+4];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pass 2: vertical sum of five row_sum rows → A, subtract centre, apply rules
// ─────────────────────────────────────────────────────────────────────────────

// Transition logic (from the spec):
//   EMPTY    (0) → EGG      (1) if 3 ≤ A ≤ 5,  else EMPTY (0)
//   EGG      (1) → JUVENILE (2) unconditionally
//   JUVENILE (2) → ADULT    (3) unconditionally
//   ADULT    (3) → ADULT    (3) if 4 ≤ A ≤ 9,  else EMPTY (0)
//
// SIMD implementation:
//  1. Compute A by summing five row_sum vectors (vertical sum).
//  2. Subtract the centre cell's adult flag (A counts the centre, we don't want it).
//  3. For each of the four states, produce a mask (0xFF/0x00) with vceqq_u8.
//  4. AND each mask with the target value; OR the four contributions together.
//     Because exactly one state mask is 0xFF per lane, the OR is the final value.
//
// A ∈ [0, 24] after subtraction — fits safely in uint8_t.
static inline void apply_rules(
    const uint8_t* __restrict__ cur_row,   // current grid, row y
    const uint8_t* __restrict__ rs0,       // row_sums[y-2]
    const uint8_t* __restrict__ rs1,       // row_sums[y-1]
    const uint8_t* __restrict__ rs2,       // row_sums[y  ]
    const uint8_t* __restrict__ rs3,       // row_sums[y+1]
    const uint8_t* __restrict__ rs4,       // row_sums[y+2]
    uint8_t* __restrict__       nxt_row,   // next grid, row y
    int N)
{
    // Preload small integer constants (stay in registers across loop iterations)
    const uint8x16_t k1 = vdupq_n_u8(1u);
    const uint8x16_t k2 = vdupq_n_u8(2u);
    const uint8x16_t k3 = vdupq_n_u8(3u);
    const uint8x16_t k4 = vdupq_n_u8(4u);
    const uint8x16_t k5 = vdupq_n_u8(5u);
    const uint8x16_t k9 = vdupq_n_u8(9u);

    int x = 0;
    for (; x + 16 <= N; x += 16) {

        // ── Step 1: vertical sum of five row_sum lanes ──────────────
        uint8x16_t A =
            vaddq_u8(
                vaddq_u8(vld1q_u8(rs0 + x), vld1q_u8(rs1 + x)),
                vaddq_u8(
                    vaddq_u8(vld1q_u8(rs2 + x), vld1q_u8(rs3 + x)),
                    vld1q_u8(rs4 + x)));

        // ── Step 2: subtract centre cell's adult contribution ───────
        uint8x16_t c = vld1q_u8(cur_row + x);
        A = vsubq_u8(A, vandq_u8(vceqq_u8(c, k3), k1));  // -1 if ADULT, -0 otherwise

        // ── Step 3: per-state masks (0xFF where condition holds) ────
        uint8x16_t is_empty    = vceqq_u8(c, vdupq_n_u8(0u));   // c == EMPTY
        uint8x16_t is_egg      = vceqq_u8(c, k1);                // c == EGG
        uint8x16_t is_juvenile = vceqq_u8(c, k2);                // c == JUVENILE
        uint8x16_t is_adult    = vceqq_u8(c, k3);                // c == ADULT

        // ── Step 4: next-state contributions ────────────────────────

        // EMPTY → EGG(1) iff 3 ≤ A ≤ 5; the AND chain propagates 0xFF → 0x01
        uint8x16_t spawn_cond = vandq_u8(vcgeq_u8(A, k3), vcleq_u8(A, k5));
        uint8x16_t n_empty    = vandq_u8(vandq_u8(is_empty, spawn_cond), k1);

        // EGG → JUVENILE(2)  always
        uint8x16_t n_egg      = vandq_u8(is_egg, k2);

        // JUVENILE → ADULT(3)  always
        uint8x16_t n_juvenile = vandq_u8(is_juvenile, k3);

        // ADULT → ADULT(3) iff 4 ≤ A ≤ 9; 0xFF & 3 → 0x03 where true
        uint8x16_t surv_cond  = vandq_u8(vcgeq_u8(A, k4), vcleq_u8(A, k9));
        uint8x16_t n_adult    = vandq_u8(vandq_u8(is_adult, surv_cond), k3);

        // Combine: exactly one of n_empty/n_egg/n_juvenile/n_adult is non-zero
        // per lane, so OR gives the correct next state.
        vst1q_u8(nxt_row + x, vorrq_u8(vorrq_u8(n_empty, n_egg),
                                        vorrq_u8(n_juvenile, n_adult)));
    }

    // Scalar tail (needed for correctness but never reached for N that are
    // powers of two ≥ 512, since 512 = 32 × 16)
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
//  Worker thread
// ─────────────────────────────────────────────────────────────────────────────

struct WorkCtx {
    uint8_t* grids[2];   // ping-pong grid buffers; grids[0] = initial data
    uint8_t* row_sums;   // N×N scratch: horizontal 5-sum per (row, col)
    int      N;
    int      generations;
};

// Each thread owns a contiguous stripe of rows: [y_start, y_end).
// The stripe never changes across generations — no work stealing needed.
//
// Within each generation:
//   Pass 1 → barrier → Pass 2 → barrier
//
// The barrier count equals NTHREADS; the main thread also calls worker(0,…)
// so all 8 arrive at each barrier.
static void worker(int tid, int nthreads, WorkCtx* wc, std::barrier<>* bar)
{
    const int N = wc->N;

    // Distribute rows as evenly as possible; first `rem` threads get one extra.
    const int base    = N / nthreads;
    const int rem     = N % nthreads;
    const int y_start = tid * base + std::min(tid, rem);
    const int y_end   = y_start + base + (tid < rem ? 1 : 0);

    // Thread-local padded adult row.  Allocated once, reused across all
    // generations and all rows in the stripe.  Heap to avoid stack blowup
    // for large N (N+4 can be up to 32772 bytes).
    std::vector<uint8_t> ap(static_cast<size_t>(N) + 4);

    for (int gen = 0; gen < wc->generations; ++gen) {
        const uint8_t* cur = wc->grids[gen       & 1];   // read from
        uint8_t*       nxt = wc->grids[(gen + 1) & 1];   // write to
        uint8_t*       rs  = wc->row_sums;

        // ── Pass 1: compute row_sums for this thread's stripe ─────────
        //
        // Each iteration is independent; memory access is sequential
        // (one row of `cur` read, one row of `rs` written).
        for (int y = y_start; y < y_end; ++y) {
            const uint8_t* gr = cur + static_cast<size_t>(y) * N;
            build_adult_padded(gr, ap.data(), N);
            sliding_sum5(ap.data(), rs + static_cast<size_t>(y) * N, N);
        }

        // ── Barrier 1: all row_sums rows must be visible before Pass 2 ─
        bar->arrive_and_wait();

        // ── Pass 2: vertical sum + rule application ───────────────────
        //
        // For row y we need row_sums of y-2..y+2 (toroidal wrap at 0 and N-1).
        // The % operator is cheap — N is always a power of two, so
        // alternatively we could use bitmasking, but this is clear and the
        // compiler may optimise the loop-invariant parts.
        for (int y = y_start; y < y_end; ++y) {
            const int y0 = (y - 2 + N) % N;
            const int y1 = (y - 1 + N) % N;
            // y  stays as-is
            const int y3 = (y + 1) % N;
            const int y4 = (y + 2) % N;

            apply_rules(
                cur + static_cast<size_t>(y)  * N,  // current state
                rs  + static_cast<size_t>(y0) * N,  // row_sums[y-2]
                rs  + static_cast<size_t>(y1) * N,  // row_sums[y-1]
                rs  + static_cast<size_t>(y)  * N,  // row_sums[y  ]
                rs  + static_cast<size_t>(y3) * N,  // row_sums[y+1]
                rs  + static_cast<size_t>(y4) * N,  // row_sums[y+2]
                nxt + static_cast<size_t>(y)  * N,  // next state
                N);
        }

        // ── Barrier 2: all of `nxt` must be written before next gen's Pass 1
        bar->arrive_and_wait();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  main — I/O, timer, thread management
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

    // 64-byte alignment ensures SIMD stores don't straddle cache lines and
    // each row start is cache-line-aligned for N that are multiples of 64.
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

    // One barrier shared by all NTHREADS participants (including main thread).
    std::barrier<> bar(NTHREADS);

    auto t0 = std::chrono::steady_clock::now();

    // Spawn NTHREADS-1 background threads; main thread runs as thread 0.
    std::vector<std::thread> threads;
    threads.reserve(NTHREADS - 1);
    for (int t = 1; t < NTHREADS; ++t)
        threads.emplace_back(worker, t, NTHREADS, &wc, &bar);

    worker(0, NTHREADS, &wc, &bar);   // main thread works

    for (auto& th : threads) th.join();

    auto t1 = std::chrono::steady_clock::now();

    // Print ONLY the elapsed simulation time to stdout (matches reference format)
    std::printf("%.3f ms\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    // ── Write output ─────────────────────────────────────────────────────────
    // After `generations` ping-pong steps, the result lives in grids[generations & 1].
    // Proof: gen g writes to grids[(g+1)&1]; the last iteration is g=generations-1,
    // writing to grids[generations & 1].
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