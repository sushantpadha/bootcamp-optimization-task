// spawn_sim_packed_arm.cpp  —  Monster Spawning Grid, ARM NEON, 2-bit packed grid.
//
// STORAGE CHANGE
// ──────────────
// Previous versions stored each cell as a uint8_t (1 byte), using only 2 of
// the 8 bits.  This version packs 4 cells into each byte:
//
//   bits [1:0]  cell at position 4k+0   (EMPTY=00, EGG=01, JUV=10, ADULT=11)
//   bits [3:2]  cell at position 4k+1
//   bits [5:4]  cell at position 4k+2
//   bits [7:6]  cell at position 4k+3
//
// Memory per grid:   N² / 4 bytes  (was N²)
// For N=32768:       256 MiB       (was 1 GiB)   → 4× reduction
//
// WHAT STAYS BYTE-VALUED
// ──────────────────────
// The ring buffer (row_sums, counts 0–5) and V (5×5 sums, counts 0–24)
// cannot be packed — they store integers, not 2-bit states.  Only `cur`
// and `nxt` are packed.
//
// PACK / UNPACK WITH NEON
// ───────────────────────
// Unpacking 16 packed bytes → 64 sequential state bytes:
//   1. Extract 4 stride-4 sub-arrays (shifts + masks)
//   2. vst4q_u8 interleaves them back to sequential order in one instruction
//
// Packing 64 sequential state bytes → 16 packed bytes:
//   1. vld4q_u8 de-interleaves into 4 stride-4 sub-arrays in one instruction
//   2. Shift each sub-array by 0/2/4/6 bits and OR together
//
// vld4q_u8 / vst4q_u8 are ARM structure load/store instructions that handle
// the 4-way interleave/de-interleave as a single hardware operation — they
// exist precisely for this kind of packed-array use case.
//
// UNCHANGED FROM v2
// ─────────────────
//   sliding_sum5, add_rows, slide_V, worker structure, threading, barriers.
//   `ap` buffer returns (N+4 bytes) for adult-byte expansion before sliding sum.

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
//  Core pack / unpack primitives
// ─────────────────────────────────────────────────────────────────────────────

// Unpack 16 packed bytes (64 cells, 2 bits each) → 64 sequential state bytes.
//
// Step 1: extract 4 "stride-4" sub-arrays using shifts and masks.
//   d.val[0][k] = state of cell 4k+0   (bits 1:0 of packed byte k)
//   d.val[1][k] = state of cell 4k+1   (bits 3:2)
//   d.val[2][k] = state of cell 4k+2   (bits 5:4)
//   d.val[3][k] = state of cell 4k+3   (bits 7:6)
// Step 2: vst4q_u8 writes them interleaved:
//   out = [cell0, cell1, cell2, cell3, cell4, cell5, ...] ✓
static inline void unpack64(const uint8_t* __restrict__ packed,
                              uint8_t* __restrict__       out)
{
    const uint8x16_t mask = vdupq_n_u8(0x03u);
    uint8x16_t p = vld1q_u8(packed);
    uint8x16x4_t d;
    d.val[0] = vandq_u8(p, mask);                 // cells 4k+0
    d.val[1] = vandq_u8(vshrq_n_u8(p, 2), mask);  // cells 4k+1
    d.val[2] = vandq_u8(vshrq_n_u8(p, 4), mask);  // cells 4k+2
    d.val[3] = vshrq_n_u8(p, 6);                  // cells 4k+3  (top 2 bits, no mask needed)
    vst4q_u8(out, d);   // hardware 4-way interleave → 64 sequential bytes
}

// Pack 64 sequential state bytes (values 0–3) → 16 packed bytes.
//
// vld4q_u8 is the inverse of vst4q_u8: it de-interleaves 64 bytes into
// 4 stride-4 sub-arrays.  Then shift and OR to build the packed bytes.
//   packed[k] = d.val[0][k]        (bits 1:0)
//             | d.val[1][k] << 2   (bits 3:2)
//             | d.val[2][k] << 4   (bits 5:4)
//             | d.val[3][k] << 6   (bits 7:6)
static inline void pack64(const uint8_t* __restrict__ states,
                            uint8_t* __restrict__       packed)
{
    uint8x16x4_t d = vld4q_u8(states);   // hardware 4-way de-interleave
    vst1q_u8(packed,
        vorrq_u8(
            vorrq_u8(d.val[0],               vshlq_n_u8(d.val[1], 2)),
            vorrq_u8(vshlq_n_u8(d.val[2], 4), vshlq_n_u8(d.val[3], 6))));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Row-sum computation  (reads packed grid, writes byte ring-buffer slot)
// ─────────────────────────────────────────────────────────────────────────────

// Build the adult_padded byte array from a packed grid row.
// Reads N/4 packed bytes, writes N+4 adult bytes (0 or 1) with wrap-around.
//
// Uses the same vst4q_u8 trick as unpack64, but instead of storing full
// states, it stores adult flags (state==3 ? 1 : 0) interleaved sequentially.
static inline void unpack_adult_padded(const uint8_t* __restrict__ packed_row,
                                        uint8_t* __restrict__       ap,
                                        int N)
{
    // Scalar boundary (4 cells with toroidal wrap)
    auto adult_at = [&](int x) -> uint8_t {
        return ((packed_row[x >> 2] >> ((x & 3) * 2)) & 0x03u) == 3u ? 1u : 0u;
    };
    ap[0]   = adult_at(N - 2);
    ap[1]   = adult_at(N - 1);
    ap[N+2] = adult_at(0);
    ap[N+3] = adult_at(1);

    const uint8x16_t mask = vdupq_n_u8(0x03u);
    const uint8x16_t k3   = vdupq_n_u8(3u);
    const uint8x16_t k1   = vdupq_n_u8(1u);

    int x = 0;
    for (; x + 64 <= N; x += 64) {
        uint8x16_t p = vld1q_u8(packed_row + (x >> 2));
        uint8x16x4_t d;
        // Extract state of each sub-array, compare to 3, clamp to 0/1
        d.val[0] = vandq_u8(vceqq_u8(vandq_u8(p, mask),                    k3), k1);
        d.val[1] = vandq_u8(vceqq_u8(vandq_u8(vshrq_n_u8(p, 2), mask),    k3), k1);
        d.val[2] = vandq_u8(vceqq_u8(vandq_u8(vshrq_n_u8(p, 4), mask),    k3), k1);
        d.val[3] = vandq_u8(vceqq_u8(vshrq_n_u8(p, 6),                     k3), k1);
        vst4q_u8(ap + 2 + x, d);   // interleave adult flags to sequential order
    }
    for (; x < N; ++x)
        ap[2 + x] = adult_at(x);
}

// Horizontal 5-wide sliding window sum (unchanged — works on byte arrays).
static inline void sliding_sum5(const uint8_t* __restrict__ ap,
                                  uint8_t* __restrict__       rs,
                                  int N)
{
    int x = 0;
    for (; x + 16 <= N; x += 16) {
        uint8x16_t s =
            vaddq_u8(
                vaddq_u8(vld1q_u8(ap+x), vld1q_u8(ap+x+1)),
                vaddq_u8(
                    vaddq_u8(vld1q_u8(ap+x+2), vld1q_u8(ap+x+3)),
                    vld1q_u8(ap+x+4)));
        vst1q_u8(rs + x, s);
    }
    for (; x < N; ++x)
        rs[x] = ap[x]+ap[x+1]+ap[x+2]+ap[x+3]+ap[x+4];
}

// ─────────────────────────────────────────────────────────────────────────────
//  V maintenance  (unchanged — byte arrays)
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
        vst1q_u8(V+x, vaddq_u8(vsubq_u8(vld1q_u8(V+x), vld1q_u8(rs_old+x)),
                                vld1q_u8(rs_new+x)));
    for (; x < N; ++x)
        V[x] = (uint8_t)(V[x] - rs_old[x] + rs_new[x]);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rule application  (reads packed cur, writes packed nxt)
// ─────────────────────────────────────────────────────────────────────────────

// Process 64 cells: unpack states, apply rules using V, pack results.
// The 64-cell granularity matches the 16-packed-byte unpack/pack primitives.
static inline void apply_rules_packed(
    const uint8_t* __restrict__ cur_packed,  // N/4 bytes
    const uint8_t* __restrict__ V,           // N bytes (5×5 sums, centre included)
    uint8_t* __restrict__       nxt_packed,  // N/4 bytes
    int N)
{
    const uint8x16_t k1=vdupq_n_u8(1),k2=vdupq_n_u8(2),k3=vdupq_n_u8(3);
    const uint8x16_t k4=vdupq_n_u8(4),k5=vdupq_n_u8(5),k9=vdupq_n_u8(9);

    // Stack buffers for one 64-cell chunk — hot in L1, never spill to DRAM.
    alignas(64) uint8_t states[64];
    alignas(64) uint8_t next_s[64];

    for (int x = 0; x < N; x += 64) {
        // Unpack: 16 packed bytes → 64 state bytes
        unpack64(cur_packed + (x >> 2), states);

        // Apply rules in 16-cell NEON chunks
        for (int i = 0; i < 64; i += 16) {
            uint8x16_t A = vld1q_u8(V + x + i);
            uint8x16_t c = vld1q_u8(states + i);

            // Subtract centre adult (A currently counts centre cell too)
            A = vsubq_u8(A, vandq_u8(vceqq_u8(c, k3), k1));

            uint8x16_t is0=vceqq_u8(c,vdupq_n_u8(0)), is1=vceqq_u8(c,k1);
            uint8x16_t is2=vceqq_u8(c,k2),             is3=vceqq_u8(c,k3);

            uint8x16_t n0=vandq_u8(vandq_u8(is0,vandq_u8(vcgeq_u8(A,k3),vcleq_u8(A,k5))),k1);
            uint8x16_t n1=vandq_u8(is1,k2);
            uint8x16_t n2=vandq_u8(is2,k3);
            uint8x16_t n3=vandq_u8(vandq_u8(is3,vandq_u8(vcgeq_u8(A,k4),vcleq_u8(A,k9))),k3);

            vst1q_u8(next_s + i, vorrq_u8(vorrq_u8(n0,n1),vorrq_u8(n2,n3)));
        }

        // Pack: 64 next-state bytes → 16 packed bytes
        pack64(next_s, nxt_packed + (x >> 2));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Worker thread
// ─────────────────────────────────────────────────────────────────────────────

struct WorkCtx {
    uint8_t* grids[2];   // packed grid buffers, each N²/4 bytes
    int      N;
    int      generations;
};

static void worker(int tid, int nthreads, WorkCtx* wc, std::barrier<>* bar)
{
    const int N    = wc->N;
    const int NPK  = N >> 2;   // packed bytes per row

    const int base    = N / nthreads;
    const int rem     = N % nthreads;
    const int y_start = tid * base + std::min(tid, rem);
    const int y_end   = y_start + base + (tid < rem ? 1 : 0);

    // Thread-local storage:
    //   ring_buf : 8 × N bytes  (byte-valued row_sums, 0–5)
    //   V        : N bytes      (byte-valued 5×5 sums,  0–24)
    //   ap       : N+4 bytes    (unpacked adult flags for sliding sum)
    // For N=32768: 256KiB + 32KiB + 32KiB = 320KiB → fits in L2.
    std::vector<uint8_t> ring_buf((size_t)RING * N, 0u);
    std::vector<uint8_t> V  (N, 0u);
    std::vector<uint8_t> ap (static_cast<size_t>(N) + 4, 0u);

    auto ring_slot = [&](int d) -> uint8_t* {
        return ring_buf.data() + (size_t)(d & RMASK) * N;
    };

    for (int gen = 0; gen < wc->generations; ++gen) {
        const uint8_t* cur = wc->grids[ gen      & 1];   // packed, N²/4 bytes
        uint8_t*       nxt = wc->grids[(gen + 1) & 1];

        // ── Initialise V ────────────────────────────────────────────────
        std::fill(V.begin(), V.end(), 0u);

        for (int d = y_start - 2; d <= y_start + 2; ++d) {
            int gr = ((d % N) + N) % N;
            uint8_t* slot = ring_slot(d);
            unpack_adult_padded(cur + (size_t)gr * NPK, ap.data(), N);
            sliding_sum5(ap.data(), slot, N);
            add_rows(V.data(), slot, N);
        }

        // ── Sliding loop ────────────────────────────────────────────────
        for (int y = y_start; y < y_end; ++y) {

            // Apply rules: reads packed cur row and V, writes packed nxt row
            apply_rules_packed(cur + (size_t)y * NPK, V.data(),
                               nxt + (size_t)y * NPK, N);

            // Compute row_sum for the row entering the window (y+3)
            int      new_gr = (y + 3) % N;
            uint8_t* rs_new = ring_slot(y + 3);
            unpack_adult_padded(cur + (size_t)new_gr * NPK, ap.data(), N);
            sliding_sum5(ap.data(), rs_new, N);

            // Slide V
            slide_V(V.data(), rs_new, ring_slot(y - 2), N);
        }

        bar->arrive_and_wait();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers: pack/unpack full N×N grid  (called only at I/O time, not per gen)
// ─────────────────────────────────────────────────────────────────────────────

// Convert N² byte grid → N²/4 packed grid.
static void pack_grid(const uint8_t* bytes, uint8_t* packed, size_t cells)
{
    const size_t packed_cells = cells >> 2;
    const uint8x16_t k1=vdupq_n_u8(1),k2=vdupq_n_u8(2);
    const uint8x16_t k4=vdupq_n_u8(4),k6=vdupq_n_u8(6);   // shift amounts (unused; use immediates)
    size_t i = 0;
    for (; i + 64 <= cells; i += 64) {
        // De-interleave 64 sequential state bytes into 4 stride-4 sub-arrays
        uint8x16x4_t d = vld4q_u8(bytes + i);
        // Reconstruct packed byte: d.val[0] | (d.val[1]<<2) | (d.val[2]<<4) | (d.val[3]<<6)
        vst1q_u8(packed + (i >> 2),
            vorrq_u8(
                vorrq_u8(d.val[0], vshlq_n_u8(d.val[1], 2)),
                vorrq_u8(vshlq_n_u8(d.val[2], 4), vshlq_n_u8(d.val[3], 6))));
    }
    for (; i < cells; i += 4) {
        packed[i >> 2] = (bytes[i] & 3u)
                       | ((bytes[i+1] & 3u) << 2)
                       | ((bytes[i+2] & 3u) << 4)
                       | ((bytes[i+3] & 3u) << 6);
    }
    (void)k1; (void)k2; (void)k4; (void)k6;
}

// Convert N²/4 packed grid → N² byte grid.
static void unpack_grid(const uint8_t* packed, uint8_t* bytes, size_t cells)
{
    size_t i = 0;
    for (; i + 64 <= cells; i += 64) {
        uint8x16_t p = vld1q_u8(packed + (i >> 2));
        const uint8x16_t mask = vdupq_n_u8(0x03u);
        uint8x16x4_t d;
        d.val[0] = vandq_u8(p, mask);
        d.val[1] = vandq_u8(vshrq_n_u8(p, 2), mask);
        d.val[2] = vandq_u8(vshrq_n_u8(p, 4), mask);
        d.val[3] = vshrq_n_u8(p, 6);
        vst4q_u8(bytes + i, d);
    }
    for (; i < cells; i += 4) {
        uint8_t b = packed[i >> 2];
        bytes[i]   =  b       & 3u;
        bytes[i+1] = (b >> 2) & 3u;
        bytes[i+2] = (b >> 4) & 3u;
        bytes[i+3] = (b >> 6) & 3u;
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

    const int    N      = static_cast<int>(width);
    const size_t CELLS  = static_cast<size_t>(N) * N;
    const size_t PACKED = CELLS >> 2;   // N²/4 bytes per packed grid

    auto alloc = [](size_t n) -> uint8_t* {
        void* p = nullptr;
        if (posix_memalign(&p, 64, n) != 0 || !p) {
            std::fprintf(stderr, "Fatal: alloc failed\n"); std::exit(1);
        }
        return static_cast<uint8_t*>(p);
    };

    // Read byte grid from file
    std::vector<uint8_t> input_bytes(CELLS);
    if (std::fread(input_bytes.data(), 1, CELLS, fin) != CELLS) {
        std::fprintf(stderr, "Error: input truncated\n");
        std::fclose(fin); return 4;
    }
    std::fclose(fin);

    // Pack to 2-bit format for simulation
    uint8_t* grid_a = alloc(PACKED);
    uint8_t* grid_b = alloc(PACKED);
    pack_grid(input_bytes.data(), grid_a, CELLS);
    input_bytes.clear();   // free the 1× byte buffer — no longer needed

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

    // Unpack result to byte format for file output
    uint8_t* packed_result = wc.grids[generations & 1];
    std::vector<uint8_t> output_bytes(CELLS);
    unpack_grid(packed_result, output_bytes.data(), CELLS);

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) { std::fprintf(stderr, "Error: cannot open '%s'\n", argv[2]); return 5; }

    const bool ok =
        (std::fwrite(&width,               8, 1,     fout) == 1) &&
        (std::fwrite(&height,              8, 1,     fout) == 1) &&
        (std::fwrite(output_bytes.data(),  1, CELLS, fout) == CELLS);
    std::fclose(fout);
    if (!ok) { std::fprintf(stderr, "Error: write error\n"); return 6; }

    std::free(grid_a);
    std::free(grid_b);
    return 0;
}