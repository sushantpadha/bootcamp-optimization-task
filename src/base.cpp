// spawn_sim_v2.cpp — Monster Spawning Grid (dual bit-plane implementation).
//
// WHAT CHANGED FROM THE REFERENCE
//   • State stored as two bit-planes (p0 = low bit, p1 = high bit) rather than
//     one byte per cell.  For N=32768 this cuts the working set from 1 GiB to
//     128 MiB per buffer — the single largest leverage available before SIMD.
//
//   • Toroidal vertical wrap handled by ghost/halo rows (memcpy, O(N) overhead)
//     so the inner loop has no modulo arithmetic for row indices.
//     Toroidal horizontal wrap handled by a per-row extended scratch buffer of
//     N+4 bytes with column ghosts pre-filled; same idea, different axis.
//
//   • Neighbour count uses a separable sliding-window sum:
//       1. Horizontal pass  — 5-wide sliding sum of adult bits → H[row][col].
//          Rows are processed on demand; only 5 rows are live at once.
//       2. Vertical pass    — accumulate 5 consecutive H-rows into a_buf,
//          subtract the centre cell, giving the 24-neighbour count.
//     The vertical accumulation iterates over the five hring rows sequentially
//     (one pass per row), which keeps the access pattern cache-friendly.
//
// STATE ENCODING
//   EMPTY=00  EGG=01  JUVENILE=10  ADULT=11
//   p0 holds the low bit; p1 holds the high bit.
//   ADULT ≡ p0_bit AND p1_bit.
//
// BIT/WORD LAYOUT
//   Cell at (live_row r, col c) → padded row r+HALO,
//                                  word w = c/64, bit b = c%64.
//   Bit 0 of each word is the leftmost cell in its group of 64.
//
// PLANE LAYOUT (each plane: total_rows × rw  uint64_t words)
//   Padded rows 0..HALO-1           : top ghost (copies of last HALO live rows)
//   Padded rows HALO..HALO+N-1      : live data
//   Padded rows HALO+N..HALO+N+HALO-1: bottom ghost (copies of first HALO live rows)
//
// CONSTRAINTS
//   N must be a multiple of 64 (= BITS_PER_WORD).  The stated input domain is
//   power-of-two square grids; all such grids with N ≥ 64 satisfy this.
//
// INPUT / OUTPUT FORMAT: unchanged from reference.
//
// COMPILE (Graviton4 / Neoverse-V2, no SIMD yet):
//   g++ -std=c++23 -O3 -march=armv9-a -mtune=neoverse-v2 \
//       -fno-omit-frame-pointer -Wall -Wextra \
//       -o spawn_sim_v2 spawn_sim_v2.cpp
//
// NEXT STEPS (not done here)
//   • SVE2 inner loops for horizontal unpack and vertical accumulation
//   • Bit-parallel Wallace-tree neighbour count (avoids byte unpacking entirely)
//   • Row-strip tiling to keep the hring in L2
//   • Parallel row decomposition across 8 threads

#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static_assert(sizeof(uint64_t) == 8, "uint64_t must be 8 bytes");

// ─────────────────────────────────────────────────────────────────────────────
//  Compile-time constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr size_t  RANGE         = 2;
static constexpr size_t  HALO          = RANGE;              // ghost rows per side = 2
static constexpr size_t  WINDOW        = RANGE * 2 + 1;      // stencil width = 5
static constexpr size_t  BITS_PER_WORD = 64;
static constexpr size_t  CACHE_LINE    = 64;                 // bytes; alignment target

// State values — must match the I/O format.
static constexpr uint8_t S_EMPTY    = 0;
static constexpr uint8_t S_EGG      = 1;
static constexpr uint8_t S_JUVENILE = 2;
static constexpr uint8_t S_ADULT    = 3;

// Transition thresholds.
static constexpr uint8_t BIRTH_LO = 3, BIRTH_HI = 5;
static constexpr uint8_t SURV_LO  = 4, SURV_HI  = 9;

// ─────────────────────────────────────────────────────────────────────────────
//  Aligned allocation
// ─────────────────────────────────────────────────────────────────────────────

// Wraps std::aligned_alloc, rounding `bytes` up to the next multiple of `align`.
// Terminates the process on OOM (acceptable for a benchmark binary).
[[nodiscard]] static void* xaligned_alloc(size_t align, size_t bytes)
{
    const size_t rounded = (bytes + align - 1) & ~(align - 1);
    void* p = std::aligned_alloc(align, rounded);
    if (__builtin_expect(!p, 0)) {
        std::fprintf(stderr, "fatal: out of memory requesting %zu bytes\n", rounded);
        std::exit(EXIT_FAILURE);
    }
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
//  BitGrid
// ─────────────────────────────────────────────────────────────────────────────

struct BitGrid {
    size_t    N;            // logical grid side in cells
    size_t    rw;           // uint64_t words per padded row  (= N / BITS_PER_WORD)
    size_t    total_rows;   // padded row count               (= N + 2*HALO)
    uint64_t* p0;           // bit-plane 0 — low bit of state
    uint64_t* p1;           // bit-plane 1 — high bit of state
};

[[nodiscard]] static BitGrid alloc_bitgrid(size_t N)
{
    const size_t rw         = N / BITS_PER_WORD;
    const size_t total_rows = N + 2 * HALO;
    const size_t bytes      = total_rows * rw * sizeof(uint64_t);

    BitGrid g;
    g.N          = N;
    g.rw         = rw;
    g.total_rows = total_rows;
    g.p0 = static_cast<uint64_t*>(xaligned_alloc(CACHE_LINE, bytes));
    g.p1 = static_cast<uint64_t*>(xaligned_alloc(CACHE_LINE, bytes));
    std::memset(g.p0, 0, bytes);
    std::memset(g.p1, 0, bytes);
    return g;
}

static void free_bitgrid(BitGrid& g)
{
    std::free(g.p0);  g.p0 = nullptr;
    std::free(g.p1);  g.p1 = nullptr;
}

// Pointer to the first word of padded row `pr` in `plane`.
// Templated on T so the same call works for both uint64_t* and const uint64_t*
// without requiring two overloads.
template <typename T>
[[nodiscard]] static inline T* row_ptr(T* plane, const BitGrid& g, size_t pr) noexcept
{
    return plane + pr * g.rw;
}

// ─────────────────────────────────────────────────────────────────────────────
//  I/O conversion between byte-per-cell format and bit planes
// ─────────────────────────────────────────────────────────────────────────────

// Populate the live-data rows of `g` from a flat row-major byte array.
static void bytes_to_bitgrid(const uint8_t* __restrict__ cells, BitGrid& g)
{
    for (size_t r = 0; r < g.N; ++r) {
        uint64_t* rp0 = row_ptr(g.p0, g, r + HALO);
        uint64_t* rp1 = row_ptr(g.p1, g, r + HALO);
        const uint8_t* row_in = cells + r * g.N;

        for (size_t w = 0; w < g.rw; ++w) {
            uint64_t w0 = 0, w1 = 0;
            for (size_t b = 0; b < BITS_PER_WORD; ++b) {
                const uint64_t s = row_in[w * BITS_PER_WORD + b];
                w0 |= (uint64_t)(s & 1u)         << b;
                w1 |= (uint64_t)((s >> 1u) & 1u) << b;
            }
            rp0[w] = w0;
            rp1[w] = w1;
        }
    }
}

// Write the live-data rows of `g` back to a flat row-major byte array.
static void bitgrid_to_bytes(const BitGrid& g, uint8_t* __restrict__ cells)
{
    for (size_t r = 0; r < g.N; ++r) {
        const uint64_t* rp0 = row_ptr(g.p0, g, r + HALO);
        const uint64_t* rp1 = row_ptr(g.p1, g, r + HALO);
        uint8_t* row_out = cells + r * g.N;

        for (size_t w = 0; w < g.rw; ++w) {
            const uint64_t a0 = rp0[w];
            const uint64_t a1 = rp1[w];
            for (size_t b = 0; b < BITS_PER_WORD; ++b) {
                row_out[w * BITS_PER_WORD + b] =
                    (uint8_t)(((a0 >> b) & 1u) | (((a1 >> b) & 1u) << 1));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Halo update — toroidal vertical wrap via ghost rows
// ─────────────────────────────────────────────────────────────────────────────
//
// Ghost row layout (padded indices):
//   0 → copy of live row N-2   (= padded HALO+N-2)
//   1 → copy of live row N-1   (= padded HALO+N-1)
//   HALO+N   → copy of live row 0   (= padded HALO)
//   HALO+N+1 → copy of live row 1   (= padded HALO+1)
//
// Must be called on the source grid before each step().
static void update_halo(BitGrid& g)
{
    const size_t row_bytes = g.rw * sizeof(uint64_t);

    uint64_t* planes[2] = { g.p0, g.p1 };
    for (int pi = 0; pi < 2; ++pi) {
        uint64_t* pl = planes[pi];

        // Top ghost rows ← last two live rows.
        std::memcpy(row_ptr(pl, g, 0), row_ptr(pl, g, g.N + HALO - 2), row_bytes);
        std::memcpy(row_ptr(pl, g, 1), row_ptr(pl, g, g.N + HALO - 1), row_bytes);

        // Bottom ghost rows ← first two live rows.
        std::memcpy(row_ptr(pl, g, g.N + HALO    ), row_ptr(pl, g, HALO    ), row_bytes);
        std::memcpy(row_ptr(pl, g, g.N + HALO + 1), row_ptr(pl, g, HALO + 1), row_bytes);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Scratch memory — allocated once per run, reused every generation
// ─────────────────────────────────────────────────────────────────────────────

struct StepScratch {
    // Extended adult row for the horizontal pass: N+4 bytes.
    // ext[0..1]       : left column ghosts  (= adult(N-2), adult(N-1))
    // ext[2..N+1]     : live adult values   (= adult(0..N-1))
    // ext[N+2..N+3]   : right column ghosts (= adult(0),   adult(1))
    uint8_t* ext;

    // Ring buffer of horizontal sum rows: WINDOW × N bytes.
    // Ring slot for padded row pr lives at hring_data[(pr % WINDOW) * N].
    // hring[pr][c] = number of ADULT cells in row pr, columns [c-RANGE..c+RANGE].
    uint8_t* hring_data;

    // Per-row neighbour count after the vertical pass: N bytes.
    // a_buf[c] = total 24-neighbour ADULT count for the current output column c.
    uint8_t* a_buf;
};

[[nodiscard]] static StepScratch alloc_scratch(size_t N)
{
    StepScratch s;
    s.ext        = static_cast<uint8_t*>(xaligned_alloc(CACHE_LINE, N + 4));
    s.hring_data = static_cast<uint8_t*>(xaligned_alloc(CACHE_LINE, WINDOW * N));
    s.a_buf      = static_cast<uint8_t*>(xaligned_alloc(CACHE_LINE, N));
    return s;
}

static void free_scratch(StepScratch& s)
{
    std::free(s.ext);
    std::free(s.hring_data);
    std::free(s.a_buf);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Horizontal pass: one padded row → 5-wide sliding-window adult sum
// ─────────────────────────────────────────────────────────────────────────────
//
// Writes hsum[0..N-1] where hsum[c] = Σ adult(c+dx) for dx in [-RANGE, +RANGE].
// Toroidal column wrap is handled by the column ghost cells in ext[].
//
// Note: hsum[c] counts the centre cell itself (adult(c)), not just the neighbours.
// The centre contribution is subtracted later, in the vertical pass.
static void compute_hsum_row(
    const uint64_t* __restrict__ rw_p0,
    const uint64_t* __restrict__ rw_p1,
    size_t N, size_t rw,
    uint8_t* __restrict__ ext,   // scratch: N+4 bytes
    uint8_t* __restrict__ hsum)  // output:  N bytes
{
    // Step 1: unpack adult bits into ext[HALO .. HALO+N-1].
    // ADULT ≡ p0_bit AND p1_bit, so merge the planes before unpacking.
    for (size_t w = 0; w < rw; ++w) {
        const uint64_t av = rw_p0[w] & rw_p1[w];
        for (size_t b = 0; b < BITS_PER_WORD; ++b) {
            ext[HALO + w * BITS_PER_WORD + b] = (uint8_t)((av >> b) & 1u);
        }
    }

    // Step 2: fill column ghosts for toroidal horizontal wrap.
    //   ext[0]   = adult(N-2)  =  ext[N]   (= ext[HALO + N-2])
    //   ext[1]   = adult(N-1)  =  ext[N+1] (= ext[HALO + N-1])
    //   ext[N+2] = adult(0)    =  ext[2]   (= ext[HALO + 0])
    //   ext[N+3] = adult(1)    =  ext[3]   (= ext[HALO + 1])
    ext[0]     = ext[N];
    ext[1]     = ext[N + 1];
    ext[N + 2] = ext[2];
    ext[N + 3] = ext[3];

    // Step 3: 5-wide sliding-window sum.
    //   hsum[c] = ext[c] + ext[c+1] + ext[c+2] + ext[c+3] + ext[c+4]
    //
    // Invariant at the top of each iteration: wsum = ext[c..c+3].
    // We add ext[c+4] = ext[c+WINDOW-1], store the result, then remove ext[c].
    uint8_t wsum = ext[0] + ext[1] + ext[2] + ext[3];
    for (size_t c = 0; c < N; ++c) {
        wsum    = (uint8_t)(wsum + ext[c + WINDOW - 1]);
        hsum[c] = wsum;
        wsum    = (uint8_t)(wsum - ext[c]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  One generation step: src → dst
// ─────────────────────────────────────────────────────────────────────────────
//  Precondition: update_halo(src) has been called.
static void step(const BitGrid& src, BitGrid& dst, StepScratch& sc)
{
    const size_t N  = src.N;
    const size_t rw = src.rw;

    // hring_row(pr): pointer to the ring-buffer slot for padded row pr.
    // Slot index = pr % WINDOW cycles over [0..4] as pr advances, which is
    // exactly the right period to recycle slots that are no longer needed.
    auto hring_row = [&](size_t pr) -> uint8_t* {
        return sc.hring_data + (pr % WINDOW) * N;
    };

    // Pre-fill the ring for padded rows 0..WINDOW-2 (= rows 0..3).
    // When the main loop begins at pr=HALO=2, it will fill row pr+HALO=4,
    // giving us all five rows 0..4 before the first vertical sum.
    for (size_t pr = 0; pr < WINDOW - 1; ++pr) {
        compute_hsum_row(
            row_ptr(src.p0, src, pr), row_ptr(src.p1, src, pr),
            N, rw, sc.ext, hring_row(pr));
    }

    // Process each live row (padded rows HALO .. HALO+N-1).
    for (size_t pr = HALO; pr < HALO + N; ++pr) {

        // ── Horizontal pass: bring lookahead row into the ring ─────────────
        // The lookahead is pr+HALO = pr+2; its slot (pr+2)%5 is now free
        // because the last time it was used was for the row pr-3 pass.
        const size_t look = pr + HALO;
        compute_hsum_row(
            row_ptr(src.p0, src, look), row_ptr(src.p1, src, look),
            N, rw, sc.ext, hring_row(look));

        // ── Vertical pass: accumulate five hring rows into a_buf ──────────
        // Iterating row-by-row (outer=dr, inner=c) gives sequential reads
        // of each N-byte hring row — all five fit easily in L1/L2 together.
        std::memset(sc.a_buf, 0, N);
        for (size_t dr = 0; dr < WINDOW; ++dr) {
            const uint8_t* hr = hring_row(pr - HALO + dr);
            for (size_t c = 0; c < N; ++c) {
                sc.a_buf[c] = (uint8_t)(sc.a_buf[c] + hr[c]);
            }
        }

        // Subtract each cell's own adult contribution: the horizontal sums
        // included the centre cell, but the stencil excludes it.
        const uint64_t* src_rp0 = row_ptr(src.p0, src, pr);
        const uint64_t* src_rp1 = row_ptr(src.p1, src, pr);
        for (size_t w = 0; w < rw; ++w) {
            const uint64_t av = src_rp0[w] & src_rp1[w];   // 1-bit = ADULT
            for (size_t b = 0; b < BITS_PER_WORD; ++b) {
                sc.a_buf[w * BITS_PER_WORD + b] =
                    (uint8_t)(sc.a_buf[w * BITS_PER_WORD + b] - ((av >> b) & 1u));
            }
        }

        // ── Transition + bit-plane packing ────────────────────────────────
        // Build one word (64 cells) of output at a time to avoid per-bit
        // read-modify-write on the destination planes.
        uint64_t* dst_rp0 = row_ptr(dst.p0, dst, pr);
        uint64_t* dst_rp1 = row_ptr(dst.p1, dst, pr);

        for (size_t w = 0; w < rw; ++w) {
            uint64_t out0 = 0, out1 = 0;
            const size_t base = w * BITS_PER_WORD;

            for (size_t b = 0; b < BITS_PER_WORD; ++b) {
                const uint8_t A     = sc.a_buf[base + b];
                const uint8_t p0b   = (uint8_t)((src_rp0[w] >> b) & 1u);
                const uint8_t p1b   = (uint8_t)((src_rp1[w] >> b) & 1u);
                const uint8_t state = (uint8_t)((p1b << 1) | p0b);

                uint8_t ns;
                switch (state) {
                    case S_EMPTY:    ns = (A >= BIRTH_LO && A <= BIRTH_HI) ? S_EGG      : S_EMPTY; break;
                    case S_EGG:      ns = S_JUVENILE;                                              break;
                    case S_JUVENILE: ns = S_ADULT;                                                 break;
                    default:         ns = (A >= SURV_LO  && A <= SURV_HI)  ? S_ADULT    : S_EMPTY; break;
                }

                out0 |= (uint64_t)(ns & 1u)         << b;
                out1 |= (uint64_t)((ns >> 1u) & 1u) << b;
            }

            dst_rp0[w] = out0;
            dst_rp1[w] = out1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "Usage: %s <input.bin> <output.bin> [generations]\n", argv[0]);
        return 1;
    }

    int generations = 10'000;
    if (argc == 4) {
        char* end;
        long g = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || g <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        generations = (int)g;
    }

    // ── Read input ─────────────────────────────────────────────────────────
    FILE* fin = std::fopen(argv[1], "rb");
    if (__builtin_expect(!fin, 0)) {
        std::fprintf(stderr, "Error: cannot open '%s'\n", argv[1]);
        return 2;
    }

    uint64_t width, height;
    if (std::fread(&width,  sizeof width,  1, fin) != 1 ||
        std::fread(&height, sizeof height, 1, fin) != 1) {
        std::fprintf(stderr, "Error: input file too short (header)\n");
        std::fclose(fin);
        return 3;
    }

    if (width == 0 || width != height) {
        std::fprintf(stderr,
            "Error: grid must be square and non-empty"
            " (got %" PRIu64 " × %" PRIu64 ")\n", width, height);
        std::fclose(fin);
        return 3;
    }

    const size_t N = (size_t)width;

    // Bit-plane layout requires N to be a multiple of BITS_PER_WORD.
    // All power-of-2 grids with N ≥ 64 satisfy this; smaller inputs require
    // a different representation (not needed per the stated problem domain).
    if (N < BITS_PER_WORD || (N % BITS_PER_WORD) != 0) {
        std::fprintf(stderr,
            "Error: N=%zu must be a multiple of %zu for this implementation\n",
            N, BITS_PER_WORD);
        std::fclose(fin);
        return 3;
    }

    const size_t Ar  = N * N;
    auto* raw        = static_cast<uint8_t*>(xaligned_alloc(CACHE_LINE, Ar));

    if (std::fread(raw, 1, Ar, fin) != Ar) {
        std::fprintf(stderr, "Error: input file too short (cell data)\n");
        std::fclose(fin);
        std::free(raw);
        return 4;
    }
    std::fclose(fin);

    // ── Initialise ─────────────────────────────────────────────────────────
    BitGrid     grid_a  = alloc_bitgrid(N);
    BitGrid     grid_b  = alloc_bitgrid(N);
    StepScratch scratch = alloc_scratch(N);

    bytes_to_bitgrid(raw, grid_a);
    std::free(raw);
    raw = nullptr;

    BitGrid* cur = &grid_a;
    BitGrid* nxt = &grid_b;

    // ── Simulate ───────────────────────────────────────────────────────────
    auto t0 = std::chrono::steady_clock::now();

    for (int gen = 0; gen < generations; ++gen) {
        update_halo(*cur);
        step(*cur, *nxt, scratch);
        std::swap(cur, nxt);
    }

    auto t1 = std::chrono::steady_clock::now();

    const double ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double mcps = (double)N * N * generations / (ms * 1e3);
    std::printf("%.3f ms  (%.3f ms/gen,  %.1f Mcell/s)\n", ms, ms / generations, mcps);

    // ── Write output ───────────────────────────────────────────────────────
    raw = static_cast<uint8_t*>(xaligned_alloc(CACHE_LINE, Ar));
    bitgrid_to_bytes(*cur, raw);

    FILE* fout = std::fopen(argv[2], "wb");
    if (__builtin_expect(!fout, 0)) {
        std::fprintf(stderr, "Error: cannot open '%s' for writing\n", argv[2]);
        std::free(raw);
        return 5;
    }

    const uint64_t dim = (uint64_t)N;
    if (std::fwrite(&dim, sizeof dim, 1, fout) != 1 ||
        std::fwrite(&dim, sizeof dim, 1, fout) != 1 ||
        std::fwrite(raw,  1, Ar,      fout)   != Ar) {
        std::fprintf(stderr, "Error: write error on '%s'\n", argv[2]);
        std::fclose(fout);
        std::free(raw);
        return 6;
    }
    std::fclose(fout);
    std::free(raw);

    // ── Cleanup ────────────────────────────────────────────────────────────
    free_scratch(scratch);
    free_bitgrid(grid_a);
    free_bitgrid(grid_b);

    return 0;
}