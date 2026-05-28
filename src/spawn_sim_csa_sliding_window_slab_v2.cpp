// spawn_sim_csa_tree_sliding_window_threadpool_sve2_tiled_column_major.cpp —
// Column-major tiled hot-loop with in-register vertical accumulator,
// on-the-fly horizontal partial recomputation (ring buffer eliminated),
// non-temporal output stores, and persistent row-chunk worker threads
// for the Monster Spawning Grid reference simulator.
//
// Key changes vs previous version:
//   1. Column-major tiled hot loop: word_index is the outer loop,
//      row_index is the inner loop within each tile.  The 5-wide vertical
//      accumulator lives in five SVE registers for the duration of each
//      (tile, word_index) pass — no rolling_total arrays in memory.
//   2. Ring buffer eliminated: horizontal partials for the outgoing row
//      are recomputed on-the-fly from the adults bitplane.  The adults
//      data for row r-2 is still hot in L1 when needed (loaded 2 outer
//      iterations ago).  HotRowScratch and all associated structures are
//      gone entirely.
//   3. Non-temporal stores (svstnt1_u64) for next_adults and next_eggs:
//      these planes are write-only this generation, so bypassing the
//      cache on the write path leaves L1/L2 free for the rolling
//      accumulator and input bitplanes.
//
// INPUT FORMAT:
//   Binary file, little-endian:
//     bytes  0– 7: uint64_t width
//     bytes  8–15: uint64_t height (must equal width; only square grids)
//     bytes 16–  : width × height bytes of cell data, row-major.
//                  Each byte: 0=EMPTY, 1=EGG, 2=JUVENILE, 3=ADULT.
//
// OUTPUT FORMAT:
//   Same binary format, encoding the grid after the requested generations.
//
// USAGE:
//   spawn_sim <input.bin> <output.bin> [generations]
//   generations defaults to 10000 if not specified.
//
// TIMING:
//   Prints wall-clock simulation time (excluding I/O and pack/unpack)
//   to stdout as a single line: "<N.NNN> ms"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <type_traits>
#include <vector>

#include <arm_sve.h>

static constexpr uint8_t EMPTY    = 0;
static constexpr uint8_t EGG      = 1;
static constexpr uint8_t JUVENILE = 2;
static constexpr uint8_t ADULT    = 3;

// Number of rows processed per tile in the column-major hot loop.
// Chosen so that (TILE_ROWS + 4) rows of three 64-bit-per-cell bitplanes
// fit comfortably in a 32 KB L1D cache despite the strided access pattern.
// Formula: (L1_bytes / 2) / (3_planes * words_per_row * 8) >= TILE_ROWS + 4
// For wpr=16 (1024-wide grid): 16384 / 384 ≈ 42; use 32 for alignment margin.
// Larger grids (wpr=64) should use a smaller tile; 32 remains safe for all
// common server L1 sizes (≥ 32 KB).
static constexpr int TILE_ROWS = 32;

static constexpr int TARGET_THREAD_COUNT = 8;

// ---------------------------------------------------------------------------
// Aligned allocator (used only for the main BitPlane vectors)
// ---------------------------------------------------------------------------

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type   = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count)
    {
        if (count > static_cast<std::size_t>(-1) / sizeof(T))
            throw std::bad_array_new_length();
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, count * sizeof(T)) != 0)
            throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept { std::free(ptr); }

    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

template <typename T, typename U, std::size_t A>
bool operator==(const AlignedAllocator<T,A>&, const AlignedAllocator<U,A>&) { return true; }
template <typename T, typename U, std::size_t A>
bool operator!=(const AlignedAllocator<T,A>&, const AlignedAllocator<U,A>&) { return false; }

using BitPlane = std::vector<uint64_t, AlignedAllocator<uint64_t, 64>>;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

struct RowRange { int begin = 0; int end = 0; };

// Returned by the scalar horizontal-partial helper.
struct HorizontalPartialWord {
    uint64_t b0 = 0ULL;
    uint64_t b1 = 0ULL;
    uint64_t b2 = 0ULL;
};

static RowRange compute_row_range(int grid_size, int thread_count, int worker_id)
{
    const int base  = grid_size / thread_count;
    const int rem   = grid_size % thread_count;
    const int begin = worker_id * base + std::min(worker_id, rem);
    const int count = base + (worker_id < rem ? 1 : 0);
    return RowRange{begin, begin + count};
}

static void pin_current_thread_to_cpu(int worker_id)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(worker_id, &mask);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

inline int wrap_coordinate(int coord, int grid_size)
{
    return (coord + grid_size) & (grid_size - 1);
}

inline size_t get_flat_word_index(int row, int col, int wpr)
{
    return static_cast<size_t>(row) * wpr + (col >> 6);
}

inline uint64_t get_bit_mask(int col) { return 1ULL << (col & 63); }

inline bool is_cell_set(const BitPlane& bp, int wpr, int row, int col)
{
    return (bp[get_flat_word_index(row, col, wpr)] & get_bit_mask(col)) != 0;
}

inline void set_cell_bit(BitPlane& bp, int wpr, int row, int col)
{
    bp[get_flat_word_index(row, col, wpr)] |= get_bit_mask(col);
}

// ---------------------------------------------------------------------------
// Bit-plane packing / unpacking
// ---------------------------------------------------------------------------

static void pack_input_into_bit_planes(
    const std::vector<uint8_t>& input_grid, int grid_size, int wpr,
    BitPlane& adults, BitPlane& juveniles, BitPlane& eggs)
{
    for (int r = 0; r < grid_size; ++r)
        for (int c = 0; c < grid_size; ++c) {
            uint8_t v = input_grid[static_cast<size_t>(r) * grid_size + c];
            if      (v == ADULT)    set_cell_bit(adults,   wpr, r, c);
            else if (v == JUVENILE) set_cell_bit(juveniles, wpr, r, c);
            else if (v == EGG)      set_cell_bit(eggs,      wpr, r, c);
        }
}

static void unpack_bit_planes_to_output_grid(
    int grid_size, int wpr,
    const BitPlane& adults, const BitPlane& juveniles, const BitPlane& eggs,
    std::vector<uint8_t>& out)
{
    out.assign(static_cast<size_t>(grid_size) * grid_size, EMPTY);
    for (int r = 0; r < grid_size; ++r)
        for (int c = 0; c < grid_size; ++c) {
            uint8_t v = EMPTY;
            if      (is_cell_set(adults,   wpr, r, c)) v = ADULT;
            else if (is_cell_set(juveniles, wpr, r, c)) v = JUVENILE;
            else if (is_cell_set(eggs,      wpr, r, c)) v = EGG;
            out[static_cast<size_t>(r) * grid_size + c] = v;
        }
}

// ---------------------------------------------------------------------------
// Naive neighbor counter (used only for the toroidal wrapping border)
// ---------------------------------------------------------------------------

static int count_adult_neighbors_naive(
    const BitPlane& adults, int grid_size, int wpr, int cr, int cc)
{
    int n = 0;
    for (int dr = -2; dr <= 2; ++dr) {
        int nr = wrap_coordinate(cr + dr, grid_size);
        for (int dc = -2; dc <= 2; ++dc) {
            if (dr == 0 && dc == 0) continue;
            int nc = wrap_coordinate(cc + dc, grid_size);
            if (is_cell_set(adults, wpr, nr, nc)) ++n;
        }
    }
    return n;
}

// ---------------------------------------------------------------------------
// Carry-save adder primitives
// Forward-declared here so the horizontal-partial helpers below can use them
// before their bodies appear later in the file.
// ---------------------------------------------------------------------------

inline void carry_save_add(
    uint64_t a, uint64_t b, uint64_t c,
    uint64_t& sum, uint64_t& carry);

inline void carry_save_add_u64_sve(
    svbool_t pg,
    svuint64_t a, svuint64_t b, svuint64_t c,
    svuint64_t& sum, svuint64_t& carry);

// ---------------------------------------------------------------------------
// Horizontal partial computation
// Computes a 3-bit carry-save representation of the horizontal 5-neighbor
// adult count for a single word_index in one row, directly from the raw
// adults pointer.  No ring buffer; callers recompute as needed.
// ---------------------------------------------------------------------------

__attribute__((always_inline))
static inline HorizontalPartialWord
compute_horizontal_partial_scalar_from_adult_row(
    const uint64_t* __restrict adult_row, int word_index)
{
    const uint64_t prev   = adult_row[word_index - 1];
    const uint64_t curr   = adult_row[word_index];
    const uint64_t next   = adult_row[word_index + 1];
    const uint64_t left_2 = (curr << 2) | (prev >> 62);
    const uint64_t left_1 = (curr << 1) | (prev >> 63);
    const uint64_t right_1 = (curr >> 1) | (next << 63);
    const uint64_t right_2 = (curr >> 2) | (next << 62);

    uint64_t sum_1 = 0, carry_1 = 0, sum_2 = 0, carry_2 = 0;
    carry_save_add(left_2, left_1, curr,    sum_1, carry_1);
    carry_save_add(right_1, right_2, sum_1, sum_2, carry_2);
    return HorizontalPartialWord{sum_2, carry_1 ^ carry_2, carry_1 & carry_2};
}

__attribute__((always_inline))
static inline void
compute_horizontal_partial_pair2_u64_sve_from_adult_row(
    svbool_t pg,
    const uint64_t* __restrict adult_row,
    int word_index,
    svuint64_t& partial_b0,
    svuint64_t& partial_b1,
    svuint64_t& partial_b2)
{
    const svuint64_t prev = svld1_u64(pg, adult_row + word_index - 1);
    const svuint64_t curr = svld1_u64(pg, adult_row + word_index);
    const svuint64_t next = svld1_u64(pg, adult_row + word_index + 1);

    const svuint64_t left_2  = svorr_x(pg, svlsl_n_u64_x(pg, curr, 2), svlsr_n_u64_x(pg, prev, 62));
    const svuint64_t left_1  = svorr_x(pg, svlsl_n_u64_x(pg, curr, 1), svlsr_n_u64_x(pg, prev, 63));
    const svuint64_t right_1 = svorr_x(pg, svlsr_n_u64_x(pg, curr, 1), svlsl_n_u64_x(pg, next, 63));
    const svuint64_t right_2 = svorr_x(pg, svlsr_n_u64_x(pg, curr, 2), svlsl_n_u64_x(pg, next, 62));

    svuint64_t sum_1, carry_1, sum_2, carry_2;
    carry_save_add_u64_sve(pg, left_2,  left_1,  curr,  sum_1, carry_1);
    carry_save_add_u64_sve(pg, right_1, right_2, sum_1, sum_2, carry_2);

    partial_b0 = sum_2;
    partial_b1 = sveor_x(pg, carry_1, carry_2);
    partial_b2 = svand_x(pg, carry_1, carry_2);
}

// ---------------------------------------------------------------------------
// CSA implementations
// ---------------------------------------------------------------------------

inline void carry_save_add(
    uint64_t a, uint64_t b, uint64_t c,
    uint64_t& sum, uint64_t& carry)
{
    sum   = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

inline void carry_save_add_u64_sve(
    svbool_t pg,
    svuint64_t a, svuint64_t b, svuint64_t c,
    svuint64_t& sum, svuint64_t& carry)
{
    sum   = sveor_x(pg, sveor_x(pg, a, b), c);
    carry = svbsl_u64(svorr_x(pg, a, b), svand_x(pg, a, b), c);
}

// ---------------------------------------------------------------------------
// Vertical accumulator add / subtract (SVE)
// ---------------------------------------------------------------------------

inline void add_horizontal_partial_to_total_u64_sve(
    svbool_t pg,
    svuint64_t pb0, svuint64_t pb1, svuint64_t pb2,
    svuint64_t& tb0, svuint64_t& tb1, svuint64_t& tb2,
    svuint64_t& tb3, svuint64_t& tb4)
{
    svuint64_t carry = svand_x(pg, tb0, pb0);
    tb0 = sveor_x(pg, tb0, pb0);
    carry_save_add_u64_sve(pg, tb1, pb1, carry, tb1, carry);
    carry_save_add_u64_sve(pg, tb2, pb2, carry, tb2, carry);
    const svuint64_t carry4 = svand_x(pg, tb3, carry);
    tb3 = sveor_x(pg, tb3, carry);
    tb4 = sveor_x(pg, tb4, carry4);
}

inline void subtract_horizontal_partial_from_total_u64_sve(
    svbool_t pg,
    svuint64_t pb0, svuint64_t pb1, svuint64_t pb2,
    svuint64_t& tb0, svuint64_t& tb1, svuint64_t& tb2,
    svuint64_t& tb3, svuint64_t& tb4)
{
    const svuint64_t ones = svdup_u64(~0ULL);
    svuint64_t borrow     = svand_x(pg, sveor_x(pg, tb0, ones), pb0);
    tb0 = sveor_x(pg, tb0, pb0);

    const svuint64_t ntb1 = sveor_x(pg, tb1, ones);
    svuint64_t nb = svbsl_u64(svorr_x(pg, ntb1, pb1), svand_x(pg, ntb1, pb1), borrow);
    tb1 = sveor_x(pg, sveor_x(pg, tb1, pb1), borrow);
    borrow = nb;

    const svuint64_t ntb2 = sveor_x(pg, tb2, ones);
    nb = svbsl_u64(svorr_x(pg, ntb2, pb2), svand_x(pg, ntb2, pb2), borrow);
    tb2 = sveor_x(pg, sveor_x(pg, tb2, pb2), borrow);
    borrow = nb;

    nb  = svand_x(pg, sveor_x(pg, tb3, ones), borrow);
    tb3 = sveor_x(pg, tb3, borrow);
    tb4 = sveor_x(pg, tb4, nb);
}

// ---------------------------------------------------------------------------
// Vertical accumulator add / subtract (scalar)
// ---------------------------------------------------------------------------

inline void add_horizontal_partial_to_total(
    uint64_t pb0, uint64_t pb1, uint64_t pb2,
    uint64_t& tb0, uint64_t& tb1, uint64_t& tb2,
    uint64_t& tb3, uint64_t& tb4)
{
    uint64_t carry = tb0 & pb0;
    tb0 ^= pb0;
    carry_save_add(tb1, pb1, carry, tb1, carry);
    carry_save_add(tb2, pb2, carry, tb2, carry);
    const uint64_t c4 = tb3 & carry;
    tb3 ^= carry;
    tb4 ^= c4;
}

inline void subtract_horizontal_partial_from_total(
    uint64_t pb0, uint64_t pb1, uint64_t pb2,
    uint64_t& tb0, uint64_t& tb1, uint64_t& tb2,
    uint64_t& tb3, uint64_t& tb4)
{
    uint64_t borrow = (~tb0) & pb0;
    tb0 ^= pb0;
    uint64_t nb = ((~tb1) & (pb1 | borrow)) | (pb1 & borrow);
    tb1 ^= pb1 ^ borrow;
    borrow = nb;
    nb = ((~tb2) & (pb2 | borrow)) | (pb2 & borrow);
    tb2 ^= pb2 ^ borrow;
    borrow = nb;
    nb  = (~tb3) & borrow;
    tb3 ^= borrow;
    tb4 ^= nb;
}

// ---------------------------------------------------------------------------
// Cell transition: map 5-bit accumulator + cell state → next state (SVE)
// ---------------------------------------------------------------------------

inline void store_hot_result_from_total_with_center_u64_sve(
    svbool_t pg,
    svuint64_t tb0, svuint64_t tb1, svuint64_t tb2,
    svuint64_t tb3, svuint64_t tb4,
    svuint64_t adult_word, svuint64_t juvenile_word, svuint64_t egg_word,
    svuint64_t& next_adult_word, svuint64_t& next_egg_word)
{
    const svuint64_t ones     = svdup_u64(~0ULL);
    const svuint64_t blocked  = svorr_x(pg, juvenile_word, egg_word);
    const svuint64_t empty_w  = sveor_x(pg, svorr_x(pg, adult_word, blocked), ones);
    const svuint64_t nb4      = sveor_x(pg, tb4, ones);
    const svuint64_t nb3      = sveor_x(pg, tb3, ones);
    const svuint64_t nb2      = sveor_x(pg, tb2, ones);
    const svuint64_t nb1      = sveor_x(pg, tb1, ones);
    const svuint64_t b1_or_b0 = svorr_x(pg, tb1, tb0);
    const svuint64_t b1_and_b0 = svand_x(pg, tb1, tb0);

    // adult survives if count in [5, 10]
    const svuint64_t adult_5_to_10 =
        svand_x(pg, nb4,
            svorr_x(pg,
                svand_x(pg, nb3, svand_x(pg, tb2, b1_or_b0)),
                svand_x(pg, tb3, svand_x(pg, nb2, sveor_x(pg, b1_and_b0, ones)))));

    // egg spawns in empty cell if count in [3, 5]
    const svuint64_t egg_3_to_5 =
        svand_x(pg, svand_x(pg, nb4, nb3),
            svorr_x(pg,
                svand_x(pg, nb2, b1_and_b0),
                svand_x(pg, tb2, nb1)));

    next_adult_word = svorr_x(pg, svand_x(pg, adult_word, adult_5_to_10), juvenile_word);
    next_egg_word   = svand_x(pg, empty_w, egg_3_to_5);
}

// ---------------------------------------------------------------------------
// Cell transition: scalar version
// ---------------------------------------------------------------------------

inline void store_hot_result_from_total_with_center(
    uint64_t tb0, uint64_t tb1, uint64_t tb2, uint64_t tb3, uint64_t tb4,
    uint64_t adult_word, uint64_t juvenile_word, uint64_t egg_word,
    uint64_t& next_adult_word, uint64_t& next_egg_word)
{
    const uint64_t blocked    = juvenile_word | egg_word;
    const uint64_t nb4        = ~tb4;
    const uint64_t nb3        = ~tb3;
    const uint64_t nb2        = ~tb2;
    const uint64_t nb1        = ~tb1;
    const uint64_t b1_or_b0   = tb1 | tb0;
    const uint64_t b1_and_b0  = tb1 & tb0;

    const uint64_t adult_5_to_10 =
        nb4 & ((nb3 & tb2 & b1_or_b0) | (tb3 & nb2 & ~b1_and_b0));
    const uint64_t egg_3_to_5 =
        nb4 & nb3 & ((nb2 & b1_and_b0) | (tb2 & nb1));

    next_adult_word = (adult_word & adult_5_to_10) | juvenile_word;
    next_egg_word   = ~(adult_word | blocked) & egg_3_to_5;
}

// ---------------------------------------------------------------------------
// Hot row computation — column-major tiled loop
//
// Loop order: tiles (row blocks) → word_index → row_index.
// The 5-wide vertical accumulator (tb0..tb4) lives in SVE registers for the
// entire (tile, word_index) inner scan.  No rolling_total arrays are written
// to memory.  Horizontal partials for both outgoing and incoming rows are
// recomputed from the adults bitplane on each iteration, which is cheaper
// than loading/storing a ring buffer because the adults data for rows r-2
// and r+3 is already warm in L1 cache.
// ---------------------------------------------------------------------------

__attribute__((noinline)) static void run_one_generation_hot_rows(
    int row_begin, int row_end,
    int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict adults    = current_adults.data();
    const uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs      = current_eggs.data();
    uint64_t* __restrict next_adults_ptr = next_adults.data();
    uint64_t* __restrict next_eggs_ptr   = next_eggs.data();

    const int hot_row_begin  = std::max(row_begin, 2);
    const int hot_row_end    = std::min(row_end, grid_size - 2);
    const int hot_word_begin = 1;
    const int hot_word_end   = words_per_row - 1;

    if (hot_row_begin >= hot_row_end || hot_word_begin >= hot_word_end) return;

    const int  sve_words = static_cast<int>(svcntd()); // 2 for 128-bit SVE
    const svbool_t pg    = svptrue_b64();

    // Largest word_index such that [word_index, word_index + sve_words) ⊆ hot range
    const int hot_word_count      = hot_word_end - hot_word_begin;
    const int aligned_hot_word_end =
        hot_word_begin + (hot_word_count / sve_words) * sve_words;

    // ── Outer tile loop ──────────────────────────────────────────────────────
    for (int tile_start = hot_row_begin;
         tile_start < hot_row_end;
         tile_start += TILE_ROWS)
    {
        const int tile_end = std::min(tile_start + TILE_ROWS, hot_row_end);

        // ── SVE word-column loop ─────────────────────────────────────────────
        // Each iteration processes sve_words adjacent word columns simultaneously.
        // vb0..vb4 are the in-register vertical accumulator — no memory traffic.
        for (int word_index = hot_word_begin;
             word_index < aligned_hot_word_end;
             word_index += sve_words)
        {
            // Seed accumulator: sum horizontal partials for the 5 rows centred
            // on tile_start.  These adults rows are ≤ 5 × 3 × 16 = 240 bytes —
            // always hot in L1 before the first tile iteration.
            svuint64_t vb0 = svdup_u64(0), vb1 = svdup_u64(0),
                       vb2 = svdup_u64(0), vb3 = svdup_u64(0),
                       vb4 = svdup_u64(0);

            for (int k = -2; k <= 2; ++k) {
                const uint64_t* seed_row =
                    adults + static_cast<size_t>(tile_start + k) * words_per_row;
                svuint64_t pb0, pb1, pb2;
                compute_horizontal_partial_pair2_u64_sve_from_adult_row(
                    pg, seed_row, word_index, pb0, pb1, pb2);
                add_horizontal_partial_to_total_u64_sve(
                    pg, pb0, pb1, pb2, vb0, vb1, vb2, vb3, vb4);
            }

            // ── Inner row scan ───────────────────────────────────────────────
            for (int row_index = tile_start; row_index < tile_end; ++row_index) {
                const size_t flat =
                    static_cast<size_t>(row_index) * words_per_row +
                    static_cast<size_t>(word_index);

                // Emit output using the in-register accumulator.
                // NT stores bypass L1/L2 — these planes won't be read this gen.
                svuint64_t next_adult_word, next_egg_word;
                store_hot_result_from_total_with_center_u64_sve(
                    pg, vb0, vb1, vb2, vb3, vb4,
                    svld1_u64(pg, adults    + flat),
                    svld1_u64(pg, juveniles + flat),
                    svld1_u64(pg, eggs      + flat),
                    next_adult_word, next_egg_word);

                svstnt1_u64(pg, next_adults_ptr + flat, next_adult_word);
                svstnt1_u64(pg, next_eggs_ptr   + flat, next_egg_word);

                // Slide the window forward — skip for the very last row of the
                // tile so we don't recompute beyond what the next tile re-seeds.
                if (row_index + 1 < tile_end) {
                    // Outgoing row (row_index - 2): adults data is 2 outer
                    // loop iterations old — still hot in L1 for typical grids.
                    const uint64_t* out_row =
                        adults + static_cast<size_t>(row_index - 2) * words_per_row;
                    // Incoming row (row_index + 3): already needed shortly by
                    // the output path, so pre-fetching is not necessary.
                    const uint64_t* in_row =
                        adults + static_cast<size_t>(row_index + 3) * words_per_row;

                    svuint64_t ob0, ob1, ob2, ib0, ib1, ib2;
                    compute_horizontal_partial_pair2_u64_sve_from_adult_row(
                        pg, out_row, word_index, ob0, ob1, ob2);
                    compute_horizontal_partial_pair2_u64_sve_from_adult_row(
                        pg, in_row,  word_index, ib0, ib1, ib2);

                    subtract_horizontal_partial_from_total_u64_sve(
                        pg, ob0, ob1, ob2, vb0, vb1, vb2, vb3, vb4);
                    add_horizontal_partial_to_total_u64_sve(
                        pg, ib0, ib1, ib2, vb0, vb1, vb2, vb3, vb4);
                }
            }
            // vb0..vb4 expire here — zero memory writeback
        }

        // ── Scalar tail for any word_indices not covered by SVE ──────────────
        for (int word_index = aligned_hot_word_end;
             word_index < hot_word_end;
             ++word_index)
        {
            uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0;

            for (int k = -2; k <= 2; ++k) {
                const uint64_t* seed_row =
                    adults + static_cast<size_t>(tile_start + k) * words_per_row;
                const HorizontalPartialWord p =
                    compute_horizontal_partial_scalar_from_adult_row(seed_row, word_index);
                add_horizontal_partial_to_total(
                    p.b0, p.b1, p.b2, b0, b1, b2, b3, b4);
            }

            for (int row_index = tile_start; row_index < tile_end; ++row_index) {
                const size_t flat =
                    static_cast<size_t>(row_index) * words_per_row + word_index;

                store_hot_result_from_total_with_center(
                    b0, b1, b2, b3, b4,
                    adults[flat], juveniles[flat], eggs[flat],
                    next_adults_ptr[flat], next_eggs_ptr[flat]);

                if (row_index + 1 < tile_end) {
                    const uint64_t* out_row =
                        adults + static_cast<size_t>(row_index - 2) * words_per_row;
                    const uint64_t* in_row =
                        adults + static_cast<size_t>(row_index + 3) * words_per_row;
                    const HorizontalPartialWord op =
                        compute_horizontal_partial_scalar_from_adult_row(out_row, word_index);
                    const HorizontalPartialWord ip =
                        compute_horizontal_partial_scalar_from_adult_row(in_row,  word_index);
                    subtract_horizontal_partial_from_total(
                        op.b0, op.b1, op.b2, b0, b1, b2, b3, b4);
                    add_horizontal_partial_to_total(
                        ip.b0, ip.b1, ip.b2, b0, b1, b2, b3, b4);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Edge handlers — unchanged from previous version
// ---------------------------------------------------------------------------

// Left edge: word 0 of hot rows [2, grid_size-2).
// Assigns rather than ORs so that juvenile promotion is folded in here,
// removing the need for a separate init pass.  valid_mask = ~0ULL<<2 covers
// bits 2..63; bits 0-1 (wrapping columns) are left at the juvenile value / 0
// so that the wrapping function can OR in the toroidal result correctly.
__attribute__((noinline)) static void run_one_generation_left_edge_valid_rows(
    int row_begin, int row_end,
    int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict adults    = current_adults.data();
    const uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs      = current_eggs.data();
    uint64_t* __restrict na = next_adults.data();
    uint64_t* __restrict ne = next_eggs.data();
    row_begin = std::max(row_begin, 2);
    row_end   = std::min(row_end,   grid_size - 2);
    const uint64_t valid_mask = ~0ULL << 2;

    for (int r = row_begin; r < row_end; ++r) {
        const size_t rm2 = static_cast<size_t>(r - 2) * words_per_row;
        const size_t rm1 = static_cast<size_t>(r - 1) * words_per_row;
        const size_t rb  = static_cast<size_t>(r)     * words_per_row;
        const size_t rp1 = static_cast<size_t>(r + 1) * words_per_row;
        const size_t rp2 = static_cast<size_t>(r + 2) * words_per_row;

        auto shift5 = [&](const uint64_t* base) -> std::array<uint64_t,5> {
            return {
                adults[base[0]] << 2,
                adults[base[0]] << 1,
                adults[base[0]],
                (adults[base[0]] >> 1) | (adults[base[1]] << 63),
                (adults[base[0]] >> 2) | (adults[base[1]] << 62),
            };
        };
        // Inline the 25 neighbour shifts
        uint64_t b0=0,b1=0,b2=0,b3=0,b4=0;
        auto add_mask = [&](uint64_t m){
            uint64_t c=b0&m; b0^=m; m=c;
            c=b1&m; b1^=m; m=c;
            c=b2&m; b2^=m; m=c;
            c=b3&m; b3^=m; m=c;
            b4^=m;
        };

        // top row (r-2)
        add_mask(adults[rm2]<<2);
        add_mask(adults[rm2]<<1);
        add_mask(adults[rm2]);
        add_mask((adults[rm2]>>1)|(adults[rm2+1]<<63));
        add_mask((adults[rm2]>>2)|(adults[rm2+1]<<62));
        // upper row (r-1)
        add_mask(adults[rm1]<<2);
        add_mask(adults[rm1]<<1);
        add_mask(adults[rm1]);
        add_mask((adults[rm1]>>1)|(adults[rm1+1]<<63));
        add_mask((adults[rm1]>>2)|(adults[rm1+1]<<62));
        // centre row (r) — no self
        add_mask(adults[rb]<<2);
        add_mask(adults[rb]<<1);
        add_mask((adults[rb]>>1)|(adults[rb+1]<<63));
        add_mask((adults[rb]>>2)|(adults[rb+1]<<62));
        // lower row (r+1)
        add_mask(adults[rp1]<<2);
        add_mask(adults[rp1]<<1);
        add_mask(adults[rp1]);
        add_mask((adults[rp1]>>1)|(adults[rp1+1]<<63));
        add_mask((adults[rp1]>>2)|(adults[rp1+1]<<62));
        // bottom row (r+2)
        add_mask(adults[rp2]<<2);
        add_mask(adults[rp2]<<1);
        add_mask(adults[rp2]);
        add_mask((adults[rp2]>>1)|(adults[rp2+1]<<63));
        add_mask((adults[rp2]>>2)|(adults[rp2+1]<<62));

        const uint64_t aw       = adults[rb];
        const uint64_t blocked  = juveniles[rb] | eggs[rb];
        const uint64_t av       = aw    & valid_mask;
        const uint64_t ev       = ~(aw | blocked) & valid_mask;
        const uint64_t ge3  = b4|b3|b2|(b1&b0);
        const uint64_t ge4  = b4|b3|b2;
        const uint64_t ge6  = b4|b3|(b2&b1);
        const uint64_t ge10 = b4|(b3&(b2|b1));

        na[rb] = juveniles[rb] | (av & ge4 & ~ge10);
        ne[rb] = ev & ge3 & ~ge6;
    }
}

// Right edge: last word of hot rows.
__attribute__((noinline)) static void run_one_generation_right_edge_valid_rows(
    int row_begin, int row_end,
    int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict adults    = current_adults.data();
    const uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs      = current_eggs.data();
    uint64_t* __restrict na = next_adults.data();
    uint64_t* __restrict ne = next_eggs.data();
    row_begin = std::max(row_begin, 2);
    row_end   = std::min(row_end,   grid_size - 2);
    const int lw = words_per_row - 1;
    const int pw = words_per_row - 2;
    const uint64_t valid_mask = (1ULL << 62) - 1ULL;

    for (int r = row_begin; r < row_end; ++r) {
        const size_t rm2 = static_cast<size_t>(r - 2) * words_per_row;
        const size_t rm1 = static_cast<size_t>(r - 1) * words_per_row;
        const size_t rb  = static_cast<size_t>(r)     * words_per_row;
        const size_t rp1 = static_cast<size_t>(r + 1) * words_per_row;
        const size_t rp2 = static_cast<size_t>(r + 2) * words_per_row;

        uint64_t b0=0,b1=0,b2=0,b3=0,b4=0;
        auto add_mask = [&](uint64_t m){
            uint64_t c=b0&m; b0^=m; m=c;
            c=b1&m; b1^=m; m=c;
            c=b2&m; b2^=m; m=c;
            c=b3&m; b3^=m; m=c;
            b4^=m;
        };

        add_mask((adults[rm2+lw]<<2)|(adults[rm2+pw]>>62));
        add_mask((adults[rm2+lw]<<1)|(adults[rm2+pw]>>63));
        add_mask(adults[rm2+lw]);
        add_mask(adults[rm2+lw]>>1);
        add_mask(adults[rm2+lw]>>2);

        add_mask((adults[rm1+lw]<<2)|(adults[rm1+pw]>>62));
        add_mask((adults[rm1+lw]<<1)|(adults[rm1+pw]>>63));
        add_mask(adults[rm1+lw]);
        add_mask(adults[rm1+lw]>>1);
        add_mask(adults[rm1+lw]>>2);

        add_mask((adults[rb+lw]<<2)|(adults[rb+pw]>>62));
        add_mask((adults[rb+lw]<<1)|(adults[rb+pw]>>63));
        add_mask(adults[rb+lw]>>1);
        add_mask(adults[rb+lw]>>2);

        add_mask((adults[rp1+lw]<<2)|(adults[rp1+pw]>>62));
        add_mask((adults[rp1+lw]<<1)|(adults[rp1+pw]>>63));
        add_mask(adults[rp1+lw]);
        add_mask(adults[rp1+lw]>>1);
        add_mask(adults[rp1+lw]>>2);

        add_mask((adults[rp2+lw]<<2)|(adults[rp2+pw]>>62));
        add_mask((adults[rp2+lw]<<1)|(adults[rp2+pw]>>63));
        add_mask(adults[rp2+lw]);
        add_mask(adults[rp2+lw]>>1);
        add_mask(adults[rp2+lw]>>2);

        const size_t ei = rb + lw;
        const uint64_t aw      = adults[ei];
        const uint64_t blocked = juveniles[ei] | eggs[ei];
        const uint64_t av      = aw    & valid_mask;
        const uint64_t ev      = ~(aw | blocked) & valid_mask;
        const uint64_t ge3  = b4|b3|b2|(b1&b0);
        const uint64_t ge4  = b4|b3|b2;
        const uint64_t ge6  = b4|b3|(b2&b1);
        const uint64_t ge10 = b4|(b3&(b2|b1));

        na[ei] = juveniles[ei] | (av & ge4 & ~ge10);
        ne[ei] = ev & ge3 & ~ge6;
    }
}

// Toroidal wrapping rows: top 2, bottom 2, and the wrapping columns of hot
// interior rows.  For non-hot rows, this function owns the full init (juvenile
// promotion + zero of eggs) — no separate init pass is needed.
__attribute__((noinline)) static void run_one_generation_wrapping_rows(
    int row_begin, int row_end,
    int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict adults    = current_adults.data();
    const uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs      = current_eggs.data();
    uint64_t* __restrict na = next_adults.data();
    uint64_t* __restrict ne = next_eggs.data();
    const int top_end    = 2;
    const int bot_begin  = grid_size - 2;

    auto process_cell = [&](int r, int c) {
        const size_t wi   = static_cast<size_t>(r) * words_per_row + (c >> 6);
        const uint64_t cm = 1ULL << (c & 63);
        if ((juveniles[wi] | eggs[wi]) & cm) return;
        const int n = count_adult_neighbors_naive(
            current_adults, grid_size, words_per_row, r, c);
        if (adults[wi] & cm) {
            if (n >= 4 && n <= 9) na[wi] |= cm;
        } else if (n >= 3 && n <= 5) {
            ne[wi] |= cm;
        }
    };

    // Top non-hot rows — init then process all columns
    for (int r = row_begin; r < row_end && r < top_end; ++r) {
        const size_t rb = static_cast<size_t>(r) * words_per_row;
        for (int w = 0; w < words_per_row; ++w) {
            na[rb + w] = juveniles[rb + w];
            ne[rb + w] = 0ULL;
        }
        for (int c = 0; c < grid_size; ++c) process_cell(r, c);
    }

    // Interior hot rows — wrapping columns only (left/right edge already written)
    const int int_begin = std::max(row_begin, 2);
    const int int_end   = std::min(row_end,   grid_size - 2);
    for (int r = int_begin; r < int_end; ++r) {
        for (int c = 0;             c < 2;          ++c) process_cell(r, c);
        for (int c = grid_size - 2; c < grid_size;  ++c) process_cell(r, c);
    }

    // Bottom non-hot rows — init then process all columns
    for (int r = std::max(row_begin, bot_begin); r < row_end; ++r) {
        const size_t rb = static_cast<size_t>(r) * words_per_row;
        for (int w = 0; w < words_per_row; ++w) {
            na[rb + w] = juveniles[rb + w];
            ne[rb + w] = 0ULL;
        }
        for (int c = 0; c < grid_size; ++c) process_cell(r, c);
    }
}

__attribute__((noinline)) static void run_one_generation_edge_rows(
    int row_begin, int row_end,
    int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    run_one_generation_left_edge_valid_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs,
        next_adults, next_eggs);

    run_one_generation_right_edge_valid_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs,
        next_adults, next_eggs);

    run_one_generation_wrapping_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs,
        next_adults, next_eggs);
}

// ---------------------------------------------------------------------------
// Per-thread generation chunk — no scratch parameter
// ---------------------------------------------------------------------------

static void process_generation_chunk(
    int row_begin, int row_end,
    int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    run_one_generation_hot_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs,
        next_adults, next_eggs);

    run_one_generation_edge_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs,
        next_adults, next_eggs);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "Usage: %s <input.bin> <output.bin> [generations]\n", argv[0]);
        return 1;
    }

    int generations = 10000;
    if (argc == 4) {
        char* end = nullptr;
        long g = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || g <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        generations = static_cast<int>(g);
    }

    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::fprintf(stderr, "Error: cannot open input '%s'\n", argv[1]);
        return 2;
    }
    uint64_t width = 0, height = 0;
    if (std::fread(&width,  sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, fin) != 1) {
        std::fprintf(stderr, "Error: input too short (header)\n");
        std::fclose(fin); return 3;
    }
    if (width == 0 || width != height) {
        std::fprintf(stderr,
            "Error: grid must be square and non-empty, got %" PRIu64 " × %" PRIu64 "\n",
            width, height);
        std::fclose(fin); return 3;
    }

    const int    grid_size  = static_cast<int>(width);
    const size_t cell_count = width * width;

    std::vector<uint8_t> input_grid(cell_count);
    if (std::fread(input_grid.data(), 1, cell_count, fin) != cell_count) {
        std::fprintf(stderr, "Error: input too short (cells)\n");
        std::fclose(fin); return 4;
    }
    std::fclose(fin);

    const int    words_per_row = grid_size >> 6;
    const size_t total_words   = static_cast<size_t>(grid_size) * words_per_row;

    alignas(64) BitPlane current_adults  (total_words, 0ULL);
    alignas(64) BitPlane current_juveniles(total_words, 0ULL);
    alignas(64) BitPlane current_eggs    (total_words, 0ULL);
    alignas(64) BitPlane next_adults     (total_words, 0ULL);
    alignas(64) BitPlane next_eggs       (total_words, 0ULL);

    pack_input_into_bit_planes(
        input_grid, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs);

    auto start_time = std::chrono::steady_clock::now();

    const int thread_count =
        std::max(1, std::min(TARGET_THREAD_COUNT, grid_size));
    std::vector<RowRange> row_ranges(static_cast<size_t>(thread_count));
    for (int wid = 0; wid < thread_count; ++wid)
        row_ranges[static_cast<size_t>(wid)] =
            compute_row_range(grid_size, thread_count, wid);

    std::barrier generation_start_barrier(thread_count);
    std::barrier generation_done_barrier (thread_count);
    std::atomic<bool> stop_workers{false};

    auto worker_body = [&](int worker_id) {
        pin_current_thread_to_cpu(worker_id);
        const RowRange owned = row_ranges[static_cast<size_t>(worker_id)];

        for (;;) {
            generation_start_barrier.arrive_and_wait();
            if (stop_workers.load(std::memory_order_acquire)) break;

            process_generation_chunk(
                owned.begin, owned.end,
                grid_size, words_per_row,
                current_adults, current_juveniles, current_eggs,
                next_adults, next_eggs);

            generation_done_barrier.arrive_and_wait();
        }
    };

    pin_current_thread_to_cpu(0);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(std::max(0, thread_count - 1)));
    for (int wid = 1; wid < thread_count; ++wid)
        workers.emplace_back(worker_body, wid);

    const RowRange main_owned = row_ranges.front();
    for (int gen = 0; gen < generations; ++gen) {
        generation_start_barrier.arrive_and_wait();

        process_generation_chunk(
            main_owned.begin, main_owned.end,
            grid_size, words_per_row,
            current_adults, current_juveniles, current_eggs,
            next_adults, next_eggs);

        generation_done_barrier.arrive_and_wait();

        current_adults.swap(next_adults);
        current_juveniles.swap(current_eggs);
        current_eggs.swap(next_eggs);
    }

    stop_workers.store(true, std::memory_order_release);
    generation_start_barrier.arrive_and_wait();
    for (std::thread& t : workers) t.join();

    auto end_time = std::chrono::steady_clock::now();
    std::printf("%.3f ms\n",
        std::chrono::duration<double, std::milli>(end_time - start_time).count());

    std::vector<uint8_t> output_grid;
    unpack_bit_planes_to_output_grid(
        grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs,
        output_grid);

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output '%s'\n", argv[2]);
        return 5;
    }
    if (std::fwrite(&width,  sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(&height, sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(output_grid.data(), 1, cell_count, fout) != cell_count) {
        std::fprintf(stderr, "Error: write error on '%s'\n", argv[2]);
        std::fclose(fout); return 6;
    }
    std::fclose(fout);
    return 0;
}