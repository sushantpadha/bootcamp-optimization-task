// spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2.cpp
//
// Changes from previous version:
//   - VerticalAccumulatorRow is now AoS (split into VerticalAccumulatorLow {b0,b1,b2}
//     and VerticalAccumulatorHigh {b3,b4}) enabling svld3/svst3 and svld2/svst2
//     instead of five separate svld1/svst1 calls.
//   - build_vertical_accumulator_row is now fully SVE2 vectorized (was scalar).
//   - Scalar tail fallbacks removed from compute_horizontal_partial_row and
//     run_one_generation_hot_rows; replaced with a single svwhilelt_b64 predicated
//     loop that handles both full and partial iterations.
//   - ring_head increment uses a conditional branch instead of modulo-5.
//   - Dead scalar helpers (carry_save_add, add_horizontal_partial_to_total,
//     subtract_horizontal_partial_from_total) removed.
//
// INPUT FORMAT:
//   Binary file, little-endian:
//     bytes  0– 7: uint64_t width
//     bytes  8–15: uint64_t height (must equal width; only square grids are supported)
//     bytes 16–  : width × height bytes of cell data, row-major.
//                  Each byte encodes one cell: 0=EMPTY, 1=EGG, 2=JUVENILE, 3=ADULT.
//
// OUTPUT FORMAT:
//   Same binary format as input, encoding the grid after the requested number of generations.
//
// USAGE:
//   spawn_sim <input.bin> <output.bin> [generations]
//   generations defaults to 10000 if not specified.
//
// TIMING:
//   Prints the wall-clock time of the simulation (excluding file I/O and the
//   input/output packing steps) to stdout as a single line: "<N.NNN> ms"

#include <array>
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cinttypes>
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

static constexpr uint8_t EMPTY = 0;
static constexpr uint8_t EGG = 1;
static constexpr uint8_t JUVENILE = 2;
static constexpr uint8_t ADULT = 3;

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept
    {
    }

    [[nodiscard]] T* allocate(std::size_t count)
    {
        if (count > static_cast<std::size_t>(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        void* pointer = nullptr;
        if (posix_memalign(&pointer, Alignment, count * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }

        return static_cast<T*>(pointer);
    }

    void deallocate(T* pointer, std::size_t) noexcept
    {
        std::free(pointer);
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&)
{
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&)
{
    return false;
}

using BitPlane = std::vector<uint64_t, AlignedAllocator<uint64_t, 64>>;

struct RowRange {
    int begin = 0;
    int end = 0;
};

static constexpr int TARGET_THREAD_COUNT = 8;

// ---------------------------------------------------------------------------
// Horizontal partial: 3-bit AoS word (unchanged)
// ---------------------------------------------------------------------------

struct HorizontalPartialWord {
    uint64_t b0 = 0ULL;
    uint64_t b1 = 0ULL;
    uint64_t b2 = 0ULL;
};

static_assert(
    sizeof(HorizontalPartialWord) == sizeof(uint64_t) * 3,
    "HorizontalPartialWord must stay tightly packed for interleaved SVE2 loads/stores");

using HorizontalPartialWordBuffer =
    std::vector<HorizontalPartialWord, AlignedAllocator<HorizontalPartialWord, 64>>;

struct HorizontalPartialRow {
    HorizontalPartialWordBuffer words;

    HorizontalPartialRow() = default;

    explicit HorizontalPartialRow(int words_per_row)
        : words(static_cast<size_t>(words_per_row))
    {
    }
};

// ---------------------------------------------------------------------------
// Vertical accumulator: 5-bit AoS split into {b0,b1,b2} and {b3,b4} so that
// svld3/svst3 and svld2/svst2 can be used directly.  This mirrors the layout
// of HorizontalPartialWord and avoids five separate svld1/svst1 calls.
// ---------------------------------------------------------------------------

struct VerticalAccumulatorLow {
    uint64_t b0 = 0ULL;
    uint64_t b1 = 0ULL;
    uint64_t b2 = 0ULL;
};

struct VerticalAccumulatorHigh {
    uint64_t b3 = 0ULL;
    uint64_t b4 = 0ULL;
};

static_assert(
    sizeof(VerticalAccumulatorLow) == sizeof(uint64_t) * 3,
    "VerticalAccumulatorLow must be tightly packed for svld3/svst3");
static_assert(
    sizeof(VerticalAccumulatorHigh) == sizeof(uint64_t) * 2,
    "VerticalAccumulatorHigh must be tightly packed for svld2/svst2");

using VerticalAccumulatorLowBuffer =
    std::vector<VerticalAccumulatorLow, AlignedAllocator<VerticalAccumulatorLow, 64>>;
using VerticalAccumulatorHighBuffer =
    std::vector<VerticalAccumulatorHigh, AlignedAllocator<VerticalAccumulatorHigh, 64>>;

struct VerticalAccumulatorRow {
    VerticalAccumulatorLowBuffer low;
    VerticalAccumulatorHighBuffer high;

    VerticalAccumulatorRow() = default;

    explicit VerticalAccumulatorRow(int words_per_row)
        : low(static_cast<size_t>(words_per_row))
        , high(static_cast<size_t>(words_per_row))
    {
    }
};

struct HotRowScratch {
    std::array<HorizontalPartialRow, 5> row_partials;
    HorizontalPartialRow incoming_partial;
    VerticalAccumulatorRow rolling_total;

    HotRowScratch() = default;

    explicit HotRowScratch(int words_per_row)
    {
        for (HorizontalPartialRow& row_partial : row_partials) {
            row_partial = HorizontalPartialRow(words_per_row);
        }
        incoming_partial = HorizontalPartialRow(words_per_row);
        rolling_total = VerticalAccumulatorRow(words_per_row);
    }
};

static RowRange compute_row_range(int grid_size, int thread_count, int worker_id)
{
    const int base_rows = grid_size / thread_count;
    const int remainder = grid_size % thread_count;
    const int begin =
        worker_id * base_rows + std::min(worker_id, remainder);
    const int count = base_rows + (worker_id < remainder ? 1 : 0);
    return RowRange{begin, begin + count};
}

static void pin_current_thread_to_cpu(int worker_id)
{
    cpu_set_t affinity_mask;
    CPU_ZERO(&affinity_mask);
    CPU_SET(worker_id, &affinity_mask);
    (void)pthread_setaffinity_np(
        pthread_self(),
        sizeof(affinity_mask),
        &affinity_mask);
}

inline int wrap_coordinate(int coordinate, int grid_size)
{
    return (coordinate + grid_size) & (grid_size - 1);
}

inline size_t get_flat_word_index(int row_index, int col_index, int words_per_row)
{
    int word_index_in_row = col_index >> 6;
    return static_cast<size_t>(row_index) * words_per_row + word_index_in_row;
}

inline uint64_t get_bit_mask(int col_index)
{
    int bit_index_in_word = col_index & 63;
    return 1ULL << bit_index_in_word;
}

inline bool is_cell_set(
    const BitPlane& bit_plane,
    int words_per_row,
    int row_index,
    int col_index)
{
    size_t flat_word_index =
        get_flat_word_index(row_index, col_index, words_per_row);
    uint64_t bit_mask = get_bit_mask(col_index);
    return (bit_plane[flat_word_index] & bit_mask) != 0;
}

inline void set_cell_bit(
    BitPlane& bit_plane,
    int words_per_row,
    int row_index,
    int col_index)
{
    size_t flat_word_index =
        get_flat_word_index(row_index, col_index, words_per_row);
    uint64_t bit_mask = get_bit_mask(col_index);
    bit_plane[flat_word_index] |= bit_mask;
}

static void pack_input_into_bit_planes(
    const std::vector<uint8_t>& input_grid,
    int grid_size,
    int words_per_row,
    BitPlane& current_adults,
    BitPlane& current_juveniles,
    BitPlane& current_eggs)
{
    for (int row_index = 0; row_index < grid_size; ++row_index) {
        for (int col_index = 0; col_index < grid_size; ++col_index) {
            uint8_t cell_state =
                input_grid[static_cast<size_t>(row_index) * grid_size + col_index];

            if (cell_state == ADULT) {
                set_cell_bit(current_adults, words_per_row, row_index, col_index);
            } else if (cell_state == JUVENILE) {
                set_cell_bit(current_juveniles, words_per_row, row_index, col_index);
            } else if (cell_state == EGG) {
                set_cell_bit(current_eggs, words_per_row, row_index, col_index);
            }
        }
    }
}

static int count_adult_neighbors_naive(
    const BitPlane& current_adults,
    int grid_size,
    int words_per_row,
    int center_row,
    int center_col)
{
    int adult_neighbor_count = 0;

    for (int row_offset = -2; row_offset <= 2; ++row_offset) {
        int neighbor_row = wrap_coordinate(center_row + row_offset, grid_size);

        for (int col_offset = -2; col_offset <= 2; ++col_offset) {
            if (row_offset == 0 && col_offset == 0) {
                continue;
            }

            int neighbor_col = wrap_coordinate(center_col + col_offset, grid_size);
            if (is_cell_set(current_adults, words_per_row, neighbor_row, neighbor_col)) {
                ++adult_neighbor_count;
            }
        }
    }

    return adult_neighbor_count;
}

// ---------------------------------------------------------------------------
// SVE2 carry-save adder (used in compute_horizontal_partial_row)
// ---------------------------------------------------------------------------

inline void carry_save_add_u64_sve(
    svbool_t pg,
    svuint64_t a,
    svuint64_t b,
    svuint64_t c,
    svuint64_t& sum,
    svuint64_t& carry)
{
    sum = sveor_x(pg, sveor_x(pg, a, b), c);

    const svuint64_t ab = svand_x(pg, a, b);
    const svuint64_t ac = svand_x(pg, a, c);
    const svuint64_t bc = svand_x(pg, b, c);
    carry = svorr_x(pg, svorr_x(pg, ab, ac), bc);
}

// ---------------------------------------------------------------------------
// SVE2 accumulator add/subtract (ripple carry; operates on full SVE vectors)
// ---------------------------------------------------------------------------

inline void add_horizontal_partial_to_total_u64_sve(
    svbool_t pg,
    svuint64_t partial_b0,
    svuint64_t partial_b1,
    svuint64_t partial_b2,
    svuint64_t& total_b0,
    svuint64_t& total_b1,
    svuint64_t& total_b2,
    svuint64_t& total_b3,
    svuint64_t& total_b4)
{
    svuint64_t carry = svand_x(pg, total_b0, partial_b0);
    total_b0 = sveor_x(pg, total_b0, partial_b0);

    carry_save_add_u64_sve(
        pg,
        total_b1,
        partial_b1,
        carry,
        total_b1,
        carry);
    carry_save_add_u64_sve(
        pg,
        total_b2,
        partial_b2,
        carry,
        total_b2,
        carry);

    const svuint64_t carry4 = svand_x(pg, total_b3, carry);
    total_b3 = sveor_x(pg, total_b3, carry);
    total_b4 = sveor_x(pg, total_b4, carry4);
}

inline void subtract_horizontal_partial_from_total_u64_sve(
    svbool_t pg,
    svuint64_t partial_b0,
    svuint64_t partial_b1,
    svuint64_t partial_b2,
    svuint64_t& total_b0,
    svuint64_t& total_b1,
    svuint64_t& total_b2,
    svuint64_t& total_b3,
    svuint64_t& total_b4)
{
    const svuint64_t all_ones = svdup_u64(~0ULL);
    svuint64_t borrow =
        svand_x(pg, sveor_x(pg, total_b0, all_ones), partial_b0);
    total_b0 = sveor_x(pg, total_b0, partial_b0);

    svuint64_t next_borrow = svorr_x(
        pg,
        svand_x(
            pg,
            sveor_x(pg, total_b1, all_ones),
            svorr_x(pg, partial_b1, borrow)),
        svand_x(pg, partial_b1, borrow));
    total_b1 = sveor_x(pg, sveor_x(pg, total_b1, partial_b1), borrow);
    borrow = next_borrow;

    next_borrow = svorr_x(
        pg,
        svand_x(
            pg,
            sveor_x(pg, total_b2, all_ones),
            svorr_x(pg, partial_b2, borrow)),
        svand_x(pg, partial_b2, borrow));
    total_b2 = sveor_x(pg, sveor_x(pg, total_b2, partial_b2), borrow);
    borrow = next_borrow;

    next_borrow = svand_x(pg, sveor_x(pg, total_b3, all_ones), borrow);
    total_b3 = sveor_x(pg, total_b3, borrow);
    total_b4 = sveor_x(pg, total_b4, next_borrow);
}

// ---------------------------------------------------------------------------
// SVE2 transition logic (unchanged)
// ---------------------------------------------------------------------------

inline void store_hot_result_from_total_with_center_u64_sve(
    svbool_t pg,
    svuint64_t total_b0,
    svuint64_t total_b1,
    svuint64_t total_b2,
    svuint64_t total_b3,
    svuint64_t total_b4,
    svuint64_t adult_word,
    svuint64_t juvenile_word,
    svuint64_t egg_word,
    svuint64_t& next_adult_word,
    svuint64_t& next_egg_word)
{
    const svuint64_t all_ones = svdup_u64(~0ULL);
    const svuint64_t blocked_word = svorr_x(pg, juvenile_word, egg_word);
    const svuint64_t occupied_word = svorr_x(pg, adult_word, blocked_word);
    const svuint64_t empty_word = sveor_x(pg, occupied_word, all_ones);
    const svuint64_t not_b4 = sveor_x(pg, total_b4, all_ones);
    const svuint64_t not_b3 = sveor_x(pg, total_b3, all_ones);
    const svuint64_t not_b2 = sveor_x(pg, total_b2, all_ones);
    const svuint64_t not_b1 = sveor_x(pg, total_b1, all_ones);
    const svuint64_t b1_or_b0 = svorr_x(pg, total_b1, total_b0);
    const svuint64_t b1_and_b0 = svand_x(pg, total_b1, total_b0);

    const svuint64_t adult_5_to_10 =
        svand_x(
            pg,
            not_b4,
            svorr_x(
                pg,
                svand_x(pg, not_b3, svand_x(pg, total_b2, b1_or_b0)),
                svand_x(
                    pg,
                    total_b3,
                    svand_x(pg, not_b2, sveor_x(pg, b1_and_b0, all_ones)))));
    const svuint64_t egg_3_to_5 =
        svand_x(
            pg,
            svand_x(pg, not_b4, not_b3),
            svorr_x(
                pg,
                svand_x(pg, not_b2, b1_and_b0),
                svand_x(pg, total_b2, not_b1)));

    next_adult_word = svorr_x(
        pg,
        svand_x(pg, adult_word, adult_5_to_10),
        juvenile_word);
    next_egg_word = svand_x(pg, empty_word, egg_3_to_5);
}

// ---------------------------------------------------------------------------
// Scalar transition logic (used by edge/wrapping paths, unchanged)
// ---------------------------------------------------------------------------

inline void store_hot_result_from_total_with_center(
    uint64_t total_b0,
    uint64_t total_b1,
    uint64_t total_b2,
    uint64_t total_b3,
    uint64_t total_b4,
    uint64_t adult_word,
    uint64_t juvenile_word,
    uint64_t egg_word,
    uint64_t& next_adult_word,
    uint64_t& next_egg_word)
{
    const uint64_t blocked_word = juvenile_word | egg_word;
    const uint64_t not_b4 = ~total_b4;
    const uint64_t not_b3 = ~total_b3;
    const uint64_t not_b2 = ~total_b2;
    const uint64_t not_b1 = ~total_b1;
    const uint64_t b1_or_b0 = total_b1 | total_b0;
    const uint64_t b1_and_b0 = total_b1 & total_b0;

    const uint64_t adult_5_to_10 =
        not_b4 &
        ((not_b3 & total_b2 & b1_or_b0) |
         (total_b3 & not_b2 & ~b1_and_b0));
    const uint64_t egg_3_to_5 =
        not_b4 &
        not_b3 &
        ((not_b2 & b1_and_b0) |
         (total_b2 & not_b1));

    next_adult_word = (adult_word & adult_5_to_10) | juvenile_word;
    next_egg_word = ~(adult_word | blocked_word) & egg_3_to_5;
}

// ---------------------------------------------------------------------------
// compute_horizontal_partial_row
//
// Single SVE2 predicated loop replaces the previous full-predicate loop +
// scalar tail.  svwhilelt_b64 generates an all-true predicate for full
// iterations and a partial predicate for the tail, so no separate path is
// needed.
// ---------------------------------------------------------------------------

static void compute_horizontal_partial_row(
    int source_row,
    int words_per_row,
    const BitPlane& current_adults,
    HorizontalPartialRow& row_partial)
{
    const uint64_t* __restrict adults = current_adults.data();
    HorizontalPartialWord* __restrict partial_words = row_partial.words.data();
    const int hot_word_begin = 1;
    const int hot_word_end = words_per_row - 1;

    if (hot_word_begin >= hot_word_end) {
        return;
    }

    const uint64_t* __restrict adult_row =
        adults + static_cast<size_t>(source_row) * words_per_row;

    const int sve_words = static_cast<int>(svcntd());
    for (int word_index = hot_word_begin; word_index < hot_word_end; word_index += sve_words) {
        const svbool_t pg = svwhilelt_b64(
            static_cast<uint64_t>(word_index),
            static_cast<uint64_t>(hot_word_end));

        const svuint64_t prev = svld1_u64(pg, adult_row + word_index - 1);
        const svuint64_t curr = svld1_u64(pg, adult_row + word_index);
        const svuint64_t next = svld1_u64(pg, adult_row + word_index + 1);

        const svuint64_t left_2 =
            svorr_x(pg, svlsl_n_u64_x(pg, curr, 2), svlsr_n_u64_x(pg, prev, 62));
        const svuint64_t left_1 =
            svorr_x(pg, svlsl_n_u64_x(pg, curr, 1), svlsr_n_u64_x(pg, prev, 63));
        const svuint64_t right_1 =
            svorr_x(pg, svlsr_n_u64_x(pg, curr, 1), svlsl_n_u64_x(pg, next, 63));
        const svuint64_t right_2 =
            svorr_x(pg, svlsr_n_u64_x(pg, curr, 2), svlsl_n_u64_x(pg, next, 62));

        svuint64_t sum_1;
        svuint64_t carry_1;
        svuint64_t sum_2;
        svuint64_t carry_2;

        carry_save_add_u64_sve(pg, left_2, left_1, curr, sum_1, carry_1);
        carry_save_add_u64_sve(pg, right_1, right_2, sum_1, sum_2, carry_2);

        svst3_u64(
            pg,
            reinterpret_cast<uint64_t*>(partial_words + word_index),
            svcreate3_u64(
                sum_2,
                sveor_x(pg, carry_1, carry_2),
                svand_x(pg, carry_1, carry_2)));
    }
}

// ---------------------------------------------------------------------------
// build_vertical_accumulator_row
//
// Fully SVE2 vectorized (was entirely scalar).  Uses svld3 to load each
// HorizontalPartialRow in AoS layout, accumulates with the SVE2 add helper,
// then stores to the AoS VerticalAccumulatorRow with svst3 + svst2.
// ---------------------------------------------------------------------------

static void build_vertical_accumulator_row(
    int words_per_row,
    const std::array<HorizontalPartialRow, 5>& row_partials,
    VerticalAccumulatorRow& rolling_total)
{
    const int hot_word_begin = 1;
    const int hot_word_end = words_per_row - 1;

    if (hot_word_begin >= hot_word_end) {
        return;
    }

    VerticalAccumulatorLow* __restrict low_ptr = rolling_total.low.data();
    VerticalAccumulatorHigh* __restrict high_ptr = rolling_total.high.data();
    const int sve_words = static_cast<int>(svcntd());

    for (int word_index = hot_word_begin; word_index < hot_word_end; word_index += sve_words) {
        const svbool_t pg = svwhilelt_b64(
            static_cast<uint64_t>(word_index),
            static_cast<uint64_t>(hot_word_end));

        svuint64_t b0 = svdup_u64(0ULL);
        svuint64_t b1 = svdup_u64(0ULL);
        svuint64_t b2 = svdup_u64(0ULL);
        svuint64_t b3 = svdup_u64(0ULL);
        svuint64_t b4 = svdup_u64(0ULL);

        for (const HorizontalPartialRow& row_partial : row_partials) {
            const svuint64x3_t partial = svld3_u64(
                pg,
                reinterpret_cast<const uint64_t*>(row_partial.words.data() + word_index));
            add_horizontal_partial_to_total_u64_sve(
                pg,
                svget3_u64(partial, 0),
                svget3_u64(partial, 1),
                svget3_u64(partial, 2),
                b0, b1, b2, b3, b4);
        }

        svst3_u64(
            pg,
            reinterpret_cast<uint64_t*>(low_ptr + word_index),
            svcreate3_u64(b0, b1, b2));
        svst2_u64(
            pg,
            reinterpret_cast<uint64_t*>(high_ptr + word_index),
            svcreate2_u64(b3, b4));
    }
}

// ---------------------------------------------------------------------------
// run_one_generation_hot_rows
//
// Changes from previous version:
//   - Five separate total_bN_ptr raw pointers replaced by acc_low_ptr and
//     acc_high_ptr pointing into the AoS VerticalAccumulatorRow.
//   - simd_word_end removed; inner word loop uses svwhilelt_b64 predicate so
//     a single loop body handles both full and partial (tail) iterations.
//   - Scalar tail loop removed entirely.
//   - ring_head increment uses conditional branch instead of modulo-5.
// ---------------------------------------------------------------------------

__attribute__((noinline)) static void run_one_generation_hot_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs,
    HotRowScratch& hot_row_scratch)
{
    const uint64_t* __restrict adults = current_adults.data();
    const uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs = current_eggs.data();
    uint64_t* __restrict next_adults_ptr = next_adults.data();
    uint64_t* __restrict next_eggs_ptr = next_eggs.data();
    VerticalAccumulatorLow* __restrict acc_low_ptr =
        hot_row_scratch.rolling_total.low.data();
    VerticalAccumulatorHigh* __restrict acc_high_ptr =
        hot_row_scratch.rolling_total.high.data();
    const int hot_row_begin = std::max(row_begin, 2);
    const int hot_row_end = std::min(row_end, grid_size - 2);
    const int hot_word_begin = 1;
    const int hot_word_end = words_per_row - 1;

    if (hot_row_begin >= hot_row_end || hot_word_begin >= hot_word_end) {
        return;
    }

    for (int ring_offset = 0; ring_offset < 5; ++ring_offset) {
        compute_horizontal_partial_row(
            hot_row_begin + ring_offset - 2,
            words_per_row,
            current_adults,
            hot_row_scratch.row_partials[static_cast<size_t>(ring_offset)]);
    }

    build_vertical_accumulator_row(
        words_per_row,
        hot_row_scratch.row_partials,
        hot_row_scratch.rolling_total);

    const int sve_words = static_cast<int>(svcntd());

    size_t ring_head = 0;
    for (int row_index = hot_row_begin; row_index + 1 < hot_row_end; ++row_index) {
        const size_t row_base =
            static_cast<size_t>(row_index) * words_per_row;
        const HorizontalPartialWord* __restrict outgoing_partial_words =
            hot_row_scratch.row_partials[ring_head].words.data();
        const HorizontalPartialWord* __restrict incoming_partial_words =
            hot_row_scratch.incoming_partial.words.data();

        compute_horizontal_partial_row(
            row_index + 3,
            words_per_row,
            current_adults,
            hot_row_scratch.incoming_partial);

        for (int word_index = hot_word_begin; word_index < hot_word_end; word_index += sve_words) {
            const svbool_t pg = svwhilelt_b64(
                static_cast<uint64_t>(word_index),
                static_cast<uint64_t>(hot_word_end));
            const size_t flat_word_index = row_base + static_cast<size_t>(word_index);

            const svuint64x3_t outgoing_partials = svld3_u64(
                pg,
                reinterpret_cast<const uint64_t*>(outgoing_partial_words + word_index));
            const svuint64x3_t incoming_partials = svld3_u64(
                pg,
                reinterpret_cast<const uint64_t*>(incoming_partial_words + word_index));

            const svuint64x3_t acc_low = svld3_u64(
                pg,
                reinterpret_cast<const uint64_t*>(acc_low_ptr + word_index));
            const svuint64x2_t acc_high = svld2_u64(
                pg,
                reinterpret_cast<const uint64_t*>(acc_high_ptr + word_index));

            svuint64_t b0 = svget3_u64(acc_low, 0);
            svuint64_t b1 = svget3_u64(acc_low, 1);
            svuint64_t b2 = svget3_u64(acc_low, 2);
            svuint64_t b3 = svget2_u64(acc_high, 0);
            svuint64_t b4 = svget2_u64(acc_high, 1);

            svuint64_t next_adult_word;
            svuint64_t next_egg_word;
            store_hot_result_from_total_with_center_u64_sve(
                pg,
                b0, b1, b2, b3, b4,
                svld1_u64(pg, adults + flat_word_index),
                svld1_u64(pg, juveniles + flat_word_index),
                svld1_u64(pg, eggs + flat_word_index),
                next_adult_word,
                next_egg_word);

            svst1_u64(pg, next_adults_ptr + flat_word_index, next_adult_word);
            svst1_u64(pg, next_eggs_ptr + flat_word_index, next_egg_word);

            subtract_horizontal_partial_from_total_u64_sve(
                pg,
                svget3_u64(outgoing_partials, 0),
                svget3_u64(outgoing_partials, 1),
                svget3_u64(outgoing_partials, 2),
                b0, b1, b2, b3, b4);
            add_horizontal_partial_to_total_u64_sve(
                pg,
                svget3_u64(incoming_partials, 0),
                svget3_u64(incoming_partials, 1),
                svget3_u64(incoming_partials, 2),
                b0, b1, b2, b3, b4);

            svst3_u64(
                pg,
                reinterpret_cast<uint64_t*>(acc_low_ptr + word_index),
                svcreate3_u64(b0, b1, b2));
            svst2_u64(
                pg,
                reinterpret_cast<uint64_t*>(acc_high_ptr + word_index),
                svcreate2_u64(b3, b4));
        }

        std::swap(hot_row_scratch.row_partials[ring_head], hot_row_scratch.incoming_partial);
        if (++ring_head == 5) ring_head = 0;
    }

    const int final_row = hot_row_end - 1;
    const size_t final_row_base =
        static_cast<size_t>(final_row) * words_per_row;

    for (int word_index = hot_word_begin; word_index < hot_word_end; word_index += sve_words) {
        const svbool_t pg = svwhilelt_b64(
            static_cast<uint64_t>(word_index),
            static_cast<uint64_t>(hot_word_end));
        const size_t flat_word_index = final_row_base + static_cast<size_t>(word_index);

        const svuint64x3_t acc_low = svld3_u64(
            pg,
            reinterpret_cast<const uint64_t*>(acc_low_ptr + word_index));
        const svuint64x2_t acc_high = svld2_u64(
            pg,
            reinterpret_cast<const uint64_t*>(acc_high_ptr + word_index));

        svuint64_t next_adult_word;
        svuint64_t next_egg_word;
        store_hot_result_from_total_with_center_u64_sve(
            pg,
            svget3_u64(acc_low, 0),
            svget3_u64(acc_low, 1),
            svget3_u64(acc_low, 2),
            svget2_u64(acc_high, 0),
            svget2_u64(acc_high, 1),
            svld1_u64(pg, adults + flat_word_index),
            svld1_u64(pg, juveniles + flat_word_index),
            svld1_u64(pg, eggs + flat_word_index),
            next_adult_word,
            next_egg_word);

        svst1_u64(pg, next_adults_ptr + flat_word_index, next_adult_word);
        svst1_u64(pg, next_eggs_ptr + flat_word_index, next_egg_word);
    }
}

// ---------------------------------------------------------------------------
// Remaining generation helpers (unchanged)
// ---------------------------------------------------------------------------

static void initialize_next_generation_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_juveniles,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict juveniles = current_juveniles.data();
    uint64_t* __restrict next_adults_ptr = next_adults.data();
    uint64_t* __restrict next_eggs_ptr = next_eggs.data();
    const int hot_row_begin = 2;
    const int hot_row_end = grid_size - 2;
    const int last_word_offset = words_per_row - 1;

    for (int row_index = row_begin; row_index < row_end; ++row_index) {
        const size_t row_base =
            static_cast<size_t>(row_index) * words_per_row;

        if (row_index < hot_row_begin || row_index >= hot_row_end) {
            for (int word_index = 0; word_index < words_per_row; ++word_index) {
                const size_t flat_word_index = row_base + word_index;
                next_adults_ptr[flat_word_index] = juveniles[flat_word_index];
                next_eggs_ptr[flat_word_index] = 0ULL;
            }
        } else {
            const size_t left_word_index = row_base;
            const size_t right_word_index = row_base + last_word_offset;
            next_adults_ptr[left_word_index] = juveniles[left_word_index];
            next_eggs_ptr[left_word_index] = 0ULL;
            next_adults_ptr[right_word_index] = juveniles[right_word_index];
            next_eggs_ptr[right_word_index] = 0ULL;
        }
    }
}

__attribute__((noinline)) static void run_one_generation_left_edge_valid_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict adults = current_adults.data();
    const uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs = current_eggs.data();
    uint64_t* __restrict next_adults_ptr = next_adults.data();
    uint64_t* __restrict next_eggs_ptr = next_eggs.data();
    row_begin = std::max(row_begin, 2);
    row_end = std::min(row_end, grid_size - 2);
    const uint64_t valid_mask = ~0ULL << 2;

    for (int row_index = row_begin; row_index < row_end; ++row_index) {
        const size_t row_minus_2_base =
            static_cast<size_t>(row_index - 2) * words_per_row;
        const size_t row_minus_1_base =
            static_cast<size_t>(row_index - 1) * words_per_row;
        const size_t row_base =
            static_cast<size_t>(row_index) * words_per_row;
        const size_t row_plus_1_base =
            static_cast<size_t>(row_index + 1) * words_per_row;
        const size_t row_plus_2_base =
            static_cast<size_t>(row_index + 2) * words_per_row;

        const uint64_t top_left_2 =
            adults[row_minus_2_base] << 2;
        const uint64_t top_left_1 =
            adults[row_minus_2_base] << 1;
        const uint64_t top_center =
            adults[row_minus_2_base];
        const uint64_t top_right_1 =
            (adults[row_minus_2_base] >> 1) |
            (adults[row_minus_2_base + 1] << 63);
        const uint64_t top_right_2 =
            (adults[row_minus_2_base] >> 2) |
            (adults[row_minus_2_base + 1] << 62);

        const uint64_t upper_left_2 =
            adults[row_minus_1_base] << 2;
        const uint64_t upper_left_1 =
            adults[row_minus_1_base] << 1;
        const uint64_t upper_center =
            adults[row_minus_1_base];
        const uint64_t upper_right_1 =
            (adults[row_minus_1_base] >> 1) |
            (adults[row_minus_1_base + 1] << 63);
        const uint64_t upper_right_2 =
            (adults[row_minus_1_base] >> 2) |
            (adults[row_minus_1_base + 1] << 62);

        const uint64_t center_left_2 =
            adults[row_base] << 2;
        const uint64_t center_left_1 =
            adults[row_base] << 1;
        const uint64_t center_right_1 =
            (adults[row_base] >> 1) |
            (adults[row_base + 1] << 63);
        const uint64_t center_right_2 =
            (adults[row_base] >> 2) |
            (adults[row_base + 1] << 62);

        const uint64_t lower_left_2 =
            adults[row_plus_1_base] << 2;
        const uint64_t lower_left_1 =
            adults[row_plus_1_base] << 1;
        const uint64_t lower_center =
            adults[row_plus_1_base];
        const uint64_t lower_right_1 =
            (adults[row_plus_1_base] >> 1) |
            (adults[row_plus_1_base + 1] << 63);
        const uint64_t lower_right_2 =
            (adults[row_plus_1_base] >> 2) |
            (adults[row_plus_1_base + 1] << 62);

        const uint64_t bottom_left_2 =
            adults[row_plus_2_base] << 2;
        const uint64_t bottom_left_1 =
            adults[row_plus_2_base] << 1;
        const uint64_t bottom_center =
            adults[row_plus_2_base];
        const uint64_t bottom_right_1 =
            (adults[row_plus_2_base] >> 1) |
            (adults[row_plus_2_base + 1] << 63);
        const uint64_t bottom_right_2 =
            (adults[row_plus_2_base] >> 2) |
            (adults[row_plus_2_base + 1] << 62);

        uint64_t b0 = 0ULL;
        uint64_t b1 = 0ULL;
        uint64_t b2 = 0ULL;
        uint64_t b3 = 0ULL;
        uint64_t b4 = 0ULL;

        auto add_mask = [&](uint64_t m) {
            uint64_t carry = b0 & m;
            b0 ^= m;
            m = carry;

            carry = b1 & m;
            b1 ^= m;
            m = carry;

            carry = b2 & m;
            b2 ^= m;
            m = carry;

            carry = b3 & m;
            b3 ^= m;
            m = carry;

            b4 ^= m;
        };

        add_mask(top_left_2);
        add_mask(top_left_1);
        add_mask(top_center);
        add_mask(top_right_1);
        add_mask(top_right_2);

        add_mask(upper_left_2);
        add_mask(upper_left_1);
        add_mask(upper_center);
        add_mask(upper_right_1);
        add_mask(upper_right_2);

        add_mask(center_left_2);
        add_mask(center_left_1);
        add_mask(center_right_1);
        add_mask(center_right_2);

        add_mask(lower_left_2);
        add_mask(lower_left_1);
        add_mask(lower_center);
        add_mask(lower_right_1);
        add_mask(lower_right_2);

        add_mask(bottom_left_2);
        add_mask(bottom_left_1);
        add_mask(bottom_center);
        add_mask(bottom_right_1);
        add_mask(bottom_right_2);

        const uint64_t adult_word = adults[row_base];
        const uint64_t blocked_word =
            juveniles[row_base] | eggs[row_base];
        const uint64_t adult_valid = adult_word & valid_mask;
        const uint64_t empty_valid = ~(adult_word | blocked_word) & valid_mask;
        const uint64_t ge3 = b4 | b3 | b2 | (b1 & b0);
        const uint64_t ge4 = b4 | b3 | b2;
        const uint64_t ge6 = b4 | b3 | (b2 & b1);
        const uint64_t ge10 = b4 | (b3 & (b2 | b1));

        next_adults_ptr[row_base] |=
            adult_valid & ge4 & ~ge10;
        next_eggs_ptr[row_base] |=
            empty_valid & ge3 & ~ge6;
    }
}

__attribute__((noinline)) static void run_one_generation_right_edge_valid_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict adults = current_adults.data();
    const uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs = current_eggs.data();
    uint64_t* __restrict next_adults_ptr = next_adults.data();
    uint64_t* __restrict next_eggs_ptr = next_eggs.data();
    row_begin = std::max(row_begin, 2);
    row_end = std::min(row_end, grid_size - 2);
    const int last_word_offset = words_per_row - 1;
    const int previous_word_offset = words_per_row - 2;
    const uint64_t valid_mask = (1ULL << 62) - 1ULL;

    for (int row_index = row_begin; row_index < row_end; ++row_index) {
        const size_t row_minus_2_base =
            static_cast<size_t>(row_index - 2) * words_per_row;
        const size_t row_minus_1_base =
            static_cast<size_t>(row_index - 1) * words_per_row;
        const size_t row_base =
            static_cast<size_t>(row_index) * words_per_row;
        const size_t row_plus_1_base =
            static_cast<size_t>(row_index + 1) * words_per_row;
        const size_t row_plus_2_base =
            static_cast<size_t>(row_index + 2) * words_per_row;

        const uint64_t top_left_2 =
            (adults[row_minus_2_base + last_word_offset] << 2) |
            (adults[row_minus_2_base + previous_word_offset] >> 62);
        const uint64_t top_left_1 =
            (adults[row_minus_2_base + last_word_offset] << 1) |
            (adults[row_minus_2_base + previous_word_offset] >> 63);
        const uint64_t top_center =
            adults[row_minus_2_base + last_word_offset];
        const uint64_t top_right_1 =
            adults[row_minus_2_base + last_word_offset] >> 1;
        const uint64_t top_right_2 =
            adults[row_minus_2_base + last_word_offset] >> 2;

        const uint64_t upper_left_2 =
            (adults[row_minus_1_base + last_word_offset] << 2) |
            (adults[row_minus_1_base + previous_word_offset] >> 62);
        const uint64_t upper_left_1 =
            (adults[row_minus_1_base + last_word_offset] << 1) |
            (adults[row_minus_1_base + previous_word_offset] >> 63);
        const uint64_t upper_center =
            adults[row_minus_1_base + last_word_offset];
        const uint64_t upper_right_1 =
            adults[row_minus_1_base + last_word_offset] >> 1;
        const uint64_t upper_right_2 =
            adults[row_minus_1_base + last_word_offset] >> 2;

        const size_t edge_word_index = row_base + last_word_offset;
        const uint64_t center_left_2 =
            (adults[edge_word_index] << 2) |
            (adults[row_base + previous_word_offset] >> 62);
        const uint64_t center_left_1 =
            (adults[edge_word_index] << 1) |
            (adults[row_base + previous_word_offset] >> 63);
        const uint64_t center_right_1 =
            adults[edge_word_index] >> 1;
        const uint64_t center_right_2 =
            adults[edge_word_index] >> 2;

        const uint64_t lower_left_2 =
            (adults[row_plus_1_base + last_word_offset] << 2) |
            (adults[row_plus_1_base + previous_word_offset] >> 62);
        const uint64_t lower_left_1 =
            (adults[row_plus_1_base + last_word_offset] << 1) |
            (adults[row_plus_1_base + previous_word_offset] >> 63);
        const uint64_t lower_center =
            adults[row_plus_1_base + last_word_offset];
        const uint64_t lower_right_1 =
            adults[row_plus_1_base + last_word_offset] >> 1;
        const uint64_t lower_right_2 =
            adults[row_plus_1_base + last_word_offset] >> 2;

        const uint64_t bottom_left_2 =
            (adults[row_plus_2_base + last_word_offset] << 2) |
            (adults[row_plus_2_base + previous_word_offset] >> 62);
        const uint64_t bottom_left_1 =
            (adults[row_plus_2_base + last_word_offset] << 1) |
            (adults[row_plus_2_base + previous_word_offset] >> 63);
        const uint64_t bottom_center =
            adults[row_plus_2_base + last_word_offset];
        const uint64_t bottom_right_1 =
            adults[row_plus_2_base + last_word_offset] >> 1;
        const uint64_t bottom_right_2 =
            adults[row_plus_2_base + last_word_offset] >> 2;

        uint64_t b0 = 0ULL;
        uint64_t b1 = 0ULL;
        uint64_t b2 = 0ULL;
        uint64_t b3 = 0ULL;
        uint64_t b4 = 0ULL;

        auto add_mask = [&](uint64_t m) {
            uint64_t carry = b0 & m;
            b0 ^= m;
            m = carry;

            carry = b1 & m;
            b1 ^= m;
            m = carry;

            carry = b2 & m;
            b2 ^= m;
            m = carry;

            carry = b3 & m;
            b3 ^= m;
            m = carry;

            b4 ^= m;
        };

        add_mask(top_left_2);
        add_mask(top_left_1);
        add_mask(top_center);
        add_mask(top_right_1);
        add_mask(top_right_2);

        add_mask(upper_left_2);
        add_mask(upper_left_1);
        add_mask(upper_center);
        add_mask(upper_right_1);
        add_mask(upper_right_2);

        add_mask(center_left_2);
        add_mask(center_left_1);
        add_mask(center_right_1);
        add_mask(center_right_2);

        add_mask(lower_left_2);
        add_mask(lower_left_1);
        add_mask(lower_center);
        add_mask(lower_right_1);
        add_mask(lower_right_2);

        add_mask(bottom_left_2);
        add_mask(bottom_left_1);
        add_mask(bottom_center);
        add_mask(bottom_right_1);
        add_mask(bottom_right_2);

        const uint64_t adult_word = adults[edge_word_index];
        const uint64_t blocked_word =
            juveniles[edge_word_index] | eggs[edge_word_index];
        const uint64_t adult_valid = adult_word & valid_mask;
        const uint64_t empty_valid = ~(adult_word | blocked_word) & valid_mask;
        const uint64_t ge3 = b4 | b3 | b2 | (b1 & b0);
        const uint64_t ge4 = b4 | b3 | b2;
        const uint64_t ge6 = b4 | b3 | (b2 & b1);
        const uint64_t ge10 = b4 | (b3 & (b2 | b1));

        next_adults_ptr[edge_word_index] |=
            adult_valid & ge4 & ~ge10;
        next_eggs_ptr[edge_word_index] |=
            empty_valid & ge3 & ~ge6;
    }
}

__attribute__((noinline)) static void run_one_generation_wrapping_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict adults = current_adults.data();
    const uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs = current_eggs.data();
    uint64_t* __restrict next_adults_ptr = next_adults.data();
    uint64_t* __restrict next_eggs_ptr = next_eggs.data();
    const int top_wrap_row_end = 2;
    const int bottom_wrap_row_begin = grid_size - 2;

    for (int row_index = row_begin;
         row_index < row_end && row_index < top_wrap_row_end;
         ++row_index) {
        const size_t row_base =
            static_cast<size_t>(row_index) * words_per_row;

        for (int col_index = 0; col_index < grid_size; ++col_index) {
            const size_t flat_word_index = row_base + (col_index >> 6);
            const uint64_t cell_mask = 1ULL << (col_index & 63);
            if (((juveniles[flat_word_index] | eggs[flat_word_index]) &
                    cell_mask) !=
                0) {
                continue;
            }

            const int adult_neighbor_count = count_adult_neighbors_naive(
                current_adults,
                grid_size,
                words_per_row,
                row_index,
                col_index);

            if ((adults[flat_word_index] & cell_mask) != 0) {
                if (adult_neighbor_count >= 4 && adult_neighbor_count <= 9) {
                    next_adults_ptr[flat_word_index] |= cell_mask;
                }
            } else if (adult_neighbor_count >= 3 && adult_neighbor_count <= 5) {
                next_eggs_ptr[flat_word_index] |= cell_mask;
            }
        }
    }

    const int interior_begin = std::max(row_begin, 2);
    const int interior_end = std::min(row_end, grid_size - 2);
    for (int row_index = interior_begin; row_index < interior_end; ++row_index) {
        const size_t row_base =
            static_cast<size_t>(row_index) * words_per_row;
        const size_t left_edge_word_index = row_base;
        const size_t right_edge_word_index = row_base + (words_per_row - 1);

        for (int col_index = 0; col_index < 2; ++col_index) {
            const uint64_t cell_mask = 1ULL << col_index;
            if (((juveniles[left_edge_word_index] | eggs[left_edge_word_index]) &
                    cell_mask) !=
                0) {
                continue;
            }

            const int adult_neighbor_count = count_adult_neighbors_naive(
                current_adults,
                grid_size,
                words_per_row,
                row_index,
                col_index);

            if ((adults[left_edge_word_index] & cell_mask) != 0) {
                if (adult_neighbor_count >= 4 && adult_neighbor_count <= 9) {
                    next_adults_ptr[left_edge_word_index] |= cell_mask;
                }
            } else if (adult_neighbor_count >= 3 && adult_neighbor_count <= 5) {
                next_eggs_ptr[left_edge_word_index] |= cell_mask;
            }
        }

        for (int col_index = grid_size - 2; col_index < grid_size; ++col_index) {
            const uint64_t cell_mask = 1ULL << (col_index & 63);
            if (((juveniles[right_edge_word_index] |
                     eggs[right_edge_word_index]) &
                    cell_mask) !=
                0) {
                continue;
            }

            const int adult_neighbor_count = count_adult_neighbors_naive(
                current_adults,
                grid_size,
                words_per_row,
                row_index,
                col_index);

            if ((adults[right_edge_word_index] & cell_mask) != 0) {
                if (adult_neighbor_count >= 4 && adult_neighbor_count <= 9) {
                    next_adults_ptr[right_edge_word_index] |= cell_mask;
                }
            } else if (adult_neighbor_count >= 3 && adult_neighbor_count <= 5) {
                next_eggs_ptr[right_edge_word_index] |= cell_mask;
            }
        }
    }

    for (int row_index = std::max(row_begin, bottom_wrap_row_begin);
         row_index < row_end;
         ++row_index) {
        const size_t row_base =
            static_cast<size_t>(row_index) * words_per_row;

        for (int col_index = 0; col_index < grid_size; ++col_index) {
            const size_t flat_word_index = row_base + (col_index >> 6);
            const uint64_t cell_mask = 1ULL << (col_index & 63);
            if (((juveniles[flat_word_index] | eggs[flat_word_index]) &
                    cell_mask) !=
                0) {
                continue;
            }

            const int adult_neighbor_count = count_adult_neighbors_naive(
                current_adults,
                grid_size,
                words_per_row,
                row_index,
                col_index);

            if ((adults[flat_word_index] & cell_mask) != 0) {
                if (adult_neighbor_count >= 4 && adult_neighbor_count <= 9) {
                    next_adults_ptr[flat_word_index] |= cell_mask;
                }
            } else if (adult_neighbor_count >= 3 && adult_neighbor_count <= 5) {
                next_eggs_ptr[flat_word_index] |= cell_mask;
            }
        }
    }
}

__attribute__((noinline)) static void run_one_generation_edge_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs)
{
    run_one_generation_left_edge_valid_rows(
        row_begin,
        row_end,
        grid_size,
        words_per_row,
        current_adults,
        current_juveniles,
        current_eggs,
        next_adults,
        next_eggs);

    run_one_generation_right_edge_valid_rows(
        row_begin,
        row_end,
        grid_size,
        words_per_row,
        current_adults,
        current_juveniles,
        current_eggs,
        next_adults,
        next_eggs);

    run_one_generation_wrapping_rows(
        row_begin,
        row_end,
        grid_size,
        words_per_row,
        current_adults,
        current_juveniles,
        current_eggs,
        next_adults,
        next_eggs);
}

static void process_generation_chunk(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs,
    HotRowScratch& hot_row_scratch)
{
    initialize_next_generation_rows(
        row_begin,
        row_end,
        grid_size,
        words_per_row,
        current_juveniles,
        next_adults,
        next_eggs);

    run_one_generation_hot_rows(
        row_begin,
        row_end,
        grid_size,
        words_per_row,
        current_adults,
        current_juveniles,
        current_eggs,
        next_adults,
        next_eggs,
        hot_row_scratch);

    run_one_generation_edge_rows(
        row_begin,
        row_end,
        grid_size,
        words_per_row,
        current_adults,
        current_juveniles,
        current_eggs,
        next_adults,
        next_eggs);
}

static void unpack_bit_planes_to_output_grid(
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    std::vector<uint8_t>& output_grid)
{
    output_grid.assign(static_cast<size_t>(grid_size) * grid_size, EMPTY);

    for (int row_index = 0; row_index < grid_size; ++row_index) {
        for (int col_index = 0; col_index < grid_size; ++col_index) {
            uint8_t cell_state = EMPTY;

            if (is_cell_set(current_adults, words_per_row, row_index, col_index)) {
                cell_state = ADULT;
            } else if (is_cell_set(current_juveniles, words_per_row, row_index, col_index)) {
                cell_state = JUVENILE;
            } else if (is_cell_set(current_eggs, words_per_row, row_index, col_index)) {
                cell_state = EGG;
            }

            output_grid[static_cast<size_t>(row_index) * grid_size + col_index] = cell_state;
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "Usage: %s <input.bin> <output.bin> [generations]\n", argv[0]);
        return 1;
    }

    int generations = 10000;
    if (argc == 4) {
        char* end = nullptr;
        long parsed_generations = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || parsed_generations <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        generations = static_cast<int>(parsed_generations);
    }

    FILE* input_file = std::fopen(argv[1], "rb");
    if (!input_file) {
        std::fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]);
        return 2;
    }

    uint64_t width = 0;
    uint64_t height = 0;
    if (std::fread(&width, sizeof(uint64_t), 1, input_file) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, input_file) != 1) {
        std::fprintf(stderr, "Error: input file too short (cannot read header)\n");
        std::fclose(input_file);
        return 3;
    }

    if (width == 0 || width != height) {
        std::fprintf(
            stderr,
            "Error: grid must be square and non-empty, got %" PRIu64 " x %" PRIu64 "\n",
            width,
            height);
        std::fclose(input_file);
        return 3;
    }

    int grid_size = static_cast<int>(width);
    size_t cell_count = width * width;

    std::vector<uint8_t> input_grid(cell_count);
    if (std::fread(input_grid.data(), 1, cell_count, input_file) != cell_count) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(input_file);
        return 4;
    }
    std::fclose(input_file);

    int words_per_row = grid_size >> 6;
    size_t total_words = static_cast<size_t>(grid_size) * words_per_row;

    alignas(64) BitPlane current_adults(total_words, 0ULL);
    alignas(64) BitPlane current_juveniles(total_words, 0ULL);
    alignas(64) BitPlane current_eggs(total_words, 0ULL);
    alignas(64) BitPlane next_adults(total_words, 0ULL);
    alignas(64) BitPlane next_eggs(total_words, 0ULL);

    // Input packing is preprocessing and is intentionally not timed.
    pack_input_into_bit_planes(
        input_grid,
        grid_size,
        words_per_row,
        current_adults,
        current_juveniles,
        current_eggs);

    auto start_time = std::chrono::steady_clock::now();

    const int thread_count = std::max(1, std::min(TARGET_THREAD_COUNT, grid_size));
    std::vector<RowRange> row_ranges(static_cast<size_t>(thread_count));
    for (int worker_id = 0; worker_id < thread_count; ++worker_id) {
        row_ranges[static_cast<size_t>(worker_id)] =
            compute_row_range(grid_size, thread_count, worker_id);
    }

    std::barrier generation_start_barrier(thread_count);
    std::barrier generation_done_barrier(thread_count);
    std::atomic<bool> stop_workers{false};

    auto worker_body = [&](int worker_id) {
        pin_current_thread_to_cpu(worker_id);
        const RowRange owned_rows = row_ranges[static_cast<size_t>(worker_id)];
        HotRowScratch hot_row_scratch(words_per_row);

        for (;;) {
            generation_start_barrier.arrive_and_wait();
            if (stop_workers.load(std::memory_order_acquire)) {
                break;
            }

            process_generation_chunk(
                owned_rows.begin,
                owned_rows.end,
                grid_size,
                words_per_row,
                current_adults,
                current_juveniles,
                current_eggs,
                next_adults,
                next_eggs,
                hot_row_scratch);

            generation_done_barrier.arrive_and_wait();
        }
    };

    HotRowScratch main_hot_row_scratch(words_per_row);
    pin_current_thread_to_cpu(0);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(std::max(0, thread_count - 1)));
    for (int worker_id = 1; worker_id < thread_count; ++worker_id) {
        workers.emplace_back(worker_body, worker_id);
    }

    const RowRange main_owned_rows = row_ranges.front();
    for (int generation = 0; generation < generations; ++generation) {
        generation_start_barrier.arrive_and_wait();

        process_generation_chunk(
            main_owned_rows.begin,
            main_owned_rows.end,
            grid_size,
            words_per_row,
            current_adults,
            current_juveniles,
            current_eggs,
            next_adults,
            next_eggs,
            main_hot_row_scratch);

        generation_done_barrier.arrive_and_wait();

        current_adults.swap(next_adults);
        current_juveniles.swap(current_eggs);
        current_eggs.swap(next_eggs);
    }

    stop_workers.store(true, std::memory_order_release);
    generation_start_barrier.arrive_and_wait();

    for (std::thread& worker : workers) {
        worker.join();
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::printf("%.3f ms\n", elapsed_ms);

    std::vector<uint8_t> output_grid;
    unpack_bit_planes_to_output_grid(
        grid_size,
        words_per_row,
        current_adults,
        current_juveniles,
        current_eggs,
        output_grid);

    FILE* output_file = std::fopen(argv[2], "wb");
    if (!output_file) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }

    if (std::fwrite(&width, sizeof(uint64_t), 1, output_file) != 1 ||
        std::fwrite(&height, sizeof(uint64_t), 1, output_file) != 1 ||
        std::fwrite(output_grid.data(), 1, cell_count, output_file) != cell_count) {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(output_file);
        return 6;
    }

    std::fclose(output_file);
    return 0;
}