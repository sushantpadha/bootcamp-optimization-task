// spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos_branchless_subtract_sve2.cpp —
// sliding-window carry-save-adder hot-loop prototype with persistent
// row-chunk worker threads, rolling per-row horizontal partial reuse,
// a cache-friendly row-by-row rolling vertical accumulator, AoS-packed
// horizontal partial storage, branchless subtract updates, an explicit
// SVE2 fast path for both horizontal partial generation and the main
// rolling-update hot loop with fused incoming-partial production, an
// exact two-row blocked hot update, a flat contiguous slab allocator
// for all per-thread scratch memory (eliminating multi-level vector
// pointer chasing), and a generation loop with the separate
// next-generation init pass eliminated by folding juvenile promotion
// directly into the edge-word assignment paths for the Monster
// Spawning Grid reference simulator.
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
#include <cstring>
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

struct HorizontalPartialWord {
    uint64_t b0 = 0ULL;
    uint64_t b1 = 0ULL;
    uint64_t b2 = 0ULL;
};

struct alignas(16) HorizontalPartialPair2 {
    uint64_t b0[2] = {0ULL, 0ULL};
    uint64_t b1[2] = {0ULL, 0ULL};
    uint64_t b2[2] = {0ULL, 0ULL};
};

static_assert(sizeof(HorizontalPartialPair2) == sizeof(uint64_t) * 6);
static_assert(alignof(HorizontalPartialPair2) == 16);

// HorizontalPartialRow is now a non-owning view into the flat HotRowScratch slab.
// The pairs pointer is set by HotRowScratch during construction; this struct
// performs no allocation of its own.
struct HorizontalPartialRow {
    HorizontalPartialPair2* pairs = nullptr;

    HorizontalPartialRow() = default;
};

inline size_t horizontal_partial_pair_index(int word_index)
{
    return static_cast<size_t>(word_index >> 1);
}

inline int horizontal_partial_pair_lane(int word_index)
{
    return word_index & 1;
}

inline HorizontalPartialWord load_horizontal_partial_scalar(
    const HorizontalPartialRow& row_partial,
    int word_index)
{
    const HorizontalPartialPair2& pair =
        row_partial.pairs[horizontal_partial_pair_index(word_index)];
    const int lane = horizontal_partial_pair_lane(word_index);
    return HorizontalPartialWord{
        pair.b0[lane],
        pair.b1[lane],
        pair.b2[lane],
    };
}

inline void store_horizontal_partial_scalar(
    HorizontalPartialRow& row_partial,
    int word_index,
    uint64_t partial_b0,
    uint64_t partial_b1,
    uint64_t partial_b2)
{
    HorizontalPartialPair2& pair =
        row_partial.pairs[horizontal_partial_pair_index(word_index)];
    const int lane = horizontal_partial_pair_lane(word_index);
    pair.b0[lane] = partial_b0;
    pair.b1[lane] = partial_b1;
    pair.b2[lane] = partial_b2;
}

inline void store_horizontal_partial_pair2_u64_sve(
    svbool_t pg,
    HorizontalPartialPair2& pair,
    svuint64_t partial_b0,
    svuint64_t partial_b1,
    svuint64_t partial_b2)
{
    svst1_u64(pg, pair.b0, partial_b0);
    svst1_u64(pg, pair.b1, partial_b1);
    svst1_u64(pg, pair.b2, partial_b2);
}

inline void load_horizontal_partial_pair2_u64_sve(
    svbool_t pg,
    const HorizontalPartialPair2& pair,
    svuint64_t& partial_b0,
    svuint64_t& partial_b1,
    svuint64_t& partial_b2)
{
    partial_b0 = svld1_u64(pg, pair.b0);
    partial_b1 = svld1_u64(pg, pair.b1);
    partial_b2 = svld1_u64(pg, pair.b2);
}

inline void carry_save_add(
    uint64_t a,
    uint64_t b,
    uint64_t c,
    uint64_t& sum,
    uint64_t& carry);

inline void carry_save_add_u64_sve(
    svbool_t pg,
    svuint64_t a,
    svuint64_t b,
    svuint64_t c,
    svuint64_t& sum,
    svuint64_t& carry);

__attribute__((always_inline)) static inline HorizontalPartialWord
compute_horizontal_partial_scalar_from_adult_row(
    const uint64_t* __restrict adult_row,
    int word_index)
{
    const uint64_t prev = adult_row[word_index - 1];
    const uint64_t curr = adult_row[word_index];
    const uint64_t next = adult_row[word_index + 1];
    const uint64_t left_2 = (curr << 2) | (prev >> 62);
    const uint64_t left_1 = (curr << 1) | (prev >> 63);
    const uint64_t right_1 = (curr >> 1) | (next << 63);
    const uint64_t right_2 = (curr >> 2) | (next << 62);

    uint64_t sum_1 = 0ULL;
    uint64_t carry_1 = 0ULL;
    uint64_t sum_2 = 0ULL;
    uint64_t carry_2 = 0ULL;

    carry_save_add(left_2, left_1, curr, sum_1, carry_1);
    carry_save_add(right_1, right_2, sum_1, sum_2, carry_2);
    return HorizontalPartialWord{
        sum_2,
        carry_1 ^ carry_2,
        carry_1 & carry_2,
    };
}

__attribute__((always_inline)) static inline void
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

    partial_b0 = sum_2;
    partial_b1 = sveor_x(pg, carry_1, carry_2);
    partial_b2 = svand_x(pg, carry_1, carry_2);
}

// VerticalAccumulatorRow is now a non-owning view into the flat HotRowScratch slab.
// Five raw uint64_t pointers replace the five separate BitPlane vectors, eliminating
// the per-plane vector header overhead and the pointer-chase on every hot-loop access.
struct VerticalAccumulatorRow {
    uint64_t* b0 = nullptr;
    uint64_t* b1 = nullptr;
    uint64_t* b2 = nullptr;
    uint64_t* b3 = nullptr;
    uint64_t* b4 = nullptr;
};

// HotRowScratch owns a single flat 64-byte-aligned slab that holds all per-thread
// scratch memory in one contiguous allocation:
//
//   [ ring_partial_0 | ring_partial_1 | ... | ring_partial_4 |
//     acc_b0 | acc_b1 | acc_b2 | acc_b3 | acc_b4 ]
//
// Each section is padded to a 64-byte boundary so that every sub-array starts
// on its own cache line.  HorizontalPartialRow and VerticalAccumulatorRow hold
// non-owning pointers into this slab; they perform no allocation themselves.
// The 'incoming_partial' slot from the previous design has been removed: v2 of
// the hot loop computes incoming partials on-the-fly and writes them directly
// back into the recycled ring slot, so a dedicated incoming buffer is never needed.
struct HotRowScratch {
    void* slab = nullptr;
    std::array<HorizontalPartialRow, 5> row_partials;
    VerticalAccumulatorRow rolling_total;

    HotRowScratch() = default;

    explicit HotRowScratch(int words_per_row)
    {
        const int pairs_per_row = (words_per_row + 1) >> 1;
        constexpr size_t kAlign = 64;

        // Round each section up to a multiple of kAlign so the next section
        // starts on a cache-line boundary.
        const size_t pair_section_bytes =
            (static_cast<size_t>(pairs_per_row) * sizeof(HorizontalPartialPair2) +
             kAlign - 1) &
            ~(kAlign - 1);
        const size_t acc_section_bytes =
            (static_cast<size_t>(words_per_row) * sizeof(uint64_t) + kAlign - 1) &
            ~(kAlign - 1);

        // 5 ring partial rows + 5 accumulator planes.
        const size_t total_bytes =
            5 * pair_section_bytes + 5 * acc_section_bytes;

        if (posix_memalign(&slab, kAlign, total_bytes) != 0) {
            throw std::bad_alloc();
        }
        std::memset(slab, 0, total_bytes);

        uint8_t* ptr = static_cast<uint8_t*>(slab);

        for (int i = 0; i < 5; ++i) {
            row_partials[i].pairs =
                reinterpret_cast<HorizontalPartialPair2*>(ptr);
            ptr += pair_section_bytes;
        }

        rolling_total.b0 = reinterpret_cast<uint64_t*>(ptr); ptr += acc_section_bytes;
        rolling_total.b1 = reinterpret_cast<uint64_t*>(ptr); ptr += acc_section_bytes;
        rolling_total.b2 = reinterpret_cast<uint64_t*>(ptr); ptr += acc_section_bytes;
        rolling_total.b3 = reinterpret_cast<uint64_t*>(ptr); ptr += acc_section_bytes;
        rolling_total.b4 = reinterpret_cast<uint64_t*>(ptr);
    }

    ~HotRowScratch()
    {
        std::free(slab);
    }

    HotRowScratch(const HotRowScratch&) = delete;
    HotRowScratch& operator=(const HotRowScratch&) = delete;

    HotRowScratch(HotRowScratch&& other) noexcept
        : slab(other.slab),
          row_partials(other.row_partials),
          rolling_total(other.rolling_total)
    {
        other.slab = nullptr;
        other.row_partials = {};
        other.rolling_total = {};
    }

    HotRowScratch& operator=(HotRowScratch&& other) noexcept
    {
        if (this != &other) {
            std::free(slab);
            slab = other.slab;
            row_partials = other.row_partials;
            rolling_total = other.rolling_total;
            other.slab = nullptr;
            other.row_partials = {};
            other.rolling_total = {};
        }
        return *this;
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

inline void carry_save_add(
    uint64_t a,
    uint64_t b,
    uint64_t c,
    uint64_t& sum,
    uint64_t& carry)
{
    sum = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

inline void carry_save_add_u64_sve(
    svbool_t pg,
    svuint64_t a,
    svuint64_t b,
    svuint64_t c,
    svuint64_t& sum,
    svuint64_t& carry)
{
    sum = sveor_x(pg, sveor_x(pg, a, b), c);
    carry = svbsl_u64(svorr_x(pg, a, b), svand_x(pg, a, b), c);
}

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

    const svuint64_t not_total_b1 = sveor_x(pg, total_b1, all_ones);
    svuint64_t next_borrow = svbsl_u64(
        svorr_x(pg, not_total_b1, partial_b1),
        svand_x(pg, not_total_b1, partial_b1),
        borrow);
    total_b1 = sveor_x(pg, sveor_x(pg, total_b1, partial_b1), borrow);
    borrow = next_borrow;

    const svuint64_t not_total_b2 = sveor_x(pg, total_b2, all_ones);
    next_borrow = svbsl_u64(
        svorr_x(pg, not_total_b2, partial_b2),
        svand_x(pg, not_total_b2, partial_b2),
        borrow);
    total_b2 = sveor_x(pg, sveor_x(pg, total_b2, partial_b2), borrow);
    borrow = next_borrow;

    next_borrow = svand_x(pg, sveor_x(pg, total_b3, all_ones), borrow);
    total_b3 = sveor_x(pg, total_b3, borrow);
    total_b4 = sveor_x(pg, total_b4, next_borrow);
}

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

inline void add_horizontal_partial_to_total(
    uint64_t partial_b0,
    uint64_t partial_b1,
    uint64_t partial_b2,
    uint64_t& total_b0,
    uint64_t& total_b1,
    uint64_t& total_b2,
    uint64_t& total_b3,
    uint64_t& total_b4)
{
    uint64_t carry = total_b0 & partial_b0;
    total_b0 ^= partial_b0;

    carry_save_add(total_b1, partial_b1, carry, total_b1, carry);
    carry_save_add(total_b2, partial_b2, carry, total_b2, carry);

    const uint64_t carry4 = total_b3 & carry;
    total_b3 ^= carry;
    total_b4 ^= carry4;
}

inline void subtract_horizontal_partial_from_total(
    uint64_t partial_b0,
    uint64_t partial_b1,
    uint64_t partial_b2,
    uint64_t& total_b0,
    uint64_t& total_b1,
    uint64_t& total_b2,
    uint64_t& total_b3,
    uint64_t& total_b4)
{
    uint64_t borrow = (~total_b0) & partial_b0;
    total_b0 ^= partial_b0;

    uint64_t next_borrow =
        ((~total_b1) & (partial_b1 | borrow)) | (partial_b1 & borrow);
    total_b1 ^= partial_b1 ^ borrow;
    borrow = next_borrow;

    next_borrow =
        ((~total_b2) & (partial_b2 | borrow)) | (partial_b2 & borrow);
    total_b2 ^= partial_b2 ^ borrow;
    borrow = next_borrow;

    next_borrow = (~total_b3) & borrow;
    total_b3 ^= borrow;
    total_b4 ^= next_borrow;
}

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

__attribute__((always_inline)) static inline void compute_horizontal_partial_row(
    int source_row,
    int words_per_row,
    const BitPlane& current_adults,
    HorizontalPartialRow& row_partial)
{
    const uint64_t* __restrict adults = current_adults.data();
    // row_partial.pairs is a raw pointer into the flat slab; no .data() needed.
    HorizontalPartialPair2* __restrict partial_pairs = row_partial.pairs;
    const int hot_word_begin = 1;
    const int hot_word_end = words_per_row - 1;

    if (hot_word_begin >= hot_word_end) {
        return;
    }

    const uint64_t* __restrict adult_row =
        adults + static_cast<size_t>(source_row) * words_per_row;

    int word_index = hot_word_begin;
    const int sve_words = static_cast<int>(svcntd());
    const svbool_t pg = svptrue_b64();
    if (sve_words == 2 && hot_word_begin + 1 < hot_word_end) {
        const uint64_t prev = adult_row[word_index - 1];
        const uint64_t curr = adult_row[word_index];
        const uint64_t next = adult_row[word_index + 1];
        const uint64_t left_2 = (curr << 2) | (prev >> 62);
        const uint64_t left_1 = (curr << 1) | (prev >> 63);
        const uint64_t right_1 = (curr >> 1) | (next << 63);
        const uint64_t right_2 = (curr >> 2) | (next << 62);

        uint64_t sum_1 = 0ULL;
        uint64_t carry_1 = 0ULL;
        uint64_t sum_2 = 0ULL;
        uint64_t carry_2 = 0ULL;

        carry_save_add(left_2, left_1, curr, sum_1, carry_1);
        carry_save_add(right_1, right_2, sum_1, sum_2, carry_2);
        store_horizontal_partial_scalar(
            row_partial,
            word_index,
            sum_2,
            carry_1 ^ carry_2,
            carry_1 & carry_2);
        ++word_index;

        const int aligned_simd_word_end =
            word_index + ((hot_word_end - word_index) / sve_words) * sve_words;
        for (; word_index < aligned_simd_word_end; word_index += sve_words) {
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

            HorizontalPartialPair2& partial_pair =
                partial_pairs[horizontal_partial_pair_index(word_index)];
            store_horizontal_partial_pair2_u64_sve(
                pg,
                partial_pair,
                sum_2,
                sveor_x(pg, carry_1, carry_2),
                svand_x(pg, carry_1, carry_2));
        }
    }

    for (; word_index < hot_word_end; ++word_index) {
        const uint64_t prev = adult_row[word_index - 1];
        const uint64_t curr = adult_row[word_index];
        const uint64_t next = adult_row[word_index + 1];
        const uint64_t left_2 = (curr << 2) | (prev >> 62);
        const uint64_t left_1 = (curr << 1) | (prev >> 63);
        const uint64_t center = curr;
        const uint64_t right_1 = (curr >> 1) | (next << 63);
        const uint64_t right_2 = (curr >> 2) | (next << 62);

        uint64_t sum_1 = 0ULL;
        uint64_t carry_1 = 0ULL;
        uint64_t sum_2 = 0ULL;
        uint64_t carry_2 = 0ULL;

        carry_save_add(left_2, left_1, center, sum_1, carry_1);
        carry_save_add(right_1, right_2, sum_1, sum_2, carry_2);
        store_horizontal_partial_scalar(
            row_partial,
            word_index,
            sum_2,
            carry_1 ^ carry_2,
            carry_1 & carry_2);
    }
}

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

    for (int word_index = hot_word_begin; word_index < hot_word_end; ++word_index) {
        uint64_t b0 = 0ULL;
        uint64_t b1 = 0ULL;
        uint64_t b2 = 0ULL;
        uint64_t b3 = 0ULL;
        uint64_t b4 = 0ULL;
        for (const HorizontalPartialRow& row_partial : row_partials) {
            const HorizontalPartialWord partial_word =
                load_horizontal_partial_scalar(row_partial, word_index);
            add_horizontal_partial_to_total(
                partial_word.b0,
                partial_word.b1,
                partial_word.b2,
                b0,
                b1,
                b2,
                b3,
                b4);
        }

        rolling_total.b0[word_index] = b0;
        rolling_total.b1[word_index] = b1;
        rolling_total.b2[word_index] = b2;
        rolling_total.b3[word_index] = b3;
        rolling_total.b4[word_index] = b4;
    }
}

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

    // rolling_total.b0..b4 are now raw pointers into the flat slab; no .data() call needed.
    uint64_t* __restrict total_b0_ptr = hot_row_scratch.rolling_total.b0;
    uint64_t* __restrict total_b1_ptr = hot_row_scratch.rolling_total.b1;
    uint64_t* __restrict total_b2_ptr = hot_row_scratch.rolling_total.b2;
    uint64_t* __restrict total_b3_ptr = hot_row_scratch.rolling_total.b3;
    uint64_t* __restrict total_b4_ptr = hot_row_scratch.rolling_total.b4;

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
    const int simd_word_end =
        hot_word_begin + ((hot_word_end - hot_word_begin) / sve_words) * sve_words;
    const bool use_aligned_main_hot_loop =
        (sve_words == 2) && (hot_word_begin + 1 < hot_word_end);
    const int aligned_hot_word_end = hot_word_end - 1;
    const svbool_t pg = svptrue_b64();

    size_t ring_head = 0;
    int row_index = hot_row_begin;
    for (; row_index + 2 < hot_row_end; row_index += 2) {
        const size_t row0_base =
            static_cast<size_t>(row_index) * words_per_row;
        const size_t row1_base =
            static_cast<size_t>(row_index + 1) * words_per_row;
        HorizontalPartialRow& recycled_row0 =
            hot_row_scratch.row_partials[ring_head];
        HorizontalPartialRow& recycled_row1 =
            hot_row_scratch.row_partials[(ring_head + 1) % hot_row_scratch.row_partials.size()];
        // row_partials[i].pairs is a raw pointer into the slab; no .data() call needed.
        HorizontalPartialPair2* __restrict recycled_partial_pairs0 =
            recycled_row0.pairs;
        HorizontalPartialPair2* __restrict recycled_partial_pairs1 =
            recycled_row1.pairs;
        const HorizontalPartialPair2* __restrict outgoing_partial_pairs0 =
            recycled_partial_pairs0;
        const HorizontalPartialPair2* __restrict outgoing_partial_pairs1 =
            recycled_partial_pairs1;
        const uint64_t* __restrict incoming_adult_row0 =
            adults + static_cast<size_t>(row_index + 3) * words_per_row;
        const uint64_t* __restrict incoming_adult_row1 =
            adults + static_cast<size_t>(row_index + 4) * words_per_row;

        int word_index = hot_word_begin;
        if (use_aligned_main_hot_loop) {
            const size_t flat_word_index0 = row0_base + static_cast<size_t>(word_index);
            const size_t flat_word_index1 = row1_base + static_cast<size_t>(word_index);
            const HorizontalPartialWord outgoing_partial0 =
                load_horizontal_partial_scalar(recycled_row0, word_index);
            const HorizontalPartialWord incoming_partial0 =
                compute_horizontal_partial_scalar_from_adult_row(
                    incoming_adult_row0,
                    word_index);
            const HorizontalPartialWord outgoing_partial1 =
                load_horizontal_partial_scalar(recycled_row1, word_index);
            const HorizontalPartialWord incoming_partial1 =
                compute_horizontal_partial_scalar_from_adult_row(
                    incoming_adult_row1,
                    word_index);
            uint64_t b0 = total_b0_ptr[word_index];
            uint64_t b1 = total_b1_ptr[word_index];
            uint64_t b2 = total_b2_ptr[word_index];
            uint64_t b3 = total_b3_ptr[word_index];
            uint64_t b4 = total_b4_ptr[word_index];

            store_hot_result_from_total_with_center(
                b0,
                b1,
                b2,
                b3,
                b4,
                adults[flat_word_index0],
                juveniles[flat_word_index0],
                eggs[flat_word_index0],
                next_adults_ptr[flat_word_index0],
                next_eggs_ptr[flat_word_index0]);

            subtract_horizontal_partial_from_total(
                outgoing_partial0.b0,
                outgoing_partial0.b1,
                outgoing_partial0.b2,
                b0,
                b1,
                b2,
                b3,
                b4);
            add_horizontal_partial_to_total(
                incoming_partial0.b0,
                incoming_partial0.b1,
                incoming_partial0.b2,
                b0,
                b1,
                b2,
                b3,
                b4);

            store_hot_result_from_total_with_center(
                b0,
                b1,
                b2,
                b3,
                b4,
                adults[flat_word_index1],
                juveniles[flat_word_index1],
                eggs[flat_word_index1],
                next_adults_ptr[flat_word_index1],
                next_eggs_ptr[flat_word_index1]);

            subtract_horizontal_partial_from_total(
                outgoing_partial1.b0,
                outgoing_partial1.b1,
                outgoing_partial1.b2,
                b0,
                b1,
                b2,
                b3,
                b4);
            add_horizontal_partial_to_total(
                incoming_partial1.b0,
                incoming_partial1.b1,
                incoming_partial1.b2,
                b0,
                b1,
                b2,
                b3,
                b4);

            total_b0_ptr[word_index] = b0;
            total_b1_ptr[word_index] = b1;
            total_b2_ptr[word_index] = b2;
            total_b3_ptr[word_index] = b3;
            total_b4_ptr[word_index] = b4;
            store_horizontal_partial_scalar(
                recycled_row0,
                word_index,
                incoming_partial0.b0,
                incoming_partial0.b1,
                incoming_partial0.b2);
            store_horizontal_partial_scalar(
                recycled_row1,
                word_index,
                incoming_partial1.b0,
                incoming_partial1.b1,
                incoming_partial1.b2);
            ++word_index;

            const int aligned_simd_word_end =
                word_index + ((aligned_hot_word_end - word_index) / sve_words) * sve_words;
            for (; word_index < aligned_simd_word_end; word_index += sve_words) {
                const size_t aligned_flat_word_index0 =
                    row0_base + static_cast<size_t>(word_index);
                const size_t aligned_flat_word_index1 =
                    row1_base + static_cast<size_t>(word_index);
                const size_t partial_pair_index =
                    horizontal_partial_pair_index(word_index);
                svuint64_t outgoing_partial0_b0;
                svuint64_t outgoing_partial0_b1;
                svuint64_t outgoing_partial0_b2;
                svuint64_t incoming_partial0_b0;
                svuint64_t incoming_partial0_b1;
                svuint64_t incoming_partial0_b2;
                svuint64_t outgoing_partial1_b0;
                svuint64_t outgoing_partial1_b1;
                svuint64_t outgoing_partial1_b2;
                svuint64_t incoming_partial1_b0;
                svuint64_t incoming_partial1_b1;
                svuint64_t incoming_partial1_b2;
                load_horizontal_partial_pair2_u64_sve(
                    pg,
                    outgoing_partial_pairs0[partial_pair_index],
                    outgoing_partial0_b0,
                    outgoing_partial0_b1,
                    outgoing_partial0_b2);
                compute_horizontal_partial_pair2_u64_sve_from_adult_row(
                    pg,
                    incoming_adult_row0,
                    word_index,
                    incoming_partial0_b0,
                    incoming_partial0_b1,
                    incoming_partial0_b2);
                load_horizontal_partial_pair2_u64_sve(
                    pg,
                    outgoing_partial_pairs1[partial_pair_index],
                    outgoing_partial1_b0,
                    outgoing_partial1_b1,
                    outgoing_partial1_b2);
                compute_horizontal_partial_pair2_u64_sve_from_adult_row(
                    pg,
                    incoming_adult_row1,
                    word_index,
                    incoming_partial1_b0,
                    incoming_partial1_b1,
                    incoming_partial1_b2);

                svuint64_t vb0 = svld1_u64(pg, total_b0_ptr + word_index);
                svuint64_t vb1 = svld1_u64(pg, total_b1_ptr + word_index);
                svuint64_t vb2 = svld1_u64(pg, total_b2_ptr + word_index);
                svuint64_t vb3 = svld1_u64(pg, total_b3_ptr + word_index);
                svuint64_t vb4 = svld1_u64(pg, total_b4_ptr + word_index);

                svuint64_t next_adult_word;
                svuint64_t next_egg_word;
                store_hot_result_from_total_with_center_u64_sve(
                    pg,
                    vb0,
                    vb1,
                    vb2,
                    vb3,
                    vb4,
                    svld1_u64(pg, adults + aligned_flat_word_index0),
                    svld1_u64(pg, juveniles + aligned_flat_word_index0),
                    svld1_u64(pg, eggs + aligned_flat_word_index0),
                    next_adult_word,
                    next_egg_word);
                svst1_u64(pg, next_adults_ptr + aligned_flat_word_index0, next_adult_word);
                svst1_u64(pg, next_eggs_ptr + aligned_flat_word_index0, next_egg_word);

                subtract_horizontal_partial_from_total_u64_sve(
                    pg,
                    outgoing_partial0_b0,
                    outgoing_partial0_b1,
                    outgoing_partial0_b2,
                    vb0,
                    vb1,
                    vb2,
                    vb3,
                    vb4);
                add_horizontal_partial_to_total_u64_sve(
                    pg,
                    incoming_partial0_b0,
                    incoming_partial0_b1,
                    incoming_partial0_b2,
                    vb0,
                    vb1,
                    vb2,
                    vb3,
                    vb4);

                store_hot_result_from_total_with_center_u64_sve(
                    pg,
                    vb0,
                    vb1,
                    vb2,
                    vb3,
                    vb4,
                    svld1_u64(pg, adults + aligned_flat_word_index1),
                    svld1_u64(pg, juveniles + aligned_flat_word_index1),
                    svld1_u64(pg, eggs + aligned_flat_word_index1),
                    next_adult_word,
                    next_egg_word);
                svst1_u64(pg, next_adults_ptr + aligned_flat_word_index1, next_adult_word);
                svst1_u64(pg, next_eggs_ptr + aligned_flat_word_index1, next_egg_word);

                subtract_horizontal_partial_from_total_u64_sve(
                    pg,
                    outgoing_partial1_b0,
                    outgoing_partial1_b1,
                    outgoing_partial1_b2,
                    vb0,
                    vb1,
                    vb2,
                    vb3,
                    vb4);
                add_horizontal_partial_to_total_u64_sve(
                    pg,
                    incoming_partial1_b0,
                    incoming_partial1_b1,
                    incoming_partial1_b2,
                    vb0,
                    vb1,
                    vb2,
                    vb3,
                    vb4);

                svst1_u64(pg, total_b0_ptr + word_index, vb0);
                svst1_u64(pg, total_b1_ptr + word_index, vb1);
                svst1_u64(pg, total_b2_ptr + word_index, vb2);
                svst1_u64(pg, total_b3_ptr + word_index, vb3);
                svst1_u64(pg, total_b4_ptr + word_index, vb4);
                store_horizontal_partial_pair2_u64_sve(
                    pg,
                    recycled_partial_pairs0[partial_pair_index],
                    incoming_partial0_b0,
                    incoming_partial0_b1,
                    incoming_partial0_b2);
                store_horizontal_partial_pair2_u64_sve(
                    pg,
                    recycled_partial_pairs1[partial_pair_index],
                    incoming_partial1_b0,
                    incoming_partial1_b1,
                    incoming_partial1_b2);
            }
        }

        for (; word_index < hot_word_end; ++word_index) {
            const size_t flat_word_index0 = row0_base + word_index;
            const size_t flat_word_index1 = row1_base + word_index;
            const HorizontalPartialWord outgoing_partial0 =
                load_horizontal_partial_scalar(recycled_row0, word_index);
            const HorizontalPartialWord incoming_partial0 =
                compute_horizontal_partial_scalar_from_adult_row(
                    incoming_adult_row0,
                    word_index);
            const HorizontalPartialWord outgoing_partial1 =
                load_horizontal_partial_scalar(recycled_row1, word_index);
            const HorizontalPartialWord incoming_partial1 =
                compute_horizontal_partial_scalar_from_adult_row(
                    incoming_adult_row1,
                    word_index);
            uint64_t b0 = total_b0_ptr[word_index];
            uint64_t b1 = total_b1_ptr[word_index];
            uint64_t b2 = total_b2_ptr[word_index];
            uint64_t b3 = total_b3_ptr[word_index];
            uint64_t b4 = total_b4_ptr[word_index];

            store_hot_result_from_total_with_center(
                b0,
                b1,
                b2,
                b3,
                b4,
                adults[flat_word_index0],
                juveniles[flat_word_index0],
                eggs[flat_word_index0],
                next_adults_ptr[flat_word_index0],
                next_eggs_ptr[flat_word_index0]);

            subtract_horizontal_partial_from_total(
                outgoing_partial0.b0,
                outgoing_partial0.b1,
                outgoing_partial0.b2,
                b0,
                b1,
                b2,
                b3,
                b4);
            add_horizontal_partial_to_total(
                incoming_partial0.b0,
                incoming_partial0.b1,
                incoming_partial0.b2,
                b0,
                b1,
                b2,
                b3,
                b4);

            store_hot_result_from_total_with_center(
                b0,
                b1,
                b2,
                b3,
                b4,
                adults[flat_word_index1],
                juveniles[flat_word_index1],
                eggs[flat_word_index1],
                next_adults_ptr[flat_word_index1],
                next_eggs_ptr[flat_word_index1]);

            subtract_horizontal_partial_from_total(
                outgoing_partial1.b0,
                outgoing_partial1.b1,
                outgoing_partial1.b2,
                b0,
                b1,
                b2,
                b3,
                b4);
            add_horizontal_partial_to_total(
                incoming_partial1.b0,
                incoming_partial1.b1,
                incoming_partial1.b2,
                b0,
                b1,
                b2,
                b3,
                b4);

            total_b0_ptr[word_index] = b0;
            total_b1_ptr[word_index] = b1;
            total_b2_ptr[word_index] = b2;
            total_b3_ptr[word_index] = b3;
            total_b4_ptr[word_index] = b4;
            store_horizontal_partial_scalar(
                recycled_row0,
                word_index,
                incoming_partial0.b0,
                incoming_partial0.b1,
                incoming_partial0.b2);
            store_horizontal_partial_scalar(
                recycled_row1,
                word_index,
                incoming_partial1.b0,
                incoming_partial1.b1,
                incoming_partial1.b2);
        }

        ring_head = (ring_head + 2) % hot_row_scratch.row_partials.size();
    }

    for (; row_index + 1 < hot_row_end; ++row_index) {
        const size_t row_base =
            static_cast<size_t>(row_index) * words_per_row;
        HorizontalPartialRow& recycled_row =
            hot_row_scratch.row_partials[ring_head];
        // raw pointer into slab — no .data() needed.
        HorizontalPartialPair2* __restrict recycled_partial_pairs =
            recycled_row.pairs;
        const HorizontalPartialPair2* __restrict outgoing_partial_pairs =
            recycled_partial_pairs;
        const uint64_t* __restrict incoming_adult_row =
            adults + static_cast<size_t>(row_index + 3) * words_per_row;

        int word_index = hot_word_begin;
        if (use_aligned_main_hot_loop) {
            const size_t flat_word_index = row_base + static_cast<size_t>(word_index);
            const HorizontalPartialWord outgoing_partial =
                load_horizontal_partial_scalar(recycled_row, word_index);
            const HorizontalPartialWord incoming_partial =
                compute_horizontal_partial_scalar_from_adult_row(
                    incoming_adult_row,
                    word_index);
            uint64_t b0 = total_b0_ptr[word_index];
            uint64_t b1 = total_b1_ptr[word_index];
            uint64_t b2 = total_b2_ptr[word_index];
            uint64_t b3 = total_b3_ptr[word_index];
            uint64_t b4 = total_b4_ptr[word_index];

            store_hot_result_from_total_with_center(
                b0,
                b1,
                b2,
                b3,
                b4,
                adults[flat_word_index],
                juveniles[flat_word_index],
                eggs[flat_word_index],
                next_adults_ptr[flat_word_index],
                next_eggs_ptr[flat_word_index]);

            subtract_horizontal_partial_from_total(
                outgoing_partial.b0,
                outgoing_partial.b1,
                outgoing_partial.b2,
                b0,
                b1,
                b2,
                b3,
                b4);
            add_horizontal_partial_to_total(
                incoming_partial.b0,
                incoming_partial.b1,
                incoming_partial.b2,
                b0,
                b1,
                b2,
                b3,
                b4);

            total_b0_ptr[word_index] = b0;
            total_b1_ptr[word_index] = b1;
            total_b2_ptr[word_index] = b2;
            total_b3_ptr[word_index] = b3;
            total_b4_ptr[word_index] = b4;
            store_horizontal_partial_scalar(
                recycled_row,
                word_index,
                incoming_partial.b0,
                incoming_partial.b1,
                incoming_partial.b2);
            ++word_index;

            const int aligned_simd_word_end =
                word_index + ((aligned_hot_word_end - word_index) / sve_words) * sve_words;
            for (; word_index < aligned_simd_word_end; word_index += sve_words) {
                const size_t aligned_flat_word_index =
                    row_base + static_cast<size_t>(word_index);
                const size_t partial_pair_index =
                    horizontal_partial_pair_index(word_index);
                svuint64_t outgoing_partial_b0;
                svuint64_t outgoing_partial_b1;
                svuint64_t outgoing_partial_b2;
                svuint64_t incoming_partial_b0;
                svuint64_t incoming_partial_b1;
                svuint64_t incoming_partial_b2;
                load_horizontal_partial_pair2_u64_sve(
                    pg,
                    outgoing_partial_pairs[partial_pair_index],
                    outgoing_partial_b0,
                    outgoing_partial_b1,
                    outgoing_partial_b2);
                compute_horizontal_partial_pair2_u64_sve_from_adult_row(
                    pg,
                    incoming_adult_row,
                    word_index,
                    incoming_partial_b0,
                    incoming_partial_b1,
                    incoming_partial_b2);

                svuint64_t vb0 = svld1_u64(pg, total_b0_ptr + word_index);
                svuint64_t vb1 = svld1_u64(pg, total_b1_ptr + word_index);
                svuint64_t vb2 = svld1_u64(pg, total_b2_ptr + word_index);
                svuint64_t vb3 = svld1_u64(pg, total_b3_ptr + word_index);
                svuint64_t vb4 = svld1_u64(pg, total_b4_ptr + word_index);

                svuint64_t next_adult_word;
                svuint64_t next_egg_word;
                store_hot_result_from_total_with_center_u64_sve(
                    pg,
                    vb0,
                    vb1,
                    vb2,
                    vb3,
                    vb4,
                    svld1_u64(pg, adults + aligned_flat_word_index),
                    svld1_u64(pg, juveniles + aligned_flat_word_index),
                    svld1_u64(pg, eggs + aligned_flat_word_index),
                    next_adult_word,
                    next_egg_word);

                svst1_u64(pg, next_adults_ptr + aligned_flat_word_index, next_adult_word);
                svst1_u64(pg, next_eggs_ptr + aligned_flat_word_index, next_egg_word);

                subtract_horizontal_partial_from_total_u64_sve(
                    pg,
                    outgoing_partial_b0,
                    outgoing_partial_b1,
                    outgoing_partial_b2,
                    vb0,
                    vb1,
                    vb2,
                    vb3,
                    vb4);
                add_horizontal_partial_to_total_u64_sve(
                    pg,
                    incoming_partial_b0,
                    incoming_partial_b1,
                    incoming_partial_b2,
                    vb0,
                    vb1,
                    vb2,
                    vb3,
                    vb4);

                svst1_u64(pg, total_b0_ptr + word_index, vb0);
                svst1_u64(pg, total_b1_ptr + word_index, vb1);
                svst1_u64(pg, total_b2_ptr + word_index, vb2);
                svst1_u64(pg, total_b3_ptr + word_index, vb3);
                svst1_u64(pg, total_b4_ptr + word_index, vb4);
                store_horizontal_partial_pair2_u64_sve(
                    pg,
                    recycled_partial_pairs[partial_pair_index],
                    incoming_partial_b0,
                    incoming_partial_b1,
                    incoming_partial_b2);
            }
        }

        for (; word_index < hot_word_end; ++word_index) {
            const size_t flat_word_index = row_base + word_index;
            const HorizontalPartialWord outgoing_partial =
                load_horizontal_partial_scalar(recycled_row, word_index);
            const HorizontalPartialWord incoming_partial =
                compute_horizontal_partial_scalar_from_adult_row(
                    incoming_adult_row,
                    word_index);
            uint64_t b0 = total_b0_ptr[word_index];
            uint64_t b1 = total_b1_ptr[word_index];
            uint64_t b2 = total_b2_ptr[word_index];
            uint64_t b3 = total_b3_ptr[word_index];
            uint64_t b4 = total_b4_ptr[word_index];

            store_hot_result_from_total_with_center(
                b0,
                b1,
                b2,
                b3,
                b4,
                adults[flat_word_index],
                juveniles[flat_word_index],
                eggs[flat_word_index],
                next_adults_ptr[flat_word_index],
                next_eggs_ptr[flat_word_index]);

            subtract_horizontal_partial_from_total(
                outgoing_partial.b0,
                outgoing_partial.b1,
                outgoing_partial.b2,
                b0,
                b1,
                b2,
                b3,
                b4);
            add_horizontal_partial_to_total(
                incoming_partial.b0,
                incoming_partial.b1,
                incoming_partial.b2,
                b0,
                b1,
                b2,
                b3,
                b4);

            total_b0_ptr[word_index] = b0;
            total_b1_ptr[word_index] = b1;
            total_b2_ptr[word_index] = b2;
            total_b3_ptr[word_index] = b3;
            total_b4_ptr[word_index] = b4;
            store_horizontal_partial_scalar(
                recycled_row,
                word_index,
                incoming_partial.b0,
                incoming_partial.b1,
                incoming_partial.b2);
        }

        ring_head = (ring_head + 1) % hot_row_scratch.row_partials.size();
    }

    const int final_row = hot_row_end - 1;
    const size_t final_row_base =
        static_cast<size_t>(final_row) * words_per_row;
    int word_index = hot_word_begin;
    if (use_aligned_main_hot_loop) {
        const size_t flat_word_index = final_row_base + static_cast<size_t>(word_index);
        store_hot_result_from_total_with_center(
            total_b0_ptr[word_index],
            total_b1_ptr[word_index],
            total_b2_ptr[word_index],
            total_b3_ptr[word_index],
            total_b4_ptr[word_index],
            adults[flat_word_index],
            juveniles[flat_word_index],
            eggs[flat_word_index],
            next_adults_ptr[flat_word_index],
            next_eggs_ptr[flat_word_index]);
        ++word_index;

        const int aligned_simd_word_end =
            word_index + ((aligned_hot_word_end - word_index) / sve_words) * sve_words;
        for (; word_index < aligned_simd_word_end; word_index += sve_words) {
            const size_t aligned_flat_word_index =
                final_row_base + static_cast<size_t>(word_index);
            svuint64_t next_adult_word;
            svuint64_t next_egg_word;
            store_hot_result_from_total_with_center_u64_sve(
                pg,
                svld1_u64(pg, total_b0_ptr + word_index),
                svld1_u64(pg, total_b1_ptr + word_index),
                svld1_u64(pg, total_b2_ptr + word_index),
                svld1_u64(pg, total_b3_ptr + word_index),
                svld1_u64(pg, total_b4_ptr + word_index),
                svld1_u64(pg, adults + aligned_flat_word_index),
                svld1_u64(pg, juveniles + aligned_flat_word_index),
                svld1_u64(pg, eggs + aligned_flat_word_index),
                next_adult_word,
                next_egg_word);
            svst1_u64(pg, next_adults_ptr + aligned_flat_word_index, next_adult_word);
            svst1_u64(pg, next_eggs_ptr + aligned_flat_word_index, next_egg_word);
        }
    } else {
        for (; word_index < simd_word_end; word_index += sve_words) {
            const size_t flat_word_index = final_row_base + static_cast<size_t>(word_index);
            svuint64_t next_adult_word;
            svuint64_t next_egg_word;
            store_hot_result_from_total_with_center_u64_sve(
                pg,
                svld1_u64(pg, total_b0_ptr + word_index),
                svld1_u64(pg, total_b1_ptr + word_index),
                svld1_u64(pg, total_b2_ptr + word_index),
                svld1_u64(pg, total_b3_ptr + word_index),
                svld1_u64(pg, total_b4_ptr + word_index),
                svld1_u64(pg, adults + flat_word_index),
                svld1_u64(pg, juveniles + flat_word_index),
                svld1_u64(pg, eggs + flat_word_index),
                next_adult_word,
                next_egg_word);
            svst1_u64(pg, next_adults_ptr + flat_word_index, next_adult_word);
            svst1_u64(pg, next_eggs_ptr + flat_word_index, next_egg_word);
        }
    }

    for (; word_index < hot_word_end; ++word_index) {
        const size_t flat_word_index = final_row_base + word_index;
        store_hot_result_from_total_with_center(
            total_b0_ptr[word_index],
            total_b1_ptr[word_index],
            total_b2_ptr[word_index],
            total_b3_ptr[word_index],
            total_b4_ptr[word_index],
            adults[flat_word_index],
            juveniles[flat_word_index],
            eggs[flat_word_index],
            next_adults_ptr[flat_word_index],
            next_eggs_ptr[flat_word_index]);
    }
}

// run_one_generation_left_edge_valid_rows — handles word 0 of hot rows [2, grid_size-2).
//
// The separate init pass has been eliminated.  Instead of pre-zeroing next_eggs and
// pre-copying juveniles into next_adults before this function runs, we now write
// both output words with a plain assignment here:
//
//   next_adults[word0] = juveniles[word0]  |  (adult_survival_mask & valid_bits)
//   next_eggs[word0]   =                      (egg_spawn_mask      & valid_bits)
//
// valid_mask = ~0ULL << 2 covers bits 2..63.  Bits 0 and 1 (the two leftmost
// wrapping columns) are zeroed in adult_valid/empty_valid by valid_mask, so those
// bit positions are left at their juvenile value in next_adults and at zero in
// next_eggs, ready for run_one_generation_wrapping_rows to OR in the wrapping result.
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

        // Direct assignment: juvenile promotion (all bits) combined with adult
        // survival (valid bits only).  Bits 0–1 of adult_valid are zero because
        // valid_mask zeros them, so those bit positions carry only the juvenile
        // value and are safe for the subsequent wrapping OR.
        next_adults_ptr[row_base] =
            juveniles[row_base] | (adult_valid & ge4 & ~ge10);
        next_eggs_ptr[row_base] =
            empty_valid & ge3 & ~ge6;
    }
}

// run_one_generation_right_edge_valid_rows — handles the last word of hot rows.
//
// Same init-pass elimination as the left edge: we assign rather than OR so that
// juvenile promotion is folded in here.  valid_mask = (1<<62)-1 covers bits 0..61;
// bits 62–63 (the two rightmost wrapping columns) are left as juveniles / zero so
// that run_one_generation_wrapping_rows can OR in the wrapping result unchanged.
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

        // Direct assignment: juvenile promotion (all bits) combined with adult
        // survival (valid bits only).  Bits 62–63 of adult_valid are zero,
        // leaving those positions at their juvenile value for the wrapping OR.
        next_adults_ptr[edge_word_index] =
            juveniles[edge_word_index] | (adult_valid & ge4 & ~ge10);
        next_eggs_ptr[edge_word_index] =
            empty_valid & ge3 & ~ge6;
    }
}

// run_one_generation_wrapping_rows — handles cells that require toroidal wrap-around.
//
// For the top two rows and bottom two rows (non-hot rows), this function owns the
// complete initialization of next_adults and next_eggs: it zero-fills next_eggs and
// copies juveniles into next_adults for every word in those rows before applying the
// cell-by-cell wrapping rule.  This replaces what initialize_next_generation_rows
// used to do for those rows.
//
// For interior hot rows, only columns 0–1 (left wrap) and grid_size-2..grid_size-1
// (right wrap) are touched here with |=.  The left/right edge functions have already
// written correct values into word 0 and the last word via assignment, so the |=
// here correctly accumulates the wrapping contribution without needing a separate init.
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

        // Non-hot row: initialize all words here (replaces the removed init pass).
        for (int word_index = 0; word_index < words_per_row; ++word_index) {
            next_adults_ptr[row_base + word_index] = juveniles[row_base + word_index];
            next_eggs_ptr[row_base + word_index] = 0ULL;
        }

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

    // Interior hot rows: handle only the wrapping columns.
    // The left/right edge functions have already written their words via assignment,
    // so |= here is safe: bits 0–1 of next_eggs[word0] are 0 (valid_mask zeroed them),
    // and bits 62–63 of next_eggs[last_word] are similarly 0.
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

        // Non-hot row: initialize all words here (replaces the removed init pass).
        for (int word_index = 0; word_index < words_per_row; ++word_index) {
            next_adults_ptr[row_base + word_index] = juveniles[row_base + word_index];
            next_eggs_ptr[row_base + word_index] = 0ULL;
        }

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
    // Left and right edge functions run first: they write via assignment and fold in
    // juvenile promotion, so wrapping can safely use |= for the two columns that
    // cross the word boundary.
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

// process_generation_chunk — no separate init pass.
//
// Previously this called initialize_next_generation_rows first to zero next_eggs
// and copy juveniles into next_adults before the hot and edge passes.  That pass
// has been eliminated: the hot-row function writes interior words via direct
// assignment, the left/right edge functions write edge words via direct assignment
// (now including juvenile promotion), and the wrapping function initializes the
// non-hot rows internally.  All words are therefore covered without a redundant
// pre-pass.
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
            "Error: grid must be square and non-empty, got %" PRIu64 " × %" PRIu64 "\n",
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