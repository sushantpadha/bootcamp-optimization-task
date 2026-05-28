// spawn_sim_neon.cpp — NEON intrinsics hot path with vsriq/vsliq/vbicq
//
// Drop-in replacement for spawn_sim_bic_all.cpp that replaces every SVE
// call with ARM NEON intrinsics (uint64x2_t, always 2×64-bit lanes).
//
// Key optimisations over the scalar fallback that the SVE path also used:
//   vsriq_n_u64 / vsliq_n_u64  — fuse the shift-then-OR for each of the
//      four horizontal neighbour columns into a single SRI/SLI instruction,
//      saving one ORR per column (4 per word) in the horizontal-partial step.
//   vbicq_u64 — eliminates all explicit NOT temporaries from the predicate
//      evaluation and the rolling-subtract hot path.
//   No svbool_t predicate registers or svcntd() runtime queries.
//
// INPUT FORMAT:
//   Binary file, little-endian:
//     bytes  0– 7: uint64_t width
//     bytes  8–15: uint64_t height (must equal width; only square grids)
//     bytes 16–  : width × height bytes, 0=EMPTY 1=EGG 2=JUVENILE 3=ADULT
//
// OUTPUT FORMAT: same binary format, after the requested generations.
//
// USAGE:
//   spawn_sim <input.bin> <output.bin> [generations]
//   generations defaults to 10000.
//
// TIMING: prints wall-clock simulation time (excluding I/O) as "<N.NNN> ms".

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

#include <arm_neon.h>

static constexpr uint8_t EMPTY    = 0;
static constexpr uint8_t EGG      = 1;
static constexpr uint8_t JUVENILE = 2;
static constexpr uint8_t ADULT    = 3;

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count)
    {
        if (count > static_cast<std::size_t>(-1) / sizeof(T))
            throw std::bad_array_new_length();
        void* pointer = nullptr;
        if (posix_memalign(&pointer, Alignment, count * sizeof(T)) != 0)
            throw std::bad_alloc();
        return static_cast<T*>(pointer);
    }

    void deallocate(T* pointer, std::size_t) noexcept { std::free(pointer); }

    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&)
{ return true; }

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&)
{ return false; }

using BitPlane = std::vector<uint64_t, AlignedAllocator<uint64_t, 64>>;

struct RowRange {
    int begin = 0;
    int end   = 0;
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

static_assert(sizeof(HorizontalPartialPair2)  == sizeof(uint64_t) * 6);
static_assert(alignof(HorizontalPartialPair2) == 16);

using HorizontalPartialPair2Buffer =
    std::vector<HorizontalPartialPair2, AlignedAllocator<HorizontalPartialPair2, 64>>;

struct HorizontalPartialRow {
    HorizontalPartialPair2Buffer pairs;

    HorizontalPartialRow() = default;
    explicit HorizontalPartialRow(int words_per_row)
        : pairs(static_cast<size_t>(words_per_row + 1) >> 1) {}
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
    const HorizontalPartialRow& row_partial, int word_index)
{
    const HorizontalPartialPair2& pair =
        row_partial.pairs[horizontal_partial_pair_index(word_index)];
    const int lane = horizontal_partial_pair_lane(word_index);
    return HorizontalPartialWord{ pair.b0[lane], pair.b1[lane], pair.b2[lane] };
}

inline void store_horizontal_partial_scalar(
    HorizontalPartialRow& row_partial, int word_index,
    uint64_t partial_b0, uint64_t partial_b1, uint64_t partial_b2)
{
    HorizontalPartialPair2& pair =
        row_partial.pairs[horizontal_partial_pair_index(word_index)];
    const int lane = horizontal_partial_pair_lane(word_index);
    pair.b0[lane] = partial_b0;
    pair.b1[lane] = partial_b1;
    pair.b2[lane] = partial_b2;
}

// ---------------------------------------------------------------------------
// NEON pair store / load
// ---------------------------------------------------------------------------

inline void store_horizontal_partial_pair2_u64_neon(
    HorizontalPartialPair2& pair,
    uint64x2_t partial_b0, uint64x2_t partial_b1, uint64x2_t partial_b2)
{
    vst1q_u64(pair.b0, partial_b0);
    vst1q_u64(pair.b1, partial_b1);
    vst1q_u64(pair.b2, partial_b2);
}

inline void load_horizontal_partial_pair2_u64_neon(
    const HorizontalPartialPair2& pair,
    uint64x2_t& partial_b0, uint64x2_t& partial_b1, uint64x2_t& partial_b2)
{
    partial_b0 = vld1q_u64(pair.b0);
    partial_b1 = vld1q_u64(pair.b1);
    partial_b2 = vld1q_u64(pair.b2);
}

// ---------------------------------------------------------------------------
// Scalar carry-save-add (used in scalar preamble / tail)
// ---------------------------------------------------------------------------

inline void carry_save_add(uint64_t a, uint64_t b, uint64_t c,
                           uint64_t& sum, uint64_t& carry)
{
    sum   = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
}

// ---------------------------------------------------------------------------
// NEON carry-save-add
// sum   = a ^ b ^ c
// carry = majority(a,b,c) = (a&b) | ((a|b) & c)
//       = vbslq_u64(c, a|b, a&b)   [BSL(mask,hi,lo) = (hi&mask)|(lo&~mask)]
// ---------------------------------------------------------------------------

inline void carry_save_add_u64_neon(
    uint64x2_t a, uint64x2_t b, uint64x2_t c,
    uint64x2_t& sum, uint64x2_t& carry)
{
    sum   = veorq_u64(veorq_u64(a, b), c);
    carry = vbslq_u64(c, vorrq_u64(a, b), vandq_u64(a, b));
}

// ---------------------------------------------------------------------------
// Horizontal partial computation — NEON with SRI/SLI fused shift-OR
//
// SRI semantics: vsriq_n_u64(Vd, Vm, n)
//   result = (Vd & top_n_mask) | (Vm >> n)
//   Since Vd = curr<<k has the bottom k bits zero, the top (64-k) bits of Vd
//   stay intact and the bottom k bits come from Vm>>62 or Vm>>63, giving:
//     left_2 = (curr<<2) | (prev>>62)   one SHL + one SRI  (saves one ORR)
//     left_1 = (curr<<1) | (prev>>63)   one SHL + one SRI
//
// SLI semantics: vsliq_n_u64(Vd, Vm, n)
//   result = (Vm << n) | (Vd & bottom_n_mask)
//     right_1 = (curr>>1) | (next<<63)   one SHR + one SLI
//     right_2 = (curr>>2) | (next<<62)   one SHR + one SLI
// ---------------------------------------------------------------------------

__attribute__((always_inline)) static inline void
compute_horizontal_partial_pair2_u64_neon_from_adult_row(
    const uint64_t* __restrict adult_row,
    int word_index,
    uint64x2_t& partial_b0, uint64x2_t& partial_b1, uint64x2_t& partial_b2)
{
    // Overlapping loads give each lane its correct neighbour words:
    //   prev_adj lane i = adult_row[word_index + i - 1]
    //   curr_v   lane i = adult_row[word_index + i]
    //   next_adj lane i = adult_row[word_index + i + 1]
    const uint64x2_t prev_adj = vld1q_u64(adult_row + word_index - 1);
    const uint64x2_t curr_v   = vld1q_u64(adult_row + word_index);
    const uint64x2_t next_adj = vld1q_u64(adult_row + word_index + 1);

    const uint64x2_t left_2  = vsriq_n_u64(vshlq_n_u64(curr_v, 2), prev_adj, 62);
    const uint64x2_t left_1  = vsriq_n_u64(vshlq_n_u64(curr_v, 1), prev_adj, 63);
    const uint64x2_t right_1 = vsliq_n_u64(vshrq_n_u64(curr_v, 1), next_adj, 63);
    const uint64x2_t right_2 = vsliq_n_u64(vshrq_n_u64(curr_v, 2), next_adj, 62);

    uint64x2_t sum_1, carry_1, sum_2, carry_2;
    carry_save_add_u64_neon(left_2, left_1, curr_v, sum_1, carry_1);
    carry_save_add_u64_neon(right_1, right_2, sum_1, sum_2, carry_2);

    partial_b0 = sum_2;
    partial_b1 = veorq_u64(carry_1, carry_2);
    partial_b2 = vandq_u64(carry_1, carry_2);
}

// ---------------------------------------------------------------------------
// Scalar horizontal partial (used for boundary words)
// ---------------------------------------------------------------------------

__attribute__((always_inline)) static inline HorizontalPartialWord
compute_horizontal_partial_scalar_from_adult_row(
    const uint64_t* __restrict adult_row, int word_index)
{
    const uint64_t prev   = adult_row[word_index - 1];
    const uint64_t curr   = adult_row[word_index];
    const uint64_t next   = adult_row[word_index + 1];
    const uint64_t left_2  = (curr << 2) | (prev >> 62);
    const uint64_t left_1  = (curr << 1) | (prev >> 63);
    const uint64_t right_1 = (curr >> 1) | (next << 63);
    const uint64_t right_2 = (curr >> 2) | (next << 62);

    uint64_t sum_1 = 0ULL, carry_1 = 0ULL, sum_2 = 0ULL, carry_2 = 0ULL;
    carry_save_add(left_2, left_1, curr, sum_1, carry_1);
    carry_save_add(right_1, right_2, sum_1, sum_2, carry_2);
    return HorizontalPartialWord{ sum_2, carry_1 ^ carry_2, carry_1 & carry_2 };
}

// ---------------------------------------------------------------------------
// VerticalAccumulatorRow / HotRowScratch
// ---------------------------------------------------------------------------

struct VerticalAccumulatorRow {
    BitPlane b0, b1, b2, b3, b4;

    VerticalAccumulatorRow() = default;
    explicit VerticalAccumulatorRow(int words_per_row)
        : b0(static_cast<size_t>(words_per_row), 0ULL),
          b1(static_cast<size_t>(words_per_row), 0ULL),
          b2(static_cast<size_t>(words_per_row), 0ULL),
          b3(static_cast<size_t>(words_per_row), 0ULL),
          b4(static_cast<size_t>(words_per_row), 0ULL) {}
};

struct HotRowScratch {
    std::array<HorizontalPartialRow, 5> row_partials;
    HorizontalPartialRow incoming_partial;
    VerticalAccumulatorRow rolling_total;

    HotRowScratch() = default;
    explicit HotRowScratch(int words_per_row)
    {
        for (HorizontalPartialRow& rp : row_partials)
            rp = HorizontalPartialRow(words_per_row);
        incoming_partial = HorizontalPartialRow(words_per_row);
        rolling_total    = VerticalAccumulatorRow(words_per_row);
    }
};

// ---------------------------------------------------------------------------
// Thread helpers
// ---------------------------------------------------------------------------

static RowRange compute_row_range(int grid_size, int thread_count, int worker_id)
{
    const int base_rows = grid_size / thread_count;
    const int remainder = grid_size % thread_count;
    const int begin = worker_id * base_rows + std::min(worker_id, remainder);
    const int count = base_rows + (worker_id < remainder ? 1 : 0);
    return RowRange{begin, begin + count};
}

static void pin_current_thread_to_cpu(int worker_id)
{
    cpu_set_t affinity_mask;
    CPU_ZERO(&affinity_mask);
    CPU_SET(worker_id, &affinity_mask);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(affinity_mask), &affinity_mask);
}

// ---------------------------------------------------------------------------
// Bit-plane helpers
// ---------------------------------------------------------------------------

inline int wrap_coordinate(int coordinate, int grid_size)
{
    return (coordinate + grid_size) & (grid_size - 1);
}

inline size_t get_flat_word_index(int row_index, int col_index, int words_per_row)
{
    return static_cast<size_t>(row_index) * words_per_row + (col_index >> 6);
}

inline uint64_t get_bit_mask(int col_index)
{
    return 1ULL << (col_index & 63);
}

inline bool is_cell_set(const BitPlane& bit_plane, int words_per_row,
                        int row_index, int col_index)
{
    return (bit_plane[get_flat_word_index(row_index, col_index, words_per_row)]
            & get_bit_mask(col_index)) != 0;
}

inline void set_cell_bit(BitPlane& bit_plane, int words_per_row,
                         int row_index, int col_index)
{
    bit_plane[get_flat_word_index(row_index, col_index, words_per_row)]
        |= get_bit_mask(col_index);
}

static void pack_input_into_bit_planes(
    const std::vector<uint8_t>& input_grid, int grid_size, int words_per_row,
    BitPlane& current_adults, BitPlane& current_juveniles, BitPlane& current_eggs)
{
    for (int row_index = 0; row_index < grid_size; ++row_index) {
        for (int col_index = 0; col_index < grid_size; ++col_index) {
            uint8_t cell_state =
                input_grid[static_cast<size_t>(row_index) * grid_size + col_index];
            if      (cell_state == ADULT)    set_cell_bit(current_adults,    words_per_row, row_index, col_index);
            else if (cell_state == JUVENILE) set_cell_bit(current_juveniles, words_per_row, row_index, col_index);
            else if (cell_state == EGG)      set_cell_bit(current_eggs,      words_per_row, row_index, col_index);
        }
    }
}

static int count_adult_neighbors_naive(
    const BitPlane& current_adults, int grid_size, int words_per_row,
    int center_row, int center_col)
{
    int adult_neighbor_count = 0;
    for (int row_offset = -2; row_offset <= 2; ++row_offset) {
        int neighbor_row = wrap_coordinate(center_row + row_offset, grid_size);
        for (int col_offset = -2; col_offset <= 2; ++col_offset) {
            if (row_offset == 0 && col_offset == 0) continue;
            int neighbor_col = wrap_coordinate(center_col + col_offset, grid_size);
            if (is_cell_set(current_adults, words_per_row, neighbor_row, neighbor_col))
                ++adult_neighbor_count;
        }
    }
    return adult_neighbor_count;
}

// ---------------------------------------------------------------------------
// Scalar add/subtract rolling total
// ---------------------------------------------------------------------------

inline void add_horizontal_partial_to_total(
    uint64_t partial_b0, uint64_t partial_b1, uint64_t partial_b2,
    uint64_t& total_b0, uint64_t& total_b1, uint64_t& total_b2,
    uint64_t& total_b3, uint64_t& total_b4)
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
    uint64_t partial_b0, uint64_t partial_b1, uint64_t partial_b2,
    uint64_t& total_b0, uint64_t& total_b1, uint64_t& total_b2,
    uint64_t& total_b3, uint64_t& total_b4)
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

// ---------------------------------------------------------------------------
// NEON add rolling total
// ---------------------------------------------------------------------------

inline void add_horizontal_partial_to_total_u64_neon(
    uint64x2_t partial_b0, uint64x2_t partial_b1, uint64x2_t partial_b2,
    uint64x2_t& total_b0, uint64x2_t& total_b1, uint64x2_t& total_b2,
    uint64x2_t& total_b3, uint64x2_t& total_b4)
{
    uint64x2_t carry = vandq_u64(total_b0, partial_b0);
    total_b0 = veorq_u64(total_b0, partial_b0);

    carry_save_add_u64_neon(total_b1, partial_b1, carry, total_b1, carry);
    carry_save_add_u64_neon(total_b2, partial_b2, carry, total_b2, carry);

    const uint64x2_t carry4 = vandq_u64(total_b3, carry);
    total_b3 = veorq_u64(total_b3, carry);
    total_b4 = veorq_u64(total_b4, carry4);
}

// ---------------------------------------------------------------------------
// NEON subtract rolling total
//
// borrow_out(b, p, bin) = ~b & (p | bin) | p & bin
//
// BSL form: vbslq(bin, ~b|p, p&~b)
//   = (~b|p) & bin | (p&~b) & ~bin  [expand]
//   = ~b&(p|bin) | p&bin             [simplify] ✓
// Saves 1 op per stage vs the 4-op vorrq/vbicq/vorrq/vandq form.
// vornq_u64(p, b) = p | ~b  provides ~b|p without a separate NOT.
// ---------------------------------------------------------------------------

inline void subtract_horizontal_partial_from_total_u64_neon(
    uint64x2_t partial_b0, uint64x2_t partial_b1, uint64x2_t partial_b2,
    uint64x2_t& total_b0, uint64x2_t& total_b1, uint64x2_t& total_b2,
    uint64x2_t& total_b3, uint64x2_t& total_b4)
{
    uint64x2_t borrow = vbicq_u64(partial_b0, total_b0);
    total_b0 = veorq_u64(total_b0, partial_b0);

    uint64x2_t next_borrow = vbslq_u64(
        borrow,
        vornq_u64(partial_b1, total_b1),   // ~b1 | p1
        vbicq_u64(partial_b1, total_b1));  // p1 & ~b1
    total_b1 = veorq_u64(veorq_u64(total_b1, partial_b1), borrow);
    borrow = next_borrow;

    next_borrow = vbslq_u64(
        borrow,
        vornq_u64(partial_b2, total_b2),   // ~b2 | p2
        vbicq_u64(partial_b2, total_b2));  // p2 & ~b2
    total_b2 = veorq_u64(veorq_u64(total_b2, partial_b2), borrow);
    borrow = next_borrow;

    next_borrow = vbicq_u64(borrow, total_b3);
    total_b3 = veorq_u64(total_b3, borrow);
    total_b4 = veorq_u64(total_b4, next_borrow);
}

// ---------------------------------------------------------------------------
// Scalar result store (centre included in total — ranges pre-shifted)
// ---------------------------------------------------------------------------

inline void store_hot_result_from_total_with_center(
    uint64_t total_b0, uint64_t total_b1, uint64_t total_b2,
    uint64_t total_b3, uint64_t total_b4,
    uint64_t adult_word, uint64_t juvenile_word, uint64_t egg_word,
    uint64_t& next_adult_word, uint64_t& next_egg_word)
{
    const uint64_t blocked_word = juvenile_word | egg_word;
    const uint64_t not_b4  = ~total_b4;
    const uint64_t not_b3  = ~total_b3;
    const uint64_t not_b2  = ~total_b2;
    const uint64_t not_b1  = ~total_b1;
    const uint64_t b1_or_b0  = total_b1 | total_b0;
    const uint64_t b1_and_b0 = total_b1 & total_b0;

    const uint64_t adult_5_to_10 =
        not_b4 &
        ((not_b3 & total_b2 & b1_or_b0) |
         (total_b3 & not_b2 & ~b1_and_b0));
    const uint64_t egg_3_to_5 =
        not_b4 & not_b3 &
        ((not_b2 & b1_and_b0) | (total_b2 & not_b1));

    next_adult_word = (adult_word & adult_5_to_10) | juvenile_word;
    next_egg_word   = ~(adult_word | blocked_word) & egg_3_to_5;
}

// ---------------------------------------------------------------------------
// NEON result store
//
// All NOT operations expressed as vbicq_u64 — no complement temporaries.
// adult_5_to_10 = ~b4 & ((~b3&b2&(b1|b0)) | (b3&~b2&~(b1&b0)))
// egg_3_to_5    = ~b4 & ~b3 & ((~b2&(b1&b0)) | (b2&~b1))
// ~occupied & egg_3_to_5 expressed directly as vbicq(egg_3_to_5, occupied).
//
// next_adult_word = (adult & adult_5_to_10) | juvenile.
// adult and juvenile are mutually exclusive bit-planes (adult & juvenile = 0),
// so juvenile & ~adult = juvenile exactly, giving:
//   vbslq(adult_word, adult_5_to_10, juvenile_word)
//   = (adult_5_to_10 & adult) | (juvenile & ~adult) = same  [1 op vs 2].
// ---------------------------------------------------------------------------

inline void store_hot_result_from_total_with_center_u64_neon(
    uint64x2_t total_b0, uint64x2_t total_b1, uint64x2_t total_b2,
    uint64x2_t total_b3, uint64x2_t total_b4,
    uint64x2_t adult_word, uint64x2_t juvenile_word, uint64x2_t egg_word,
    uint64x2_t& next_adult_word, uint64x2_t& next_egg_word)
{
    const uint64x2_t blocked_word  = vorrq_u64(juvenile_word, egg_word);
    const uint64x2_t occupied_word = vorrq_u64(adult_word, blocked_word);
    const uint64x2_t b1_or_b0      = vorrq_u64(total_b1, total_b0);
    const uint64x2_t b1_and_b0     = vandq_u64(total_b1, total_b0);

    // ~b3&b2&(b1|b0)  ← vbicq(b2,b3) & (b1|b0)
    const uint64x2_t A = vandq_u64(vbicq_u64(total_b2, total_b3), b1_or_b0);
    // b3&~b2&~(b1&b0) ← vbicq(vbicq(b3,b2), b1_and_b0)
    const uint64x2_t B = vbicq_u64(vbicq_u64(total_b3, total_b2), b1_and_b0);
    // ~b4 & (A|B)     ← vbicq(A|B, b4)
    const uint64x2_t adult_5_to_10 = vbicq_u64(vorrq_u64(A, B), total_b4);

    // ~b2&(b1&b0) ← vbicq(b1_and_b0, b2)
    const uint64x2_t C = vbicq_u64(b1_and_b0, total_b2);
    // b2&~b1      ← vbicq(b2, b1)
    const uint64x2_t D = vbicq_u64(total_b2, total_b1);
    // ~b4 & ~b3 & (C|D) ← vbicq(vbicq(C|D, b3), b4)
    const uint64x2_t egg_3_to_5 =
        vbicq_u64(vbicq_u64(vorrq_u64(C, D), total_b3), total_b4);

    next_adult_word = vbslq_u64(adult_word, adult_5_to_10, juvenile_word);
    next_egg_word   = vbicq_u64(egg_3_to_5, occupied_word);
}

// ---------------------------------------------------------------------------
// Compute horizontal partial for one row
// ---------------------------------------------------------------------------

__attribute__((always_inline)) static inline void compute_horizontal_partial_row(
    int source_row, int words_per_row,
    const BitPlane& current_adults, HorizontalPartialRow& row_partial)
{
    constexpr int neon_words = 2;

    const uint64_t* __restrict adults = current_adults.data();
    HorizontalPartialPair2* __restrict partial_pairs = row_partial.pairs.data();
    const int hot_word_begin = 1;
    const int hot_word_end   = words_per_row - 1;

    if (hot_word_begin >= hot_word_end) return;

    const uint64_t* __restrict adult_row =
        adults + static_cast<size_t>(source_row) * words_per_row;

    int word_index = hot_word_begin;

    if (hot_word_begin + 1 < hot_word_end) {
        // Scalar preamble: word at hot_word_begin (index 1) is the odd one out
        // because the NEON loop pairs words starting at an even index (2, 4, 6, …)
        {
            const uint64_t prev   = adult_row[word_index - 1];
            const uint64_t curr   = adult_row[word_index];
            const uint64_t next   = adult_row[word_index + 1];
            const uint64_t left_2  = (curr << 2) | (prev >> 62);
            const uint64_t left_1  = (curr << 1) | (prev >> 63);
            const uint64_t right_1 = (curr >> 1) | (next << 63);
            const uint64_t right_2 = (curr >> 2) | (next << 62);
            uint64_t sum_1 = 0, carry_1 = 0, sum_2 = 0, carry_2 = 0;
            carry_save_add(left_2, left_1, curr, sum_1, carry_1);
            carry_save_add(right_1, right_2, sum_1, sum_2, carry_2);
            store_horizontal_partial_scalar(row_partial, word_index,
                sum_2, carry_1 ^ carry_2, carry_1 & carry_2);
        }
        ++word_index;

        const int aligned_neon_word_end =
            word_index + ((hot_word_end - word_index) / neon_words) * neon_words;
        for (; word_index < aligned_neon_word_end; word_index += neon_words) {
            uint64x2_t partial_b0, partial_b1, partial_b2;
            compute_horizontal_partial_pair2_u64_neon_from_adult_row(
                adult_row, word_index, partial_b0, partial_b1, partial_b2);
            store_horizontal_partial_pair2_u64_neon(
                partial_pairs[horizontal_partial_pair_index(word_index)],
                partial_b0, partial_b1, partial_b2);
        }
    }

    for (; word_index < hot_word_end; ++word_index) {
        const uint64_t prev   = adult_row[word_index - 1];
        const uint64_t curr   = adult_row[word_index];
        const uint64_t next   = adult_row[word_index + 1];
        const uint64_t left_2  = (curr << 2) | (prev >> 62);
        const uint64_t left_1  = (curr << 1) | (prev >> 63);
        const uint64_t right_1 = (curr >> 1) | (next << 63);
        const uint64_t right_2 = (curr >> 2) | (next << 62);
        uint64_t sum_1 = 0, carry_1 = 0, sum_2 = 0, carry_2 = 0;
        carry_save_add(left_2, left_1, curr, sum_1, carry_1);
        carry_save_add(right_1, right_2, sum_1, sum_2, carry_2);
        store_horizontal_partial_scalar(row_partial, word_index,
            sum_2, carry_1 ^ carry_2, carry_1 & carry_2);
    }
}

// ---------------------------------------------------------------------------
// Build initial vertical accumulator from 5 row partials
// ---------------------------------------------------------------------------

static void build_vertical_accumulator_row(
    int words_per_row,
    const std::array<HorizontalPartialRow, 5>& row_partials,
    VerticalAccumulatorRow& rolling_total)
{
    const int hot_word_begin = 1;
    const int hot_word_end   = words_per_row - 1;
    if (hot_word_begin >= hot_word_end) return;

    for (int word_index = hot_word_begin; word_index < hot_word_end; ++word_index) {
        uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0;
        for (const HorizontalPartialRow& rp : row_partials) {
            const HorizontalPartialWord pw =
                load_horizontal_partial_scalar(rp, word_index);
            add_horizontal_partial_to_total(pw.b0, pw.b1, pw.b2,
                                            b0, b1, b2, b3, b4);
        }
        rolling_total.b0[word_index] = b0;
        rolling_total.b1[word_index] = b1;
        rolling_total.b2[word_index] = b2;
        rolling_total.b3[word_index] = b3;
        rolling_total.b4[word_index] = b4;
    }
}

// ---------------------------------------------------------------------------
// Hot-row main loop — NEON inner loop
// ---------------------------------------------------------------------------

__attribute__((noinline)) static void run_one_generation_hot_rows(
    int row_begin, int row_end, int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults,
    BitPlane& next_eggs,
    HotRowScratch& hot_row_scratch)
{
    constexpr int neon_words = 2;

    const uint64_t* __restrict adults       = current_adults.data();
    const uint64_t* __restrict juveniles    = current_juveniles.data();
    const uint64_t* __restrict eggs         = current_eggs.data();
    uint64_t* __restrict next_adults_ptr    = next_adults.data();
    uint64_t* __restrict next_eggs_ptr      = next_eggs.data();
    uint64_t* __restrict total_b0_ptr       = hot_row_scratch.rolling_total.b0.data();
    uint64_t* __restrict total_b1_ptr       = hot_row_scratch.rolling_total.b1.data();
    uint64_t* __restrict total_b2_ptr       = hot_row_scratch.rolling_total.b2.data();
    uint64_t* __restrict total_b3_ptr       = hot_row_scratch.rolling_total.b3.data();
    uint64_t* __restrict total_b4_ptr       = hot_row_scratch.rolling_total.b4.data();

    const int hot_row_begin  = std::max(row_begin, 2);
    const int hot_row_end    = std::min(row_end, grid_size - 2);
    const int hot_word_begin = 1;
    const int hot_word_end   = words_per_row - 1;

    if (hot_row_begin >= hot_row_end || hot_word_begin >= hot_word_end) return;

    for (int ring_offset = 0; ring_offset < 5; ++ring_offset) {
        compute_horizontal_partial_row(
            hot_row_begin + ring_offset - 2, words_per_row,
            current_adults,
            hot_row_scratch.row_partials[static_cast<size_t>(ring_offset)]);
    }
    build_vertical_accumulator_row(words_per_row, hot_row_scratch.row_partials,
                                   hot_row_scratch.rolling_total);

    // aligned_hot_word_end: last NEON-safe starting position (need wi+1 < hot_word_end)
    const int aligned_hot_word_end = hot_word_end - 1;
    const bool use_neon = (hot_word_begin + 1 < hot_word_end);

    size_t ring_head = 0;
    int row_index = hot_row_begin;

    // ---- Two-row blocked main loop ----------------------------------------
    for (; row_index + 2 < hot_row_end; row_index += 2) {
        const size_t row0_base = static_cast<size_t>(row_index)     * words_per_row;
        const size_t row1_base = static_cast<size_t>(row_index + 1) * words_per_row;

        HorizontalPartialRow& recycled_row0 =
            hot_row_scratch.row_partials[ring_head];
        HorizontalPartialRow& recycled_row1 =
            hot_row_scratch.row_partials[(ring_head + 1) % hot_row_scratch.row_partials.size()];
        HorizontalPartialPair2* __restrict recycled_pairs0 = recycled_row0.pairs.data();
        HorizontalPartialPair2* __restrict recycled_pairs1 = recycled_row1.pairs.data();
        const HorizontalPartialPair2* __restrict outgoing_pairs0 = recycled_pairs0;
        const HorizontalPartialPair2* __restrict outgoing_pairs1 = recycled_pairs1;
        const uint64_t* __restrict incoming_adult_row0 =
            adults + static_cast<size_t>(row_index + 3) * words_per_row;
        const uint64_t* __restrict incoming_adult_row1 =
            adults + static_cast<size_t>(row_index + 4) * words_per_row;

        int word_index = hot_word_begin;

        if (use_neon) {
            // -- Scalar preamble (word 1) for row 0 and row 1 ----------------
            {
                const size_t fi0 = row0_base + static_cast<size_t>(word_index);
                const size_t fi1 = row1_base + static_cast<size_t>(word_index);
                const HorizontalPartialWord out0 =
                    load_horizontal_partial_scalar(recycled_row0, word_index);
                const HorizontalPartialWord in0  =
                    compute_horizontal_partial_scalar_from_adult_row(incoming_adult_row0, word_index);
                const HorizontalPartialWord out1 =
                    load_horizontal_partial_scalar(recycled_row1, word_index);
                const HorizontalPartialWord in1  =
                    compute_horizontal_partial_scalar_from_adult_row(incoming_adult_row1, word_index);

                uint64_t b0 = total_b0_ptr[word_index];
                uint64_t b1 = total_b1_ptr[word_index];
                uint64_t b2 = total_b2_ptr[word_index];
                uint64_t b3 = total_b3_ptr[word_index];
                uint64_t b4 = total_b4_ptr[word_index];

                store_hot_result_from_total_with_center(
                    b0, b1, b2, b3, b4,
                    adults[fi0], juveniles[fi0], eggs[fi0],
                    next_adults_ptr[fi0], next_eggs_ptr[fi0]);
                subtract_horizontal_partial_from_total(
                    out0.b0, out0.b1, out0.b2, b0, b1, b2, b3, b4);
                add_horizontal_partial_to_total(
                    in0.b0, in0.b1, in0.b2, b0, b1, b2, b3, b4);

                store_hot_result_from_total_with_center(
                    b0, b1, b2, b3, b4,
                    adults[fi1], juveniles[fi1], eggs[fi1],
                    next_adults_ptr[fi1], next_eggs_ptr[fi1]);
                subtract_horizontal_partial_from_total(
                    out1.b0, out1.b1, out1.b2, b0, b1, b2, b3, b4);
                add_horizontal_partial_to_total(
                    in1.b0, in1.b1, in1.b2, b0, b1, b2, b3, b4);

                total_b0_ptr[word_index] = b0; total_b1_ptr[word_index] = b1;
                total_b2_ptr[word_index] = b2; total_b3_ptr[word_index] = b3;
                total_b4_ptr[word_index] = b4;
                store_horizontal_partial_scalar(recycled_row0, word_index,
                    in0.b0, in0.b1, in0.b2);
                store_horizontal_partial_scalar(recycled_row1, word_index,
                    in1.b0, in1.b1, in1.b2);
            }
            ++word_index;

            const int aligned_neon_end =
                word_index + ((aligned_hot_word_end - word_index) / neon_words) * neon_words;

            for (; word_index < aligned_neon_end; word_index += neon_words) {
                const size_t afi0 = row0_base + static_cast<size_t>(word_index);
                const size_t afi1 = row1_base + static_cast<size_t>(word_index);
                const size_t pp_idx = horizontal_partial_pair_index(word_index);

                uint64x2_t out0_b0, out0_b1, out0_b2;
                uint64x2_t in0_b0,  in0_b1,  in0_b2;
                uint64x2_t out1_b0, out1_b1, out1_b2;
                uint64x2_t in1_b0,  in1_b1,  in1_b2;

                load_horizontal_partial_pair2_u64_neon(
                    outgoing_pairs0[pp_idx], out0_b0, out0_b1, out0_b2);
                compute_horizontal_partial_pair2_u64_neon_from_adult_row(
                    incoming_adult_row0, word_index, in0_b0, in0_b1, in0_b2);
                load_horizontal_partial_pair2_u64_neon(
                    outgoing_pairs1[pp_idx], out1_b0, out1_b1, out1_b2);
                compute_horizontal_partial_pair2_u64_neon_from_adult_row(
                    incoming_adult_row1, word_index, in1_b0, in1_b1, in1_b2);

                uint64x2_t vb0 = vld1q_u64(total_b0_ptr + word_index);
                uint64x2_t vb1 = vld1q_u64(total_b1_ptr + word_index);
                uint64x2_t vb2 = vld1q_u64(total_b2_ptr + word_index);
                uint64x2_t vb3 = vld1q_u64(total_b3_ptr + word_index);
                uint64x2_t vb4 = vld1q_u64(total_b4_ptr + word_index);

                uint64x2_t next_adult, next_egg;

                store_hot_result_from_total_with_center_u64_neon(
                    vb0, vb1, vb2, vb3, vb4,
                    vld1q_u64(adults    + afi0),
                    vld1q_u64(juveniles + afi0),
                    vld1q_u64(eggs      + afi0),
                    next_adult, next_egg);
                vst1q_u64(next_adults_ptr + afi0, next_adult);
                vst1q_u64(next_eggs_ptr   + afi0, next_egg);

                subtract_horizontal_partial_from_total_u64_neon(
                    out0_b0, out0_b1, out0_b2, vb0, vb1, vb2, vb3, vb4);
                add_horizontal_partial_to_total_u64_neon(
                    in0_b0, in0_b1, in0_b2, vb0, vb1, vb2, vb3, vb4);

                store_hot_result_from_total_with_center_u64_neon(
                    vb0, vb1, vb2, vb3, vb4,
                    vld1q_u64(adults    + afi1),
                    vld1q_u64(juveniles + afi1),
                    vld1q_u64(eggs      + afi1),
                    next_adult, next_egg);
                vst1q_u64(next_adults_ptr + afi1, next_adult);
                vst1q_u64(next_eggs_ptr   + afi1, next_egg);

                subtract_horizontal_partial_from_total_u64_neon(
                    out1_b0, out1_b1, out1_b2, vb0, vb1, vb2, vb3, vb4);
                add_horizontal_partial_to_total_u64_neon(
                    in1_b0, in1_b1, in1_b2, vb0, vb1, vb2, vb3, vb4);

                vst1q_u64(total_b0_ptr + word_index, vb0);
                vst1q_u64(total_b1_ptr + word_index, vb1);
                vst1q_u64(total_b2_ptr + word_index, vb2);
                vst1q_u64(total_b3_ptr + word_index, vb3);
                vst1q_u64(total_b4_ptr + word_index, vb4);
                store_horizontal_partial_pair2_u64_neon(
                    recycled_pairs0[pp_idx], in0_b0, in0_b1, in0_b2);
                store_horizontal_partial_pair2_u64_neon(
                    recycled_pairs1[pp_idx], in1_b0, in1_b1, in1_b2);
            }
        }

        // Scalar tail (and full scalar path when use_neon is false)
        for (; word_index < hot_word_end; ++word_index) {
            const size_t fi0 = row0_base + word_index;
            const size_t fi1 = row1_base + word_index;
            const HorizontalPartialWord out0 =
                load_horizontal_partial_scalar(recycled_row0, word_index);
            const HorizontalPartialWord in0  =
                compute_horizontal_partial_scalar_from_adult_row(incoming_adult_row0, word_index);
            const HorizontalPartialWord out1 =
                load_horizontal_partial_scalar(recycled_row1, word_index);
            const HorizontalPartialWord in1  =
                compute_horizontal_partial_scalar_from_adult_row(incoming_adult_row1, word_index);

            uint64_t b0 = total_b0_ptr[word_index];
            uint64_t b1 = total_b1_ptr[word_index];
            uint64_t b2 = total_b2_ptr[word_index];
            uint64_t b3 = total_b3_ptr[word_index];
            uint64_t b4 = total_b4_ptr[word_index];

            store_hot_result_from_total_with_center(
                b0, b1, b2, b3, b4,
                adults[fi0], juveniles[fi0], eggs[fi0],
                next_adults_ptr[fi0], next_eggs_ptr[fi0]);
            subtract_horizontal_partial_from_total(
                out0.b0, out0.b1, out0.b2, b0, b1, b2, b3, b4);
            add_horizontal_partial_to_total(
                in0.b0, in0.b1, in0.b2, b0, b1, b2, b3, b4);

            store_hot_result_from_total_with_center(
                b0, b1, b2, b3, b4,
                adults[fi1], juveniles[fi1], eggs[fi1],
                next_adults_ptr[fi1], next_eggs_ptr[fi1]);
            subtract_horizontal_partial_from_total(
                out1.b0, out1.b1, out1.b2, b0, b1, b2, b3, b4);
            add_horizontal_partial_to_total(
                in1.b0, in1.b1, in1.b2, b0, b1, b2, b3, b4);

            total_b0_ptr[word_index] = b0; total_b1_ptr[word_index] = b1;
            total_b2_ptr[word_index] = b2; total_b3_ptr[word_index] = b3;
            total_b4_ptr[word_index] = b4;
            store_horizontal_partial_scalar(recycled_row0, word_index,
                in0.b0, in0.b1, in0.b2);
            store_horizontal_partial_scalar(recycled_row1, word_index,
                in1.b0, in1.b1, in1.b2);
        }

        ring_head = (ring_head + 2) % hot_row_scratch.row_partials.size();
    }

    // ---- Single-row fallback loop -----------------------------------------
    for (; row_index + 1 < hot_row_end; ++row_index) {
        const size_t row_base = static_cast<size_t>(row_index) * words_per_row;

        HorizontalPartialRow& recycled_row = hot_row_scratch.row_partials[ring_head];
        HorizontalPartialPair2* __restrict recycled_pairs  = recycled_row.pairs.data();
        const HorizontalPartialPair2* __restrict out_pairs = recycled_pairs;
        const uint64_t* __restrict incoming_adult_row =
            adults + static_cast<size_t>(row_index + 3) * words_per_row;

        int word_index = hot_word_begin;

        if (use_neon) {
            // Scalar preamble
            {
                const size_t fi = row_base + static_cast<size_t>(word_index);
                const HorizontalPartialWord outgoing =
                    load_horizontal_partial_scalar(recycled_row, word_index);
                const HorizontalPartialWord incoming =
                    compute_horizontal_partial_scalar_from_adult_row(incoming_adult_row, word_index);

                uint64_t b0 = total_b0_ptr[word_index];
                uint64_t b1 = total_b1_ptr[word_index];
                uint64_t b2 = total_b2_ptr[word_index];
                uint64_t b3 = total_b3_ptr[word_index];
                uint64_t b4 = total_b4_ptr[word_index];

                store_hot_result_from_total_with_center(
                    b0, b1, b2, b3, b4,
                    adults[fi], juveniles[fi], eggs[fi],
                    next_adults_ptr[fi], next_eggs_ptr[fi]);
                subtract_horizontal_partial_from_total(
                    outgoing.b0, outgoing.b1, outgoing.b2, b0, b1, b2, b3, b4);
                add_horizontal_partial_to_total(
                    incoming.b0, incoming.b1, incoming.b2, b0, b1, b2, b3, b4);

                total_b0_ptr[word_index] = b0; total_b1_ptr[word_index] = b1;
                total_b2_ptr[word_index] = b2; total_b3_ptr[word_index] = b3;
                total_b4_ptr[word_index] = b4;
                store_horizontal_partial_scalar(recycled_row, word_index,
                    incoming.b0, incoming.b1, incoming.b2);
            }
            ++word_index;

            const int aligned_neon_end =
                word_index + ((aligned_hot_word_end - word_index) / neon_words) * neon_words;

            for (; word_index < aligned_neon_end; word_index += neon_words) {
                const size_t afi = row_base + static_cast<size_t>(word_index);
                const size_t pp_idx = horizontal_partial_pair_index(word_index);

                uint64x2_t out_b0, out_b1, out_b2;
                uint64x2_t in_b0,  in_b1,  in_b2;
                load_horizontal_partial_pair2_u64_neon(
                    out_pairs[pp_idx], out_b0, out_b1, out_b2);
                compute_horizontal_partial_pair2_u64_neon_from_adult_row(
                    incoming_adult_row, word_index, in_b0, in_b1, in_b2);

                uint64x2_t vb0 = vld1q_u64(total_b0_ptr + word_index);
                uint64x2_t vb1 = vld1q_u64(total_b1_ptr + word_index);
                uint64x2_t vb2 = vld1q_u64(total_b2_ptr + word_index);
                uint64x2_t vb3 = vld1q_u64(total_b3_ptr + word_index);
                uint64x2_t vb4 = vld1q_u64(total_b4_ptr + word_index);

                uint64x2_t next_adult, next_egg;
                store_hot_result_from_total_with_center_u64_neon(
                    vb0, vb1, vb2, vb3, vb4,
                    vld1q_u64(adults    + afi),
                    vld1q_u64(juveniles + afi),
                    vld1q_u64(eggs      + afi),
                    next_adult, next_egg);
                vst1q_u64(next_adults_ptr + afi, next_adult);
                vst1q_u64(next_eggs_ptr   + afi, next_egg);

                subtract_horizontal_partial_from_total_u64_neon(
                    out_b0, out_b1, out_b2, vb0, vb1, vb2, vb3, vb4);
                add_horizontal_partial_to_total_u64_neon(
                    in_b0, in_b1, in_b2, vb0, vb1, vb2, vb3, vb4);

                vst1q_u64(total_b0_ptr + word_index, vb0);
                vst1q_u64(total_b1_ptr + word_index, vb1);
                vst1q_u64(total_b2_ptr + word_index, vb2);
                vst1q_u64(total_b3_ptr + word_index, vb3);
                vst1q_u64(total_b4_ptr + word_index, vb4);
                store_horizontal_partial_pair2_u64_neon(
                    recycled_pairs[pp_idx], in_b0, in_b1, in_b2);
            }
        }

        for (; word_index < hot_word_end; ++word_index) {
            const size_t fi = row_base + word_index;
            const HorizontalPartialWord outgoing =
                load_horizontal_partial_scalar(recycled_row, word_index);
            const HorizontalPartialWord incoming =
                compute_horizontal_partial_scalar_from_adult_row(incoming_adult_row, word_index);

            uint64_t b0 = total_b0_ptr[word_index];
            uint64_t b1 = total_b1_ptr[word_index];
            uint64_t b2 = total_b2_ptr[word_index];
            uint64_t b3 = total_b3_ptr[word_index];
            uint64_t b4 = total_b4_ptr[word_index];

            store_hot_result_from_total_with_center(
                b0, b1, b2, b3, b4,
                adults[fi], juveniles[fi], eggs[fi],
                next_adults_ptr[fi], next_eggs_ptr[fi]);
            subtract_horizontal_partial_from_total(
                outgoing.b0, outgoing.b1, outgoing.b2, b0, b1, b2, b3, b4);
            add_horizontal_partial_to_total(
                incoming.b0, incoming.b1, incoming.b2, b0, b1, b2, b3, b4);

            total_b0_ptr[word_index] = b0; total_b1_ptr[word_index] = b1;
            total_b2_ptr[word_index] = b2; total_b3_ptr[word_index] = b3;
            total_b4_ptr[word_index] = b4;
            store_horizontal_partial_scalar(recycled_row, word_index,
                incoming.b0, incoming.b1, incoming.b2);
        }

        ring_head = (ring_head + 1) % hot_row_scratch.row_partials.size();
    }

    // ---- Final row (no rolling update needed) ------------------------------
    const int    final_row      = hot_row_end - 1;
    const size_t final_row_base = static_cast<size_t>(final_row) * words_per_row;
    int word_index = hot_word_begin;

    if (use_neon) {
        // Scalar preamble
        {
            const size_t fi = final_row_base + static_cast<size_t>(word_index);
            store_hot_result_from_total_with_center(
                total_b0_ptr[word_index], total_b1_ptr[word_index],
                total_b2_ptr[word_index], total_b3_ptr[word_index],
                total_b4_ptr[word_index],
                adults[fi], juveniles[fi], eggs[fi],
                next_adults_ptr[fi], next_eggs_ptr[fi]);
        }
        ++word_index;

        const int aligned_neon_end =
            word_index + ((aligned_hot_word_end - word_index) / neon_words) * neon_words;

        for (; word_index < aligned_neon_end; word_index += neon_words) {
            const size_t afi = final_row_base + static_cast<size_t>(word_index);
            uint64x2_t next_adult, next_egg;
            store_hot_result_from_total_with_center_u64_neon(
                vld1q_u64(total_b0_ptr + word_index),
                vld1q_u64(total_b1_ptr + word_index),
                vld1q_u64(total_b2_ptr + word_index),
                vld1q_u64(total_b3_ptr + word_index),
                vld1q_u64(total_b4_ptr + word_index),
                vld1q_u64(adults    + afi),
                vld1q_u64(juveniles + afi),
                vld1q_u64(eggs      + afi),
                next_adult, next_egg);
            vst1q_u64(next_adults_ptr + afi, next_adult);
            vst1q_u64(next_eggs_ptr   + afi, next_egg);
        }
    }

    for (; word_index < hot_word_end; ++word_index) {
        const size_t fi = final_row_base + word_index;
        store_hot_result_from_total_with_center(
            total_b0_ptr[word_index], total_b1_ptr[word_index],
            total_b2_ptr[word_index], total_b3_ptr[word_index],
            total_b4_ptr[word_index],
            adults[fi], juveniles[fi], eggs[fi],
            next_adults_ptr[fi], next_eggs_ptr[fi]);
    }
}

// ---------------------------------------------------------------------------
// Initialise next-gen bitplanes (juveniles → adults, zero eggs)
// ---------------------------------------------------------------------------

static void initialize_next_generation_rows(
    int row_begin, int row_end, int grid_size, int words_per_row,
    const BitPlane& current_juveniles,
    BitPlane& next_adults, BitPlane& next_eggs)
{
    const uint64_t* __restrict juveniles  = current_juveniles.data();
    uint64_t* __restrict next_adults_ptr  = next_adults.data();
    uint64_t* __restrict next_eggs_ptr    = next_eggs.data();
    const int hot_row_begin    = 2;
    const int hot_row_end      = grid_size - 2;
    const int last_word_offset = words_per_row - 1;

    for (int row_index = row_begin; row_index < row_end; ++row_index) {
        const size_t row_base = static_cast<size_t>(row_index) * words_per_row;
        if (row_index < hot_row_begin || row_index >= hot_row_end) {
            for (int wi = 0; wi < words_per_row; ++wi) {
                next_adults_ptr[row_base + wi] = juveniles[row_base + wi];
                next_eggs_ptr  [row_base + wi] = 0ULL;
            }
        } else {
            const size_t lw = row_base;
            const size_t rw = row_base + last_word_offset;
            next_adults_ptr[lw] = juveniles[lw]; next_eggs_ptr[lw] = 0ULL;
            next_adults_ptr[rw] = juveniles[rw]; next_eggs_ptr[rw] = 0ULL;
        }
    }
}

// ---------------------------------------------------------------------------
// Edge-column handlers (unchanged from bic_all — scalar-only paths)
// ---------------------------------------------------------------------------

__attribute__((noinline)) static void run_one_generation_left_edge_valid_rows(
    int row_begin, int row_end, int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults, BitPlane& next_eggs)
{
    const uint64_t* __restrict adults      = current_adults.data();
    const uint64_t* __restrict juveniles   = current_juveniles.data();
    const uint64_t* __restrict eggs        = current_eggs.data();
    uint64_t* __restrict next_adults_ptr   = next_adults.data();
    uint64_t* __restrict next_eggs_ptr     = next_eggs.data();
    row_begin = std::max(row_begin, 2);
    row_end   = std::min(row_end,   grid_size - 2);
    const uint64_t valid_mask = ~0ULL << 2;

    for (int row_index = row_begin; row_index < row_end; ++row_index) {
        const size_t rm2 = static_cast<size_t>(row_index - 2) * words_per_row;
        const size_t rm1 = static_cast<size_t>(row_index - 1) * words_per_row;
        const size_t r0  = static_cast<size_t>(row_index)     * words_per_row;
        const size_t rp1 = static_cast<size_t>(row_index + 1) * words_per_row;
        const size_t rp2 = static_cast<size_t>(row_index + 2) * words_per_row;

        const uint64_t top_left_2  = adults[rm2] << 2;
        const uint64_t top_left_1  = adults[rm2] << 1;
        const uint64_t top_center  = adults[rm2];
        const uint64_t top_right_1 = (adults[rm2] >> 1) | (adults[rm2 + 1] << 63);
        const uint64_t top_right_2 = (adults[rm2] >> 2) | (adults[rm2 + 1] << 62);

        const uint64_t upper_left_2  = adults[rm1] << 2;
        const uint64_t upper_left_1  = adults[rm1] << 1;
        const uint64_t upper_center  = adults[rm1];
        const uint64_t upper_right_1 = (adults[rm1] >> 1) | (adults[rm1 + 1] << 63);
        const uint64_t upper_right_2 = (adults[rm1] >> 2) | (adults[rm1 + 1] << 62);

        const uint64_t center_left_2  = adults[r0] << 2;
        const uint64_t center_left_1  = adults[r0] << 1;
        const uint64_t center_right_1 = (adults[r0] >> 1) | (adults[r0 + 1] << 63);
        const uint64_t center_right_2 = (adults[r0] >> 2) | (adults[r0 + 1] << 62);

        const uint64_t lower_left_2  = adults[rp1] << 2;
        const uint64_t lower_left_1  = adults[rp1] << 1;
        const uint64_t lower_center  = adults[rp1];
        const uint64_t lower_right_1 = (adults[rp1] >> 1) | (adults[rp1 + 1] << 63);
        const uint64_t lower_right_2 = (adults[rp1] >> 2) | (adults[rp1 + 1] << 62);

        const uint64_t bottom_left_2  = adults[rp2] << 2;
        const uint64_t bottom_left_1  = adults[rp2] << 1;
        const uint64_t bottom_center  = adults[rp2];
        const uint64_t bottom_right_1 = (adults[rp2] >> 1) | (adults[rp2 + 1] << 63);
        const uint64_t bottom_right_2 = (adults[rp2] >> 2) | (adults[rp2 + 1] << 62);

        uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0;
        auto add_mask = [&](uint64_t m) {
            uint64_t carry = b0 & m; b0 ^= m; m = carry;
            carry = b1 & m; b1 ^= m; m = carry;
            carry = b2 & m; b2 ^= m; m = carry;
            carry = b3 & m; b3 ^= m; m = carry;
            b4 ^= m;
        };

        add_mask(top_left_2);   add_mask(top_left_1);   add_mask(top_center);
        add_mask(top_right_1);  add_mask(top_right_2);
        add_mask(upper_left_2); add_mask(upper_left_1); add_mask(upper_center);
        add_mask(upper_right_1);add_mask(upper_right_2);
        add_mask(center_left_2);add_mask(center_left_1);
        add_mask(center_right_1);add_mask(center_right_2);
        add_mask(lower_left_2); add_mask(lower_left_1); add_mask(lower_center);
        add_mask(lower_right_1);add_mask(lower_right_2);
        add_mask(bottom_left_2);add_mask(bottom_left_1);add_mask(bottom_center);
        add_mask(bottom_right_1);add_mask(bottom_right_2);

        const uint64_t adult_word   = adults[r0];
        const uint64_t blocked_word = juveniles[r0] | eggs[r0];
        const uint64_t adult_valid  = adult_word & valid_mask;
        const uint64_t empty_valid  = ~(adult_word | blocked_word) & valid_mask;
        const uint64_t ge3  = b4 | b3 | b2 | (b1 & b0);
        const uint64_t ge4  = b4 | b3 | b2;
        const uint64_t ge6  = b4 | b3 | (b2 & b1);
        const uint64_t ge10 = b4 | (b3 & (b2 | b1));

        next_adults_ptr[r0] |= adult_valid & ge4 & ~ge10;
        next_eggs_ptr  [r0] |= empty_valid & ge3 & ~ge6;
    }
}

__attribute__((noinline)) static void run_one_generation_right_edge_valid_rows(
    int row_begin, int row_end, int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults, BitPlane& next_eggs)
{
    const uint64_t* __restrict adults      = current_adults.data();
    const uint64_t* __restrict juveniles   = current_juveniles.data();
    const uint64_t* __restrict eggs        = current_eggs.data();
    uint64_t* __restrict next_adults_ptr   = next_adults.data();
    uint64_t* __restrict next_eggs_ptr     = next_eggs.data();
    row_begin = std::max(row_begin, 2);
    row_end   = std::min(row_end,   grid_size - 2);
    const int last_word_offset     = words_per_row - 1;
    const int previous_word_offset = words_per_row - 2;
    const uint64_t valid_mask = (1ULL << 62) - 1ULL;

    for (int row_index = row_begin; row_index < row_end; ++row_index) {
        const size_t rm2 = static_cast<size_t>(row_index - 2) * words_per_row;
        const size_t rm1 = static_cast<size_t>(row_index - 1) * words_per_row;
        const size_t r0  = static_cast<size_t>(row_index)     * words_per_row;
        const size_t rp1 = static_cast<size_t>(row_index + 1) * words_per_row;
        const size_t rp2 = static_cast<size_t>(row_index + 2) * words_per_row;

        const uint64_t top_left_2  = (adults[rm2 + last_word_offset] << 2) | (adults[rm2 + previous_word_offset] >> 62);
        const uint64_t top_left_1  = (adults[rm2 + last_word_offset] << 1) | (adults[rm2 + previous_word_offset] >> 63);
        const uint64_t top_center  =  adults[rm2 + last_word_offset];
        const uint64_t top_right_1 =  adults[rm2 + last_word_offset] >> 1;
        const uint64_t top_right_2 =  adults[rm2 + last_word_offset] >> 2;

        const uint64_t upper_left_2  = (adults[rm1 + last_word_offset] << 2) | (adults[rm1 + previous_word_offset] >> 62);
        const uint64_t upper_left_1  = (adults[rm1 + last_word_offset] << 1) | (adults[rm1 + previous_word_offset] >> 63);
        const uint64_t upper_center  =  adults[rm1 + last_word_offset];
        const uint64_t upper_right_1 =  adults[rm1 + last_word_offset] >> 1;
        const uint64_t upper_right_2 =  adults[rm1 + last_word_offset] >> 2;

        const size_t   edge_wi      = r0 + last_word_offset;
        const uint64_t center_left_2  = (adults[edge_wi] << 2) | (adults[r0 + previous_word_offset] >> 62);
        const uint64_t center_left_1  = (adults[edge_wi] << 1) | (adults[r0 + previous_word_offset] >> 63);
        const uint64_t center_right_1 =  adults[edge_wi] >> 1;
        const uint64_t center_right_2 =  adults[edge_wi] >> 2;

        const uint64_t lower_left_2  = (adults[rp1 + last_word_offset] << 2) | (adults[rp1 + previous_word_offset] >> 62);
        const uint64_t lower_left_1  = (adults[rp1 + last_word_offset] << 1) | (adults[rp1 + previous_word_offset] >> 63);
        const uint64_t lower_center  =  adults[rp1 + last_word_offset];
        const uint64_t lower_right_1 =  adults[rp1 + last_word_offset] >> 1;
        const uint64_t lower_right_2 =  adults[rp1 + last_word_offset] >> 2;

        const uint64_t bottom_left_2  = (adults[rp2 + last_word_offset] << 2) | (adults[rp2 + previous_word_offset] >> 62);
        const uint64_t bottom_left_1  = (adults[rp2 + last_word_offset] << 1) | (adults[rp2 + previous_word_offset] >> 63);
        const uint64_t bottom_center  =  adults[rp2 + last_word_offset];
        const uint64_t bottom_right_1 =  adults[rp2 + last_word_offset] >> 1;
        const uint64_t bottom_right_2 =  adults[rp2 + last_word_offset] >> 2;

        uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0;
        auto add_mask = [&](uint64_t m) {
            uint64_t carry = b0 & m; b0 ^= m; m = carry;
            carry = b1 & m; b1 ^= m; m = carry;
            carry = b2 & m; b2 ^= m; m = carry;
            carry = b3 & m; b3 ^= m; m = carry;
            b4 ^= m;
        };

        add_mask(top_left_2);   add_mask(top_left_1);   add_mask(top_center);
        add_mask(top_right_1);  add_mask(top_right_2);
        add_mask(upper_left_2); add_mask(upper_left_1); add_mask(upper_center);
        add_mask(upper_right_1);add_mask(upper_right_2);
        add_mask(center_left_2);add_mask(center_left_1);
        add_mask(center_right_1);add_mask(center_right_2);
        add_mask(lower_left_2); add_mask(lower_left_1); add_mask(lower_center);
        add_mask(lower_right_1);add_mask(lower_right_2);
        add_mask(bottom_left_2);add_mask(bottom_left_1);add_mask(bottom_center);
        add_mask(bottom_right_1);add_mask(bottom_right_2);

        const uint64_t adult_word   = adults[edge_wi];
        const uint64_t blocked_word = juveniles[edge_wi] | eggs[edge_wi];
        const uint64_t adult_valid  = adult_word & valid_mask;
        const uint64_t empty_valid  = ~(adult_word | blocked_word) & valid_mask;
        const uint64_t ge3  = b4 | b3 | b2 | (b1 & b0);
        const uint64_t ge4  = b4 | b3 | b2;
        const uint64_t ge6  = b4 | b3 | (b2 & b1);
        const uint64_t ge10 = b4 | (b3 & (b2 | b1));

        next_adults_ptr[edge_wi] |= adult_valid & ge4 & ~ge10;
        next_eggs_ptr  [edge_wi] |= empty_valid & ge3 & ~ge6;
    }
}

__attribute__((noinline)) static void run_one_generation_wrapping_rows(
    int row_begin, int row_end, int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults, BitPlane& next_eggs)
{
    const uint64_t* __restrict adults      = current_adults.data();
    const uint64_t* __restrict juveniles   = current_juveniles.data();
    const uint64_t* __restrict eggs        = current_eggs.data();
    uint64_t* __restrict next_adults_ptr   = next_adults.data();
    uint64_t* __restrict next_eggs_ptr     = next_eggs.data();
    const int top_wrap_row_end       = 2;
    const int bottom_wrap_row_begin  = grid_size - 2;

    auto process_cell = [&](int row_index, int col_index) {
        const size_t fwi = static_cast<size_t>(row_index) * words_per_row + (col_index >> 6);
        const uint64_t cell_mask = 1ULL << (col_index & 63);
        if (((juveniles[fwi] | eggs[fwi]) & cell_mask) != 0) return;
        const int anc = count_adult_neighbors_naive(
            current_adults, grid_size, words_per_row, row_index, col_index);
        if ((adults[fwi] & cell_mask) != 0) {
            if (anc >= 4 && anc <= 9) next_adults_ptr[fwi] |= cell_mask;
        } else if (anc >= 3 && anc <= 5) {
            next_eggs_ptr[fwi] |= cell_mask;
        }
    };

    for (int row_index = row_begin;
         row_index < row_end && row_index < top_wrap_row_end; ++row_index) {
        for (int col_index = 0; col_index < grid_size; ++col_index)
            process_cell(row_index, col_index);
    }

    const int interior_begin = std::max(row_begin, 2);
    const int interior_end   = std::min(row_end, grid_size - 2);
    for (int row_index = interior_begin; row_index < interior_end; ++row_index) {
        for (int col_index = 0; col_index < 2; ++col_index)
            process_cell(row_index, col_index);
        for (int col_index = grid_size - 2; col_index < grid_size; ++col_index)
            process_cell(row_index, col_index);
    }

    for (int row_index = std::max(row_begin, bottom_wrap_row_begin);
         row_index < row_end; ++row_index) {
        for (int col_index = 0; col_index < grid_size; ++col_index)
            process_cell(row_index, col_index);
    }
}

__attribute__((noinline)) static void run_one_generation_edge_rows(
    int row_begin, int row_end, int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults, BitPlane& next_eggs)
{
    run_one_generation_left_edge_valid_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs, next_adults, next_eggs);
    run_one_generation_right_edge_valid_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs, next_adults, next_eggs);
    run_one_generation_wrapping_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs, next_adults, next_eggs);
}

// ---------------------------------------------------------------------------
// Per-chunk orchestration
// ---------------------------------------------------------------------------

static void process_generation_chunk(
    int row_begin, int row_end, int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_adults, BitPlane& next_eggs,
    HotRowScratch& hot_row_scratch)
{
    initialize_next_generation_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_juveniles, next_adults, next_eggs);
    run_one_generation_hot_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs,
        next_adults, next_eggs, hot_row_scratch);
    run_one_generation_edge_rows(
        row_begin, row_end, grid_size, words_per_row,
        current_adults, current_juveniles, current_eggs,
        next_adults, next_eggs);
}

// ---------------------------------------------------------------------------
// Output unpacking
// ---------------------------------------------------------------------------

static void unpack_bit_planes_to_output_grid(
    int grid_size, int words_per_row,
    const BitPlane& current_adults,
    const BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    std::vector<uint8_t>& output_grid)
{
    output_grid.assign(static_cast<size_t>(grid_size) * grid_size, EMPTY);
    for (int row_index = 0; row_index < grid_size; ++row_index) {
        for (int col_index = 0; col_index < grid_size; ++col_index) {
            uint8_t cell_state = EMPTY;
            if      (is_cell_set(current_adults,    words_per_row, row_index, col_index)) cell_state = ADULT;
            else if (is_cell_set(current_juveniles, words_per_row, row_index, col_index)) cell_state = JUVENILE;
            else if (is_cell_set(current_eggs,      words_per_row, row_index, col_index)) cell_state = EGG;
            output_grid[static_cast<size_t>(row_index) * grid_size + col_index] = cell_state;
        }
    }
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
        long parsed = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || parsed <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        generations = static_cast<int>(parsed);
    }

    FILE* input_file = std::fopen(argv[1], "rb");
    if (!input_file) {
        std::fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]);
        return 2;
    }

    uint64_t width = 0, height = 0;
    if (std::fread(&width,  sizeof(uint64_t), 1, input_file) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, input_file) != 1) {
        std::fprintf(stderr, "Error: input file too short (cannot read header)\n");
        std::fclose(input_file);
        return 3;
    }
    if (width == 0 || width != height) {
        std::fprintf(stderr,
            "Error: grid must be square and non-empty, got %" PRIu64 " × %" PRIu64 "\n",
            width, height);
        std::fclose(input_file);
        return 3;
    }

    int    grid_size  = static_cast<int>(width);
    size_t cell_count = width * width;

    std::vector<uint8_t> input_grid(cell_count);
    if (std::fread(input_grid.data(), 1, cell_count, input_file) != cell_count) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(input_file);
        return 4;
    }
    std::fclose(input_file);

    int    words_per_row = grid_size >> 6;
    size_t total_words   = static_cast<size_t>(grid_size) * words_per_row;

    alignas(64) BitPlane current_adults(total_words,    0ULL);
    alignas(64) BitPlane current_juveniles(total_words, 0ULL);
    alignas(64) BitPlane current_eggs(total_words,      0ULL);
    alignas(64) BitPlane next_adults(total_words,       0ULL);
    alignas(64) BitPlane next_eggs(total_words,         0ULL);

    pack_input_into_bit_planes(input_grid, grid_size, words_per_row,
                               current_adults, current_juveniles, current_eggs);

    auto start_time = std::chrono::steady_clock::now();

    const int thread_count = std::max(1, std::min(TARGET_THREAD_COUNT, grid_size));
    std::vector<RowRange> row_ranges(static_cast<size_t>(thread_count));
    for (int worker_id = 0; worker_id < thread_count; ++worker_id)
        row_ranges[static_cast<size_t>(worker_id)] =
            compute_row_range(grid_size, thread_count, worker_id);

    std::barrier generation_start_barrier(thread_count);
    std::barrier generation_done_barrier(thread_count);
    std::atomic<bool> stop_workers{false};

    auto worker_body = [&](int worker_id) {
        pin_current_thread_to_cpu(worker_id);
        const RowRange owned_rows = row_ranges[static_cast<size_t>(worker_id)];
        HotRowScratch hot_row_scratch(words_per_row);

        for (;;) {
            generation_start_barrier.arrive_and_wait();
            if (stop_workers.load(std::memory_order_acquire)) break;

            process_generation_chunk(
                owned_rows.begin, owned_rows.end, grid_size, words_per_row,
                current_adults, current_juveniles, current_eggs,
                next_adults, next_eggs, hot_row_scratch);

            generation_done_barrier.arrive_and_wait();
        }
    };

    HotRowScratch main_hot_row_scratch(words_per_row);
    pin_current_thread_to_cpu(0);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(std::max(0, thread_count - 1)));
    for (int worker_id = 1; worker_id < thread_count; ++worker_id)
        workers.emplace_back(worker_body, worker_id);

    const RowRange main_owned_rows = row_ranges.front();
    for (int generation = 0; generation < generations; ++generation) {
        generation_start_barrier.arrive_and_wait();

        process_generation_chunk(
            main_owned_rows.begin, main_owned_rows.end, grid_size, words_per_row,
            current_adults, current_juveniles, current_eggs,
            next_adults, next_eggs, main_hot_row_scratch);

        generation_done_barrier.arrive_and_wait();

        current_adults.swap(next_adults);
        current_juveniles.swap(current_eggs);
        current_eggs.swap(next_eggs);
    }

    stop_workers.store(true, std::memory_order_release);
    generation_start_barrier.arrive_and_wait();
    for (std::thread& worker : workers) worker.join();

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::printf("%.3f ms\n", elapsed_ms);

    std::vector<uint8_t> output_grid;
    unpack_bit_planes_to_output_grid(grid_size, words_per_row,
                                     current_adults, current_juveniles, current_eggs,
                                     output_grid);

    FILE* output_file = std::fopen(argv[2], "wb");
    if (!output_file) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }
    if (std::fwrite(&width,  sizeof(uint64_t), 1, output_file) != 1 ||
        std::fwrite(&height, sizeof(uint64_t), 1, output_file) != 1 ||
        std::fwrite(output_grid.data(), 1, cell_count, output_file) != cell_count) {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(output_file);
        return 6;
    }
    std::fclose(output_file);
    return 0;
}
