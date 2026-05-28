// Refactored copy of the current exact-two-row pair2 kernel.
//
// The algorithm is unchanged:
//   1. build a 5-row ring of horizontal adult-neighbour partials
//   2. seed exact bit-sliced neighbour counts for the first hot row
//   3. sweep the hot interior two rows at a time:
//        emit row r
//        counts = counts - outgoing(r-2) + incoming(r+3)
//        emit row r+1
//        counts = counts - outgoing(r-1) + incoming(r+4)
//
// This version keeps the current performance-oriented layout and schedule, but
// names the intermediate representations explicitly and hides the vector/adder
// plumbing behind zero-cost helpers so the intent is easier to follow.
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

#include <arm_neon.h>
#include <arm_sve.h>

static constexpr uint8_t EMPTY = 0;
static constexpr uint8_t EGG = 1;
static constexpr uint8_t JUVENILE = 2;
static constexpr uint8_t ADULT = 3;

#if defined(__GNUC__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

// std::vector allocator that keeps the hot bit-plane and partial buffers
// aligned for the target load/store patterns.
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept
    {
    }

    // Allocates Alignment-aligned storage for count objects of type T.
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

    // Releases storage previously returned by allocate().
    void deallocate(T* pointer, std::size_t) noexcept
    {
        std::free(pointer);
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

// All AlignedAllocator instances with the same alignment are interchangeable.
template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&)
{
    return true;
}

// Standard inequality companion for allocator interoperability.
template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&)
{
    return false;
}

using BitPlane = std::vector<uint64_t, AlignedAllocator<uint64_t, 64>>;

// Half-open row interval `[begin, end)` owned by one worker thread.
struct RowRange {
    int begin = 0;
    int end = 0;
};

#ifndef TARGET_THREAD_COUNT_OVERRIDE
#define TARGET_THREAD_COUNT_OVERRIDE 8
#endif

static constexpr int TARGET_THREAD_COUNT = TARGET_THREAD_COUNT_OVERRIDE;

// ---------------------------------------------------------------------------
// Horizontal partial storage and indexing
// ---------------------------------------------------------------------------

struct HorizontalPartialWord {
    // Three bit planes encoding the per-bit sum of the five horizontally
    // adjacent adult masks `{col-2, col-1, col, col+1, col+2}` for one
    // 64-cell hot word. For any bit lane, `(b2 b1 b0)` stores a value in [0, 5].
    uint64_t b0 = 0ULL;
    uint64_t b1 = 0ULL;
    uint64_t b2 = 0ULL;
};

// Vector horizontal partials use the same 3-plane encoding as
// HorizontalPartialWord, but in the SVE path the three planes live as local
// register triples `(partial_b0, partial_b1, partial_b2)` rather than a struct
// because ACLE sizeless register types cannot be stored as data members.

struct alignas(16) HorizontalPartialPair2 {
    // Packed storage for the horizontal partials of two adjacent hot words.
    // Lane 0 stores word `2k`, lane 1 stores word `2k + 1`.
    // The planes stay split as `[b0[2], b1[2], b2[2]]` so both the scalar path
    // and the SVE path can address plane data directly without unpacking.
    uint64_t b0[2] = {0ULL, 0ULL};
    uint64_t b1[2] = {0ULL, 0ULL};
    uint64_t b2[2] = {0ULL, 0ULL};
};

static_assert(sizeof(HorizontalPartialPair2) == sizeof(uint64_t) * 6);
static_assert(alignof(HorizontalPartialPair2) == 16);

using HorizontalPartialPair2Buffer =
    std::vector<HorizontalPartialPair2, AlignedAllocator<HorizontalPartialPair2, 64>>;

inline size_t packed_horizontal_partial_pair_count(int words_per_row)
{
    // Each tile stores two word positions, so this is ceil(words_per_row / 2).
    // This is the total #of horizontal partial pairs in a row
    return static_cast<size_t>(words_per_row + 1) >> 1;
}

struct HorizontalPartialRow {
    // Cached horizontal-partial representation for one source grid row.
    // The hot-row pipeline keeps five of these rows in a ring so it can slide
    // the 5-row vertical window by recycling one row at a time.
    HorizontalPartialPair2Buffer packed_partial_pairs;

    HorizontalPartialRow() = default;

    // Allocates enough packed two-word partial tiles to cache one full source row.
    explicit HorizontalPartialRow(int words_per_row)
        : packed_partial_pairs(packed_horizontal_partial_pair_count(words_per_row))
    {
    }
};

struct ScalarBitSlicedCounts {
    // Five bit planes encoding the exact 5x5 adult-neighbour count for one
    // 64-cell hot word. For any bit lane, `(b4 b3 b2 b1 b0)` stores a value in
    // [0, 24], which is the full neighbour count seen by that cell.
    uint64_t b0 = 0ULL;
    uint64_t b1 = 0ULL;
    uint64_t b2 = 0ULL;
    uint64_t b3 = 0ULL;
    uint64_t b4 = 0ULL;
};

// Vector exact-count totals use the same 5-plane encoding as
// ScalarBitSlicedCounts, but in the SVE path the planes travel as local
// register quintuples `(count_b0 ... count_b4)` for the same ACLE reason.

struct alignas(16) VerticalAccumulatorPair2 {
    // Packed storage for the rolling exact counts of two adjacent hot words.
    // Lane 0 stores word `2k`, lane 1 stores word `2k + 1`.
    // The planes stay split as `[b0[2], ..., b4[2]]` so the scalar path can
    // still address one word at a time while the pair2 SVE path loads/stores
    // one full tile with five naturally aligned vector accesses.
    uint64_t b0[2] = {0ULL, 0ULL};
    uint64_t b1[2] = {0ULL, 0ULL};
    uint64_t b2[2] = {0ULL, 0ULL};
    uint64_t b3[2] = {0ULL, 0ULL};
    uint64_t b4[2] = {0ULL, 0ULL};
};

static_assert(sizeof(VerticalAccumulatorPair2) == sizeof(uint64_t) * 10);
static_assert(alignof(VerticalAccumulatorPair2) == 16);

using VerticalAccumulatorPair2Buffer =
    std::vector<VerticalAccumulatorPair2, AlignedAllocator<VerticalAccumulatorPair2, 64>>;

struct HotRowKernelContext {
    // Shared view of the current bit planes and scratch planes used by the
    // hot-row kernel. This bundles the raw arrays that every backend operation
    // needs so the pipeline code can talk in terms of counts/partials/state
    // transitions instead of threading ten unrelated pointers everywhere.
    const uint64_t* adults = nullptr;
    uint64_t* juveniles = nullptr;
    const uint64_t* eggs = nullptr;
    uint64_t* next_eggs = nullptr;
    VerticalAccumulatorPair2* total_pairs = nullptr;
    int hot_word_begin = 0;
    int hot_word_end = 0;
    int aligned_hot_word_end = 0;
    int sve_words = 0;
};

// Maps a scalar word index onto the packed two-word tile that owns it.
inline size_t horizontal_partial_pair_index(int word_index)
{
    return static_cast<size_t>(word_index >> 1);
}

// Selects lane 0 or 1 inside a packed two-word tile.
inline int horizontal_partial_pair_lane(int word_index)
{
    return word_index & 1;
}

inline size_t vertical_accumulator_pair_index(int word_index)
{
    return static_cast<size_t>(word_index >> 1);
}

inline int vertical_accumulator_pair_lane(int word_index)
{
    return word_index & 1;
}

inline size_t packed_vertical_accumulator_pair_count(int words_per_row)
{
    return static_cast<size_t>(words_per_row + 1) >> 1;
}

// Scalar implementation of the hot-word operations.
// One instance is stateless and simply names the logical steps for a single
// 64-bit hot word: load/store cached partials, maintain exact counts, and emit
// the next-state masks.
class ScalarHotWordBackend {
public:
    using Partial = HorizontalPartialWord;
    using Counts = ScalarBitSlicedCounts;

    // Loads the cached three-plane horizontal partial for `word_index` from the
    // packed two-word row storage and returns it as a scalar hot-word record.
    ALWAYS_INLINE Partial load_partial(
        const HorizontalPartialRow& row_partial,
        int word_index) const
    {
        const HorizontalPartialPair2& pair =
            row_partial.packed_partial_pairs[horizontal_partial_pair_index(word_index)];
        const int lane = horizontal_partial_pair_lane(word_index);
        return Partial{pair.b0[lane], pair.b1[lane], pair.b2[lane]};
    }

    // Stores one scalar hot-word partial back into the packed two-word row
    // cache slot that owns `word_index`.
    ALWAYS_INLINE void store_partial(
        HorizontalPartialRow& row_partial,
        int word_index,
        const Partial& partial) const
    {
        HorizontalPartialPair2& pair =
            row_partial.packed_partial_pairs[horizontal_partial_pair_index(word_index)];
        const int lane = horizontal_partial_pair_lane(word_index);
        pair.b0[lane] = partial.b0;
        pair.b1[lane] = partial.b1;
        pair.b2[lane] = partial.b2;
    }

    // Computes the 3-plane horizontal partial for one hot word by summing the
    // five horizontally adjacent adult masks `{col-2, col-1, col, col+1, col+2}`.
    ALWAYS_INLINE Partial compute_partial(
        const uint64_t* __restrict adult_row,
        int word_index) const
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
        return Partial{sum_2, carry_1 ^ carry_2, carry_1 & carry_2};
    }

    // Loads the current exact 5-plane adult-neighbour total for one hot word
    // from the rolling vertical accumulator arrays.
    ALWAYS_INLINE Counts load_counts(
        const HotRowKernelContext& context,
        int word_index) const
    {
        const VerticalAccumulatorPair2& pair =
            context.total_pairs[vertical_accumulator_pair_index(word_index)];
        const int lane = vertical_accumulator_pair_lane(word_index);
        return Counts{
            pair.b0[lane],
            pair.b1[lane],
            pair.b2[lane],
            pair.b3[lane],
            pair.b4[lane],
        };
    }

    // Writes an updated exact 5-plane adult-neighbour total back into the
    // rolling vertical accumulator arrays for one hot word.
    ALWAYS_INLINE void store_counts(
        const HotRowKernelContext& context,
        int word_index,
        const Counts& counts) const
    {
        VerticalAccumulatorPair2& pair =
            context.total_pairs[vertical_accumulator_pair_index(word_index)];
        const int lane = vertical_accumulator_pair_lane(word_index);
        pair.b0[lane] = counts.b0;
        pair.b1[lane] = counts.b1;
        pair.b2[lane] = counts.b2;
        pair.b3[lane] = counts.b3;
        pair.b4[lane] = counts.b4;
    }

    // Applies the transition rules to the exact count total of one hot word
    // and writes the resulting adult and egg masks into the next-state planes.
    ALWAYS_INLINE void emit_next_state(
        const HotRowKernelContext& context,
        size_t flat_word_index,
        const Counts& counts) const
    {
        const uint64_t adult_word = context.adults[flat_word_index];
        const uint64_t juvenile_word = context.juveniles[flat_word_index];
        const uint64_t egg_word = context.eggs[flat_word_index];
        write_next_state_from_counts(
            counts,
            adult_word,
            juvenile_word,
            egg_word,
            context.juveniles[flat_word_index],
            context.next_eggs[flat_word_index]);
    }

    // Advances the 5-row vertical window for one hot word by removing the
    // outgoing row partial and adding the newly incoming row partial.
    ALWAYS_INLINE void slide_window(
        Counts& counts,
        const Partial& outgoing_partial,
        const Partial& incoming_partial) const
    {
        subtract_partial_from_total(outgoing_partial, counts);
        add_partial_to_total(incoming_partial, counts);
    }

    // Adds one fully assembled horizontal partial into an exact-count total.
    ALWAYS_INLINE void add_partial(Counts& counts, const Partial& partial) const
    {
        add_partial_to_total(partial, counts);
    }

    // Adds one raw bit mask into the bit-sliced counter representation. Each
    // set bit in `mask` increments the corresponding per-cell count by one.
    ALWAYS_INLINE void add_mask(Counts& counts, uint64_t mask) const
    {
        uint64_t carry = counts.b0 & mask;
        counts.b0 ^= mask;
        mask = carry;

        carry = counts.b1 & mask;
        counts.b1 ^= mask;
        mask = carry;

        carry = counts.b2 & mask;
        counts.b2 ^= mask;
        mask = carry;

        carry = counts.b3 & mask;
        counts.b3 ^= mask;
        mask = carry;

        counts.b4 ^= mask;
    }

private:
    // Computes one carry-save adder stage for three equally weighted bit
    // masks, producing the sum bits and carry bits without carry propagation.
    static ALWAYS_INLINE void carry_save_add(
        uint64_t a,
        uint64_t b,
        uint64_t c,
        uint64_t& sum,
        uint64_t& carry)
    {
        sum = a ^ b ^ c;
        carry = (a & b) | (a & c) | (b & c);
    }

    // Adds one three-plane horizontal partial into the five-plane exact-count
    // total for a hot word.
    static ALWAYS_INLINE void add_partial_to_total(
        const Partial& partial,
        Counts& counts)
    {
        uint64_t carry = counts.b0 & partial.b0;
        counts.b0 ^= partial.b0;

        carry_save_add(counts.b1, partial.b1, carry, counts.b1, carry);
        carry_save_add(counts.b2, partial.b2, carry, counts.b2, carry);

        const uint64_t carry4 = counts.b3 & carry;
        counts.b3 ^= carry;
        counts.b4 ^= carry4;
    }

    // Removes one three-plane horizontal partial from the five-plane exact
    // total when a source row leaves the sliding 5-row vertical window.
    static ALWAYS_INLINE void subtract_partial_from_total(
        const Partial& partial,
        Counts& counts)
    {
        uint64_t borrow = (~counts.b0) & partial.b0;
        counts.b0 ^= partial.b0;

        uint64_t next_borrow =
            ((~counts.b1) & (partial.b1 | borrow)) | (partial.b1 & borrow);
        counts.b1 ^= partial.b1 ^ borrow;
        borrow = next_borrow;

        next_borrow =
            ((~counts.b2) & (partial.b2 | borrow)) | (partial.b2 & borrow);
        counts.b2 ^= partial.b2 ^ borrow;
        borrow = next_borrow;

        next_borrow = (~counts.b3) & borrow;
        counts.b3 ^= borrow;
        counts.b4 ^= next_borrow;
    }

    // Converts one exact five-plane adult-neighbour total plus the current
    // adult/juvenile/egg occupancy masks into the next adult and egg masks.
    static ALWAYS_INLINE void write_next_state_from_counts(
        const Counts& counts,
        uint64_t adult_word,
        uint64_t juvenile_word,
        uint64_t egg_word,
        uint64_t& next_adult_word,
        uint64_t& next_egg_word)
    {
        const uint64_t blocked_word = juvenile_word | egg_word;
        const uint64_t not_b4 = ~counts.b4;
        const uint64_t not_b3 = ~counts.b3;
        const uint64_t not_b2 = ~counts.b2;
        const uint64_t not_b1 = ~counts.b1;
        const uint64_t b1_or_b0 = counts.b1 | counts.b0;
        const uint64_t b1_and_b0 = counts.b1 & counts.b0;

        const uint64_t adult_5_to_10 =
            not_b4 &
            ((not_b3 & counts.b2 & b1_or_b0) |
             (counts.b3 & not_b2 & ~b1_and_b0));
        const uint64_t egg_3_to_5 =
            not_b4 &
            not_b3 &
            ((not_b2 & b1_and_b0) |
             (counts.b2 & not_b1));

        next_adult_word = (adult_word & adult_5_to_10) | juvenile_word;
        next_egg_word = ~(adult_word | blocked_word) & egg_3_to_5;
    }
};

// SVE implementation of the same hot-word operations, but executed on one
// packed two-word tile at a time. The public methods name the algorithmic
// steps; the private helpers are the only place where raw SVE intrinsics are
// allowed to appear.
class Pair2SVEHotWordBackend {
public:
    // Loads the cached horizontal partial for one packed two-word tile into
    // three SVE registers, one register per bit plane.
    ALWAYS_INLINE void load_partial(
        const HorizontalPartialPair2& pair,
        svuint64_t& partial_b0,
        svuint64_t& partial_b1,
        svuint64_t& partial_b2) const
    {
        partial_b0 = load_word(pair.b0);
        partial_b1 = load_word(pair.b1);
        partial_b2 = load_word(pair.b2);
    }

    // Stores a newly computed horizontal partial back into one packed two-word
    // cache tile.
    ALWAYS_INLINE void store_partial(
        HorizontalPartialPair2& pair,
        svuint64_t partial_b0,
        svuint64_t partial_b1,
        svuint64_t partial_b2) const
    {
        store_word(pair.b0, partial_b0);
        store_word(pair.b1, partial_b1);
        store_word(pair.b2, partial_b2);
    }

    // Computes the three-plane horizontal partial for one packed two-word tile
    // by summing the five horizontally adjacent adult masks for both words at once.
    ALWAYS_INLINE void compute_partial(
        const uint64_t* __restrict adult_row,
        int word_index,
        svuint64_t& partial_b0,
        svuint64_t& partial_b1,
        svuint64_t& partial_b2) const
    {
        const svuint64_t prev = load_word(adult_row + word_index - 1);
        const svuint64_t curr = load_word(adult_row + word_index);
        const svuint64_t next = load_word(adult_row + word_index + 1);

        const svuint64_t left_2 = shift_left_with_incoming<2>(curr, prev);
        const svuint64_t left_1 = shift_left_with_incoming<1>(curr, prev);
        const svuint64_t right_1 = shift_right_with_incoming<1>(curr, next);
        const svuint64_t right_2 = shift_right_with_incoming<2>(curr, next);

        svuint64_t sum_1;
        svuint64_t carry_1;
        svuint64_t sum_2;
        svuint64_t carry_2;

        carry_save_add(left_2, left_1, curr, sum_1, carry_1);
        carry_save_add(right_1, right_2, sum_1, sum_2, carry_2);
        partial_b0 = sum_2;
        partial_b1 = xor_words(carry_1, carry_2);
        partial_b2 = and_words(carry_1, carry_2);
    }

    // Pointer-stepped variant of compute_partial used by the aligned pair2
    // sweep so the hot vector loop can advance raw pointers instead of
    // rebuilding base+index addresses each iteration.
    ALWAYS_INLINE void compute_partial_from_word_ptr(
        const uint64_t* __restrict adult_word_ptr,
        svuint64_t& partial_b0,
        svuint64_t& partial_b1,
        svuint64_t& partial_b2) const
    {
        const svuint64_t prev = load_word(adult_word_ptr - 1);
        const svuint64_t curr = load_word(adult_word_ptr);
        const svuint64_t next = load_word(adult_word_ptr + 1);

        const svuint64_t left_2 = shift_left_with_incoming<2>(curr, prev);
        const svuint64_t left_1 = shift_left_with_incoming<1>(curr, prev);
        const svuint64_t right_1 = shift_right_with_incoming<1>(curr, next);
        const svuint64_t right_2 = shift_right_with_incoming<2>(curr, next);

        svuint64_t sum_1;
        svuint64_t carry_1;
        svuint64_t sum_2;
        svuint64_t carry_2;

        carry_save_add(left_2, left_1, curr, sum_1, carry_1);
        carry_save_add(right_1, right_2, sum_1, sum_2, carry_2);
        partial_b0 = sum_2;
        partial_b1 = xor_words(carry_1, carry_2);
        partial_b2 = and_words(carry_1, carry_2);
    }

    // Dual-stream version of compute_partial_from_word_ptr specialized for the
    // aligned pair2 hot loop so both incoming rows are built before the count
    // update path starts consuming more vector registers.
    ALWAYS_INLINE void compute_two_partials_from_word_ptrs(
        const uint64_t* __restrict adult_word_ptr0,
        const uint64_t* __restrict adult_word_ptr1,
        svuint64_t& partial0_b0,
        svuint64_t& partial0_b1,
        svuint64_t& partial0_b2,
        svuint64_t& partial1_b0,
        svuint64_t& partial1_b1,
        svuint64_t& partial1_b2) const
    {
        const svuint64_t prev0 = load_word(adult_word_ptr0 - 1);
        const svuint64_t curr0 = load_word(adult_word_ptr0);
        const svuint64_t next0 = load_word(adult_word_ptr0 + 1);
        const svuint64_t prev1 = load_word(adult_word_ptr1 - 1);
        const svuint64_t curr1 = load_word(adult_word_ptr1);
        const svuint64_t next1 = load_word(adult_word_ptr1 + 1);

        const svuint64_t left0_2 = shift_left_with_incoming<2>(curr0, prev0);
        const svuint64_t left0_1 = shift_left_with_incoming<1>(curr0, prev0);
        const svuint64_t left1_2 = shift_left_with_incoming<2>(curr1, prev1);
        const svuint64_t left1_1 = shift_left_with_incoming<1>(curr1, prev1);

        svuint64_t sum0_1;
        svuint64_t carry0_1;
        svuint64_t sum1_1;
        svuint64_t carry1_1;
        carry_save_add(left0_2, left0_1, curr0, sum0_1, carry0_1);
        carry_save_add(left1_2, left1_1, curr1, sum1_1, carry1_1);

        const svuint64_t right0_1 = shift_right_with_incoming<1>(curr0, next0);
        const svuint64_t right0_2 = shift_right_with_incoming<2>(curr0, next0);
        const svuint64_t right1_1 = shift_right_with_incoming<1>(curr1, next1);
        const svuint64_t right1_2 = shift_right_with_incoming<2>(curr1, next1);

        svuint64_t carry0_2;
        svuint64_t carry1_2;
        carry_save_add(right0_1, right0_2, sum0_1, partial0_b0, carry0_2);
        carry_save_add(right1_1, right1_2, sum1_1, partial1_b0, carry1_2);
        partial0_b1 = xor_words(carry0_1, carry0_2);
        partial0_b2 = and_words(carry0_1, carry0_2);
        partial1_b1 = xor_words(carry1_1, carry1_2);
        partial1_b2 = and_words(carry1_1, carry1_2);
    }

    // Loads the exact five-plane adult-neighbour totals for one packed two-word
    // tile from the rolling accumulator arrays.
    ALWAYS_INLINE void load_counts(
        const HotRowKernelContext& context,
        int word_index,
        svuint64_t& count_b0,
        svuint64_t& count_b1,
        svuint64_t& count_b2,
        svuint64_t& count_b3,
        svuint64_t& count_b4) const
    {
        const VerticalAccumulatorPair2& pair =
            context.total_pairs[vertical_accumulator_pair_index(word_index)];
        count_b0 = load_word(pair.b0);
        count_b1 = load_word(pair.b1);
        count_b2 = load_word(pair.b2);
        count_b3 = load_word(pair.b3);
        count_b4 = load_word(pair.b4);
    }

    // Stores updated five-plane adult-neighbour totals for one packed two-word
    // tile back into the rolling accumulator arrays.
    ALWAYS_INLINE void store_counts(
        const HotRowKernelContext& context,
        int word_index,
        svuint64_t count_b0,
        svuint64_t count_b1,
        svuint64_t count_b2,
        svuint64_t count_b3,
        svuint64_t count_b4) const
    {
        VerticalAccumulatorPair2& pair =
            context.total_pairs[vertical_accumulator_pair_index(word_index)];
        store_word(pair.b0, count_b0);
        store_word(pair.b1, count_b1);
        store_word(pair.b2, count_b2);
        store_word(pair.b3, count_b3);
        store_word(pair.b4, count_b4);
    }

    ALWAYS_INLINE void load_counts_from_pair(
        const VerticalAccumulatorPair2& total_pair,
        svuint64_t& count_b0,
        svuint64_t& count_b1,
        svuint64_t& count_b2,
        svuint64_t& count_b3,
        svuint64_t& count_b4) const
    {
        count_b0 = load_word(total_pair.b0);
        count_b1 = load_word(total_pair.b1);
        count_b2 = load_word(total_pair.b2);
        count_b3 = load_word(total_pair.b3);
        count_b4 = load_word(total_pair.b4);
    }

    ALWAYS_INLINE void store_counts_to_pair(
        VerticalAccumulatorPair2& total_pair,
        svuint64_t count_b0,
        svuint64_t count_b1,
        svuint64_t count_b2,
        svuint64_t count_b3,
        svuint64_t count_b4) const
    {
        store_word(total_pair.b0, count_b0);
        store_word(total_pair.b1, count_b1);
        store_word(total_pair.b2, count_b2);
        store_word(total_pair.b3, count_b3);
        store_word(total_pair.b4, count_b4);
    }

    // Applies the transition rules to the exact counts of one packed two-word
    // tile and writes the resulting adult and egg masks into the next-state planes.
    ALWAYS_INLINE void emit_next_state(
        const HotRowKernelContext& context,
        size_t flat_word_index,
        svuint64_t count_b0,
        svuint64_t count_b1,
        svuint64_t count_b2,
        svuint64_t count_b3,
        svuint64_t count_b4) const
    {
        svuint64_t next_adult_word;
        svuint64_t next_egg_word;
        write_next_state_from_counts(
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4,
            load_word(context.adults + flat_word_index),
            load_word(context.juveniles + flat_word_index),
            load_word(context.eggs + flat_word_index),
            next_adult_word,
            next_egg_word);
        store_word(context.juveniles + flat_word_index, next_adult_word);
        store_word(context.next_eggs + flat_word_index, next_egg_word);
    }

    ALWAYS_INLINE void emit_next_state_from_word_ptrs(
        const uint64_t* adult_word_ptr,
        uint64_t* juvenile_word_ptr,
        const uint64_t* egg_word_ptr,
        uint64_t* next_egg_word_ptr,
        svuint64_t count_b0,
        svuint64_t count_b1,
        svuint64_t count_b2,
        svuint64_t count_b3,
        svuint64_t count_b4) const
    {
        svuint64_t next_adult_word;
        svuint64_t next_egg_word;
        write_next_state_from_counts(
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4,
            load_word(adult_word_ptr),
            load_word(juvenile_word_ptr),
            load_word(egg_word_ptr),
            next_adult_word,
            next_egg_word);
        store_word(juvenile_word_ptr, next_adult_word);
        store_word(next_egg_word_ptr, next_egg_word);
    }

    // Specialized row step for the hot pair2 loop: emit next state from the
    // current total, store the results immediately, then advance the vertical
    // window for the next row. This keeps the late emit/update cluster local
    // to one helper and avoids carrying both output vectors across a second
    // helper boundary.
    ALWAYS_INLINE void emit_next_state_and_slide_from_word_ptrs(
        const uint64_t* adult_word_ptr,
        uint64_t* juvenile_word_ptr,
        const uint64_t* egg_word_ptr,
        uint64_t* next_egg_word_ptr,
        svuint64_t& count_b0,
        svuint64_t& count_b1,
        svuint64_t& count_b2,
        svuint64_t& count_b3,
        svuint64_t& count_b4,
        svuint64_t outgoing_b0,
        svuint64_t outgoing_b1,
        svuint64_t outgoing_b2,
        svuint64_t incoming_b0,
        svuint64_t incoming_b1,
        svuint64_t incoming_b2) const
    {
        const svuint64_t adult_word = load_word(adult_word_ptr);
        const svuint64_t juvenile_word = load_word(juvenile_word_ptr);
        const svuint64_t egg_word = load_word(egg_word_ptr);
        const svuint64_t adult_or_juvenile = or_words(adult_word, juvenile_word);
        const svuint64_t occupied_word = or_words(adult_or_juvenile, egg_word);
        const svuint64_t not_b4 = invert_word(count_b4);
        const svuint64_t not_b3 = invert_word(count_b3);
        const svuint64_t not_b2 = invert_word(count_b2);
        const svuint64_t not_b1 = invert_word(count_b1);
        const svuint64_t b1_or_b0 = or_words(count_b1, count_b0);
        const svuint64_t b1_and_b0 = and_words(count_b1, count_b0);
        const svuint64_t not_b1_and_b0 = invert_word(b1_and_b0);

        const svuint64_t adult_when_b3_clear = and_words(count_b2, b1_or_b0);
        const svuint64_t adult_when_b3_set = and_words(not_b2, not_b1_and_b0);
        const svuint64_t adult_5_to_10 =
            and_words(not_b4, svbsl_u64(adult_when_b3_set, adult_when_b3_clear, count_b3));

        const svuint64_t egg_3_to_5 = and_words(
            and_words(not_b4, not_b3),
            svbsl_u64(not_b1, b1_and_b0, count_b2));

        store_word(
            juvenile_word_ptr,
            svbsl_u64(adult_or_juvenile, juvenile_word, adult_5_to_10));
        store_word(next_egg_word_ptr, bic_words(egg_3_to_5, occupied_word));

        subtract_partial_from_total(
            outgoing_b0,
            outgoing_b1,
            outgoing_b2,
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4);
        add_partial_to_total(
            incoming_b0,
            incoming_b1,
            incoming_b2,
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4);
    }

    // Advances the 5-row vertical window for one packed two-word tile by
    // subtracting the outgoing row partial and adding the incoming row partial.
    ALWAYS_INLINE void slide_window(
        svuint64_t& count_b0,
        svuint64_t& count_b1,
        svuint64_t& count_b2,
        svuint64_t& count_b3,
        svuint64_t& count_b4,
        svuint64_t outgoing_b0,
        svuint64_t outgoing_b1,
        svuint64_t outgoing_b2,
        svuint64_t incoming_b0,
        svuint64_t incoming_b1,
        svuint64_t incoming_b2) const
    {
        subtract_partial_from_total(
            outgoing_b0,
            outgoing_b1,
            outgoing_b2,
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4);
        add_partial_to_total(
            incoming_b0,
            incoming_b1,
            incoming_b2,
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4);
    }

    // Adds one packed two-word horizontal partial into the exact-count total
    // registers for the same tile.
    ALWAYS_INLINE void add_partial(
        svuint64_t& count_b0,
        svuint64_t& count_b1,
        svuint64_t& count_b2,
        svuint64_t& count_b3,
        svuint64_t& count_b4,
        svuint64_t partial_b0,
        svuint64_t partial_b1,
        svuint64_t partial_b2) const
    {
        add_partial_to_total(
            partial_b0,
            partial_b1,
            partial_b2,
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4);
    }

private:
    // NEON helpers remain in the file for the scalar/vector experiments below,
    // but this candidate keeps the packed two-word hot backend fully on SVE.
    static ALWAYS_INLINE uint64x2_t neon_or(uint64x2_t lhs, uint64x2_t rhs)
    {
        return vorrq_u64(lhs, rhs);
    }

    static ALWAYS_INLINE uint64x2_t neon_and(uint64x2_t lhs, uint64x2_t rhs)
    {
        return vandq_u64(lhs, rhs);
    }

    static ALWAYS_INLINE uint64x2_t neon_xor(uint64x2_t lhs, uint64x2_t rhs)
    {
        return veorq_u64(lhs, rhs);
    }

    static ALWAYS_INLINE uint64x2_t neon_not(uint64x2_t word)
    {
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(word)));
    }

    // Returns the full-lane predicate used for all pair2 loads, stores, and ops.
    ALWAYS_INLINE svbool_t predicate() const
    {
        return svptrue_b64();
    }

    // Loads one two-lane packed tile plane into an SVE register.
    ALWAYS_INLINE svuint64_t load_word(const uint64_t* word_ptr) const
    {
        return svld1_u64(predicate(), word_ptr);
    }

    // Stores one two-lane packed tile plane from an SVE register.
    ALWAYS_INLINE void store_word(uint64_t* word_ptr, svuint64_t word) const
    {
        svst1_u64(predicate(), word_ptr, word);
    }

    // Wrapped logical OR used so the algorithm reads in domain terms instead
    // of raw intrinsic names.
    ALWAYS_INLINE svuint64_t or_words(svuint64_t lhs, svuint64_t rhs) const
    {
        return svorr_x(predicate(), lhs, rhs);
    }

    // Wrapped logical AND used by the packed-tile arithmetic helpers.
    ALWAYS_INLINE svuint64_t and_words(svuint64_t lhs, svuint64_t rhs) const
    {
        return svand_x(predicate(), lhs, rhs);
    }

    // Wrapped logical XOR used by the packed-tile arithmetic helpers.
    ALWAYS_INLINE svuint64_t xor_words(svuint64_t lhs, svuint64_t rhs) const
    {
        return sveor_x(predicate(), lhs, rhs);
    }

    // Wrapped logical three-input XOR used where the hot path combines
    // equally-weighted bit planes without carry propagation.
    ALWAYS_INLINE svuint64_t xor3_words(
        svuint64_t first,
        svuint64_t second,
        svuint64_t third) const
    {
        return sveor3(first, second, third);
    }

    // Wrapped logical bit-clear used for the common `a & ~b` shape.
    ALWAYS_INLINE svuint64_t bic_words(svuint64_t lhs, svuint64_t rhs) const
    {
        return svbic_x(predicate(), lhs, rhs);
    }

    // Bitwise NOT on a packed two-word tile.
    ALWAYS_INLINE svuint64_t invert_word(svuint64_t word) const
    {
        return svnot_x(predicate(), word);
    }

    // Logical left shift within each 64-bit lane of a packed two-word tile.
    template <int Shift>
    ALWAYS_INLINE svuint64_t shift_left(svuint64_t word) const
    {
        return svlsl_n_u64_x(predicate(), word, Shift);
    }

    // Logical right shift within each 64-bit lane of a packed two-word tile.
    template <int Shift>
    ALWAYS_INLINE svuint64_t shift_right(svuint64_t word) const
    {
        return svlsr_n_u64_x(predicate(), word, Shift);
    }

    // Shifts the current tile left and injects carry-in bits from the previous
    // word so each lane sees its true across-word neighbour bits.
    template <int Shift>
    ALWAYS_INLINE svuint64_t shift_left_with_incoming(
        svuint64_t curr,
        svuint64_t prev) const
    {
        static_assert(Shift == 1 || Shift == 2);
        svuint64_t result = shift_right<64 - Shift>(prev);
        if constexpr (Shift == 1) {
            asm("sli %0.d, %1.d, #1" : "+w"(result) : "w"(curr));
        } else {
            asm("sli %0.d, %1.d, #2" : "+w"(result) : "w"(curr));
        }
        return result;
    }

    // Shifts the current tile right and injects carry-in bits from the next
    // word so each lane sees its true across-word neighbour bits.
    template <int Shift>
    ALWAYS_INLINE svuint64_t shift_right_with_incoming(
        svuint64_t curr,
        svuint64_t next) const
    {
        static_assert(Shift == 1 || Shift == 2);
        svuint64_t result = shift_left<64 - Shift>(next);
        if constexpr (Shift == 1) {
            asm("sri %0.d, %1.d, #1" : "+w"(result) : "w"(curr));
        } else {
            asm("sri %0.d, %1.d, #2" : "+w"(result) : "w"(curr));
        }
        return result;
    }

    // Carry-save add for three equally weighted packed-tile bit masks. This
    // produces a sum plane and a carry plane without carry propagation.
    ALWAYS_INLINE void carry_save_add(
        svuint64_t a,
        svuint64_t b,
        svuint64_t c,
        svuint64_t& sum,
        svuint64_t& carry) const
    {
        sum = xor3_words(a, b, c);
        carry = svbsl_u64(or_words(a, b), and_words(a, b), c);
    }

    // Adds one packed two-word horizontal partial into the five-plane exact
    // total registers for the same tile.
    ALWAYS_INLINE void add_partial_to_total(
        svuint64_t partial_b0,
        svuint64_t partial_b1,
        svuint64_t partial_b2,
        svuint64_t& count_b0,
        svuint64_t& count_b1,
        svuint64_t& count_b2,
        svuint64_t& count_b3,
        svuint64_t& count_b4) const
    {
        svuint64_t carry = and_words(count_b0, partial_b0);
        count_b0 = xor_words(count_b0, partial_b0);

        carry_save_add(count_b1, partial_b1, carry, count_b1, carry);
        carry_save_add(count_b2, partial_b2, carry, count_b2, carry);

        const svuint64_t carry4 = and_words(count_b3, carry);
        count_b3 = xor_words(count_b3, carry);
        count_b4 = xor_words(count_b4, carry4);
    }

    // Removes one packed two-word horizontal partial from the five-plane exact
    // totals when a source row leaves the sliding vertical window.
    ALWAYS_INLINE void subtract_partial_from_total(
        svuint64_t partial_b0,
        svuint64_t partial_b1,
        svuint64_t partial_b2,
        svuint64_t& count_b0,
        svuint64_t& count_b1,
        svuint64_t& count_b2,
        svuint64_t& count_b3,
        svuint64_t& count_b4) const
    {
        svuint64_t borrow = bic_words(partial_b0, count_b0);
        count_b0 = xor_words(count_b0, partial_b0);

        const svuint64_t not_total_b1 = invert_word(count_b1);
        svuint64_t next_borrow = svbsl_u64(
            or_words(not_total_b1, partial_b1),
            bic_words(partial_b1, count_b1),
            borrow);
        count_b1 = xor3_words(count_b1, partial_b1, borrow);
        borrow = next_borrow;

        const svuint64_t not_total_b2 = invert_word(count_b2);
        next_borrow = svbsl_u64(
            or_words(not_total_b2, partial_b2),
            bic_words(partial_b2, count_b2),
            borrow);
        count_b2 = xor3_words(count_b2, partial_b2, borrow);
        borrow = next_borrow;

        next_borrow = bic_words(borrow, count_b3);
        count_b3 = xor_words(count_b3, borrow);
        count_b4 = xor_words(count_b4, next_borrow);
    }

    // Converts one packed two-word exact count total plus the current
    // adult/juvenile/egg occupancy masks into the next adult and egg masks.
    ALWAYS_INLINE void write_next_state_from_counts(
        svuint64_t count_b0,
        svuint64_t count_b1,
        svuint64_t count_b2,
        svuint64_t count_b3,
        svuint64_t count_b4,
        svuint64_t adult_word,
        svuint64_t juvenile_word,
        svuint64_t egg_word,
        svuint64_t& next_adult_word,
        svuint64_t& next_egg_word) const
    {
        const svuint64_t blocked_word = or_words(juvenile_word, egg_word);
        const svuint64_t occupied_word = or_words(adult_word, blocked_word);
        const svuint64_t empty_word = invert_word(occupied_word);
        const svuint64_t not_b4 = invert_word(count_b4);
        const svuint64_t not_b3 = invert_word(count_b3);
        const svuint64_t not_b2 = invert_word(count_b2);
        const svuint64_t not_b1 = invert_word(count_b1);
        const svuint64_t b1_or_b0 = or_words(count_b1, count_b0);
        const svuint64_t b1_and_b0 = and_words(count_b1, count_b0);
        const svuint64_t not_b1_and_b0 = invert_word(b1_and_b0);

        // Express the same logic as ternary selects so GCC can lower the
        // packed-tile emit network to `bsl` rather than a larger OR/AND tree.
        const svuint64_t adult_when_b3_clear = and_words(count_b2, b1_or_b0);
        const svuint64_t adult_when_b3_set = and_words(not_b2, not_b1_and_b0);
        const svuint64_t adult_5_to_10 =
            and_words(not_b4, svbsl_u64(adult_when_b3_set, adult_when_b3_clear, count_b3));

        const svuint64_t egg_when_b2_clear = b1_and_b0;
        const svuint64_t egg_when_b2_set = not_b1;
        const svuint64_t egg_3_to_5 = and_words(
            and_words(not_b4, not_b3),
            svbsl_u64(egg_when_b2_set, egg_when_b2_clear, count_b2));

        next_adult_word = or_words(and_words(adult_word, adult_5_to_10), juvenile_word);
        next_egg_word = and_words(empty_word, egg_3_to_5);
    }

    // NEON version of the next-state boolean network. This is the only mixed
    // ISA substitution in the experiment.
    static ALWAYS_INLINE void write_next_state_from_counts_neon(
        uint64x2_t count_b0,
        uint64x2_t count_b1,
        uint64x2_t count_b2,
        uint64x2_t count_b3,
        uint64x2_t count_b4,
        uint64x2_t adult_word,
        uint64x2_t juvenile_word,
        uint64x2_t egg_word,
        uint64x2_t& next_adult_word,
        uint64x2_t& next_egg_word)
    {
        const uint64x2_t all_ones = vdupq_n_u64(~0ULL);
        const uint64x2_t blocked_word = neon_or(juvenile_word, egg_word);
        const uint64x2_t occupied_word = neon_or(adult_word, blocked_word);
        const uint64x2_t empty_word = neon_xor(occupied_word, all_ones);
        const uint64x2_t not_b4 = neon_not(count_b4);
        const uint64x2_t not_b3 = neon_not(count_b3);
        const uint64x2_t not_b2 = neon_not(count_b2);
        const uint64x2_t not_b1 = neon_not(count_b1);
        const uint64x2_t b1_or_b0 = neon_or(count_b1, count_b0);
        const uint64x2_t b1_and_b0 = neon_and(count_b1, count_b0);

        const uint64x2_t adult_5_to_10 = neon_and(
            not_b4,
            neon_or(
                neon_and(not_b3, neon_and(count_b2, b1_or_b0)),
                neon_and(count_b3, neon_and(not_b2, neon_xor(b1_and_b0, all_ones)))));
        const uint64x2_t egg_3_to_5 = neon_and(
            neon_and(not_b4, not_b3),
            neon_or(
                neon_and(not_b2, b1_and_b0),
                neon_and(count_b2, not_b1)));

        next_adult_word = neon_or(neon_and(adult_word, adult_5_to_10), juvenile_word);
        next_egg_word = neon_and(empty_word, egg_3_to_5);
    }
};

struct VerticalAccumulatorRow {
    // Packed AoSoA backing storage for the rolling exact-count accumulator of
    // the currently active hot row. Each tile owns two adjacent hot words and
    // stores their five bit planes together.
    VerticalAccumulatorPair2Buffer packed_total_pairs;

    VerticalAccumulatorRow() = default;

    // Allocates zero-initialized storage for the packed exact-count tiles of one row.
    explicit VerticalAccumulatorRow(int words_per_row)
        : packed_total_pairs(packed_vertical_accumulator_pair_count(words_per_row))
    {
    }
};

struct HotRowScratch {
    // All reusable scratch state for one worker's hot interior rows.
    // `row_partials` caches the five source rows currently feeding the 5x5
    // window, and `rolling_total` stores the exact counts for the output row
    // that window is currently centered on.
    std::array<HorizontalPartialRow, 5> row_partials;
    VerticalAccumulatorRow rolling_total;

    HotRowScratch() = default;

    // Builds the five-row partial ring and the rolling exact-count row used by one worker.
    explicit HotRowScratch(int words_per_row)
    {
        for (HorizontalPartialRow& row_partial : row_partials) {
            row_partial = HorizontalPartialRow(words_per_row);
        }
        rolling_total = VerticalAccumulatorRow(words_per_row);
    }
};

// ---------------------------------------------------------------------------
// Grid indexing and input/output helpers
// ---------------------------------------------------------------------------

// Splits the grid rows into contiguous worker-owned chunks.
static RowRange compute_row_range(int grid_size, int thread_count, int worker_id)
{
    const int base_rows = grid_size / thread_count;
    const int remainder = grid_size % thread_count;
    const int begin =
        worker_id * base_rows + std::min(worker_id, remainder);
    const int count = base_rows + (worker_id < remainder ? 1 : 0);
    return RowRange{begin, begin + count};
}

// Pins the calling worker to a stable CPU to reduce scheduler jitter.
static void pin_current_thread_to_cpu(int worker_id)
{
#ifndef DISABLE_THREAD_PINNING
    cpu_set_t affinity_mask;
    CPU_ZERO(&affinity_mask);
    CPU_SET(worker_id, &affinity_mask);
    (void)pthread_setaffinity_np(
        pthread_self(),
        sizeof(affinity_mask),
        &affinity_mask);
#else
    (void)worker_id;
#endif
}

// Wraps a row or column index on the toroidal grid.
inline int wrap_coordinate(int coordinate, int grid_size)
{
    return (coordinate + grid_size) & (grid_size - 1);
}

// Maps a row/column position to the flat uint64_t word that stores its bit.
inline size_t get_flat_word_index(int row_index, int col_index, int words_per_row)
{
    int word_index_in_row = col_index >> 6;
    return static_cast<size_t>(row_index) * words_per_row + word_index_in_row;
}

// Returns the single-bit mask for a column position inside its containing word.
inline uint64_t get_bit_mask(int col_index)
{
    int bit_index_in_word = col_index & 63;
    return 1ULL << bit_index_in_word;
}

// Reads one cell bit from a packed bit-plane representation.
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

// Sets one cell bit inside a packed bit-plane representation.
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

// Converts the byte-per-cell input grid into the three packed state planes.
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

// Reference neighbourhood counter used only for the wrapping fallback cells.
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
// Hot-row pipeline
// ---------------------------------------------------------------------------

// Builds the horizontal-partial representation for one source adult row and
// writes it into a recyclable ring-buffer slot.
__attribute__((always_inline)) static inline void compute_horizontal_partial_row(
    int source_row,
    int words_per_row,
    const BitPlane& current_adults,
    HorizontalPartialRow& row_partial)
{
    const ScalarHotWordBackend scalar_backend;
    const Pair2SVEHotWordBackend vector_backend;
    const uint64_t* __restrict adults = current_adults.data();
    HorizontalPartialPair2* __restrict partial_pairs =
        row_partial.packed_partial_pairs.data();
    const int hot_word_begin = 1;
    const int hot_word_end = words_per_row - 1;

    if (hot_word_begin >= hot_word_end) {
        return;
    }

    const uint64_t* __restrict adult_row =
        adults + static_cast<size_t>(source_row) * words_per_row;

    int word_index = hot_word_begin;
    const int sve_words = static_cast<int>(svcntd());
    if (sve_words == 2 && hot_word_begin + 1 < hot_word_end) {
        const HorizontalPartialWord first_partial =
            scalar_backend.compute_partial(adult_row, word_index);
        scalar_backend.store_partial(row_partial, word_index, first_partial);
        ++word_index;

        const int aligned_simd_word_end =
            word_index + ((hot_word_end - word_index) / sve_words) * sve_words;
        for (; word_index < aligned_simd_word_end; word_index += sve_words) {
            svuint64_t partial_b0;
            svuint64_t partial_b1;
            svuint64_t partial_b2;
            vector_backend.compute_partial(
                adult_row,
                word_index,
                partial_b0,
                partial_b1,
                partial_b2);
            HorizontalPartialPair2& partial_storage =
                partial_pairs[horizontal_partial_pair_index(word_index)];
            vector_backend.store_partial(partial_storage, partial_b0, partial_b1, partial_b2);
        }
    }

    for (; word_index < hot_word_end; ++word_index) {
        const HorizontalPartialWord partial =
            scalar_backend.compute_partial(adult_row, word_index);
        scalar_backend.store_partial(row_partial, word_index, partial);
    }
}

// Sums the five seeded horizontal-partial rows into the first exact vertical
// accumulator used by the hot-row pipeline.
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

    const ScalarHotWordBackend scalar_backend;
    for (int word_index = hot_word_begin; word_index < hot_word_end; ++word_index) {
        ScalarBitSlicedCounts rolling_counts{};
        for (const HorizontalPartialRow& row_partial : row_partials) {
            const HorizontalPartialWord partial_word =
                scalar_backend.load_partial(row_partial, word_index);
            scalar_backend.add_partial(rolling_counts, partial_word);
        }
        VerticalAccumulatorPair2& total_pair =
            rolling_total.packed_total_pairs[vertical_accumulator_pair_index(word_index)];
        const int lane = vertical_accumulator_pair_lane(word_index);
        total_pair.b0[lane] = rolling_counts.b0;
        total_pair.b1[lane] = rolling_counts.b1;
        total_pair.b2[lane] = rolling_counts.b2;
        total_pair.b3[lane] = rolling_counts.b3;
        total_pair.b4[lane] = rolling_counts.b4;
    }
}

// Seeds the 5-row ring of horizontal partials for the first hot row in a chunk.
static void seed_hot_partial_ring(
    int hot_row_begin,
    int words_per_row,
    const BitPlane& current_adults,
    HotRowScratch& hot_row_scratch)
{
    for (int ring_offset = 0; ring_offset < 5; ++ring_offset) {
        compute_horizontal_partial_row(
            hot_row_begin + ring_offset - 2,
            words_per_row,
            current_adults,
            hot_row_scratch.row_partials[static_cast<size_t>(ring_offset)]);
    }
}

// Builds the exact vertical total for the first hot row from the seeded ring.
static void seed_first_hot_row_total(
    int words_per_row,
    HotRowScratch& hot_row_scratch)
{
    build_vertical_accumulator_row(
        words_per_row,
        hot_row_scratch.row_partials,
        hot_row_scratch.rolling_total);
}

// Pointer-stepped aligned pair2 sweep for the dominant two-row interior case.
// The math is unchanged; only the source shape differs so the hot loop can
// advance raw pointers instead of repeatedly rebuilding row/partial addresses.
__attribute__((noinline)) static void run_two_hot_rows_pair2_sve_ptrstep(
    int pair_count,
    const uint64_t* __restrict row0_adults_ptr,
    uint64_t* __restrict row0_juveniles_ptr,
    const uint64_t* __restrict row0_eggs_ptr,
    const uint64_t* __restrict row1_adults_ptr,
    uint64_t* __restrict row1_juveniles_ptr,
    const uint64_t* __restrict row1_eggs_ptr,
    const uint64_t* __restrict incoming_adult_word_ptr0,
    const uint64_t* __restrict incoming_adult_word_ptr1,
    uint64_t* __restrict row0_next_eggs_ptr,
    uint64_t* __restrict row1_next_eggs_ptr,
    VerticalAccumulatorPair2* total_pairs,
    HorizontalPartialPair2* partial_pairs0,
    HorizontalPartialPair2* partial_pairs1)
{
    constexpr ptrdiff_t pair2_words = 2;
    const Pair2SVEHotWordBackend vector_backend;

    for (int pair_index = 0; pair_index < pair_count; ++pair_index) {
        svuint64_t outgoing_row0_b0;
        svuint64_t outgoing_row0_b1;
        svuint64_t outgoing_row0_b2;
        svuint64_t incoming_row0_b0;
        svuint64_t incoming_row0_b1;
        svuint64_t incoming_row0_b2;
        svuint64_t outgoing_row1_b0;
        svuint64_t outgoing_row1_b1;
        svuint64_t outgoing_row1_b2;
        svuint64_t incoming_row1_b0;
        svuint64_t incoming_row1_b1;
        svuint64_t incoming_row1_b2;

        vector_backend.load_partial(
            *partial_pairs0,
            outgoing_row0_b0,
            outgoing_row0_b1,
            outgoing_row0_b2);
        vector_backend.load_partial(
            *partial_pairs1,
            outgoing_row1_b0,
            outgoing_row1_b1,
            outgoing_row1_b2);
        vector_backend.compute_two_partials_from_word_ptrs(
            incoming_adult_word_ptr0,
            incoming_adult_word_ptr1,
            incoming_row0_b0,
            incoming_row0_b1,
            incoming_row0_b2,
            incoming_row1_b0,
            incoming_row1_b1,
            incoming_row1_b2);

        svuint64_t count_b0;
        svuint64_t count_b1;
        svuint64_t count_b2;
        svuint64_t count_b3;
        svuint64_t count_b4;
        vector_backend.load_counts_from_pair(
            *total_pairs,
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4);

        vector_backend.emit_next_state_and_slide_from_word_ptrs(
            row0_adults_ptr,
            row0_juveniles_ptr,
            row0_eggs_ptr,
            row0_next_eggs_ptr,
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4,
            outgoing_row0_b0,
            outgoing_row0_b1,
            outgoing_row0_b2,
            incoming_row0_b0,
            incoming_row0_b1,
            incoming_row0_b2);

        vector_backend.emit_next_state_and_slide_from_word_ptrs(
            row1_adults_ptr,
            row1_juveniles_ptr,
            row1_eggs_ptr,
            row1_next_eggs_ptr,
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4,
            outgoing_row1_b0,
            outgoing_row1_b1,
            outgoing_row1_b2,
            incoming_row1_b0,
            incoming_row1_b1,
            incoming_row1_b2);

        vector_backend.store_counts_to_pair(
            *total_pairs,
            count_b0,
            count_b1,
            count_b2,
            count_b3,
            count_b4);
        vector_backend.store_partial(
            *partial_pairs0,
            incoming_row0_b0,
            incoming_row0_b1,
            incoming_row0_b2);
        vector_backend.store_partial(
            *partial_pairs1,
            incoming_row1_b0,
            incoming_row1_b1,
            incoming_row1_b2);

        row0_adults_ptr += pair2_words;
        row0_juveniles_ptr += pair2_words;
        row0_eggs_ptr += pair2_words;
        row1_adults_ptr += pair2_words;
        row1_juveniles_ptr += pair2_words;
        row1_eggs_ptr += pair2_words;
        incoming_adult_word_ptr0 += pair2_words;
        incoming_adult_word_ptr1 += pair2_words;
        row0_next_eggs_ptr += pair2_words;
        row1_next_eggs_ptr += pair2_words;
        ++total_pairs;
        ++partial_pairs0;
        ++partial_pairs1;
    }
}


// Runs the fast interior kernel for rows whose full 5x5 neighbourhood stays
// within the non-wrapping interior. This is the exact-two-row blocked pipeline.
__attribute__((noinline)) static void run_one_generation_hot_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_eggs,
    HotRowScratch& hot_row_scratch)
{
    const int hot_row_begin = std::max(row_begin, 2);
    const int hot_row_end = std::min(row_end, grid_size - 2);
    const int hot_word_begin = 1;
    const int hot_word_end = words_per_row - 1;

    if (hot_row_begin >= hot_row_end || hot_word_begin >= hot_word_end) {
        return;
    }

    // Phase 1: seed the 5-row ring of horizontal partials for the first hot row.
    seed_hot_partial_ring(hot_row_begin, words_per_row, current_adults, hot_row_scratch);

    // Phase 2: build the exact bit-sliced counts for the first hot row from the
    // five horizontal partial rows now resident in the ring.
    seed_first_hot_row_total(words_per_row, hot_row_scratch);

    const ScalarHotWordBackend scalar_backend;
    HotRowKernelContext context{
        .adults = current_adults.data(),
        .juveniles = current_juveniles.data(),
        .eggs = current_eggs.data(),
        .next_eggs = next_eggs.data(),
        .total_pairs = hot_row_scratch.rolling_total.packed_total_pairs.data(),
        .hot_word_begin = hot_word_begin,
        .hot_word_end = hot_word_end,
        .aligned_hot_word_end = hot_word_end - 1,
        .sve_words = static_cast<int>(svcntd()),
    };
    const Pair2SVEHotWordBackend vector_backend;
    const bool use_aligned_main_hot_loop =
        (context.sve_words == 2) && (hot_word_begin + 1 < hot_word_end);
    const int simd_word_end =
        hot_word_begin + ((hot_word_end - hot_word_begin) / context.sve_words) * context.sve_words;

    // The ring head always points at the row leaving the 5-row vertical window.
    // Each update step:
    //   emit current row
    //   subtract outgoing row partials
    //   add incoming row partials
    //   recycle the outgoing slot with the new incoming partials
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
        HorizontalPartialPair2* __restrict recycled_partial_pairs0 =
            recycled_row0.packed_partial_pairs.data();
        HorizontalPartialPair2* __restrict recycled_partial_pairs1 =
            recycled_row1.packed_partial_pairs.data();
        const uint64_t* __restrict incoming_adult_row0 =
            context.adults + static_cast<size_t>(row_index + 3) * words_per_row;
        const uint64_t* __restrict incoming_adult_row1 =
            context.adults + static_cast<size_t>(row_index + 4) * words_per_row;

        int word_index = context.hot_word_begin;
        if (use_aligned_main_hot_loop) {
            const size_t flat_word_index0 = row0_base + static_cast<size_t>(word_index);
            const size_t flat_word_index1 = row1_base + static_cast<size_t>(word_index);
            const HorizontalPartialWord outgoing_row0_partial =
                scalar_backend.load_partial(recycled_row0, word_index);
            const HorizontalPartialWord incoming_row0_partial =
                scalar_backend.compute_partial(incoming_adult_row0, word_index);
            const HorizontalPartialWord outgoing_row1_partial =
                scalar_backend.load_partial(recycled_row1, word_index);
            const HorizontalPartialWord incoming_row1_partial =
                scalar_backend.compute_partial(incoming_adult_row1, word_index);
            ScalarBitSlicedCounts rolling_counts =
                scalar_backend.load_counts(context, word_index);

            scalar_backend.emit_next_state(context, flat_word_index0, rolling_counts);
            scalar_backend.slide_window(
                rolling_counts, outgoing_row0_partial, incoming_row0_partial);

            scalar_backend.emit_next_state(context, flat_word_index1, rolling_counts);
            scalar_backend.slide_window(
                rolling_counts, outgoing_row1_partial, incoming_row1_partial);

            scalar_backend.store_counts(context, word_index, rolling_counts);
            scalar_backend.store_partial(recycled_row0, word_index, incoming_row0_partial);
            scalar_backend.store_partial(recycled_row1, word_index, incoming_row1_partial);
            ++word_index;

            const int aligned_simd_word_end =
                word_index +
                ((context.aligned_hot_word_end - word_index) / context.sve_words) *
                    context.sve_words;
            if (word_index < aligned_simd_word_end) {
                const int pair_count =
                    (aligned_simd_word_end - word_index) / context.sve_words;
                const size_t aligned_flat_word_index0 =
                    row0_base + static_cast<size_t>(word_index);
                const size_t aligned_flat_word_index1 =
                    row1_base + static_cast<size_t>(word_index);
                const size_t partial_pair_index =
                    horizontal_partial_pair_index(word_index);
                run_two_hot_rows_pair2_sve_ptrstep(
                    pair_count,
                    context.adults + aligned_flat_word_index0,
                    context.juveniles + aligned_flat_word_index0,
                    context.eggs + aligned_flat_word_index0,
                    context.adults + aligned_flat_word_index1,
                    context.juveniles + aligned_flat_word_index1,
                    context.eggs + aligned_flat_word_index1,
                    incoming_adult_row0 + word_index,
                    incoming_adult_row1 + word_index,
                    context.next_eggs + aligned_flat_word_index0,
                    context.next_eggs + aligned_flat_word_index1,
                    context.total_pairs + vertical_accumulator_pair_index(word_index),
                    recycled_partial_pairs0 + partial_pair_index,
                    recycled_partial_pairs1 + partial_pair_index);
                word_index = aligned_simd_word_end;
            }
        }

        for (; word_index < context.hot_word_end; ++word_index) {
            const size_t flat_word_index0 = row0_base + word_index;
            const size_t flat_word_index1 = row1_base + word_index;
            const HorizontalPartialWord outgoing_row0_partial =
                scalar_backend.load_partial(recycled_row0, word_index);
            const HorizontalPartialWord incoming_row0_partial =
                scalar_backend.compute_partial(incoming_adult_row0, word_index);
            const HorizontalPartialWord outgoing_row1_partial =
                scalar_backend.load_partial(recycled_row1, word_index);
            const HorizontalPartialWord incoming_row1_partial =
                scalar_backend.compute_partial(incoming_adult_row1, word_index);
            ScalarBitSlicedCounts rolling_counts =
                scalar_backend.load_counts(context, word_index);

            scalar_backend.emit_next_state(context, flat_word_index0, rolling_counts);
            scalar_backend.slide_window(
                rolling_counts, outgoing_row0_partial, incoming_row0_partial);

            scalar_backend.emit_next_state(context, flat_word_index1, rolling_counts);
            scalar_backend.slide_window(
                rolling_counts, outgoing_row1_partial, incoming_row1_partial);

            scalar_backend.store_counts(context, word_index, rolling_counts);
            scalar_backend.store_partial(recycled_row0, word_index, incoming_row0_partial);
            scalar_backend.store_partial(recycled_row1, word_index, incoming_row1_partial);
        }

        ring_head = (ring_head + 2) % hot_row_scratch.row_partials.size();
    }

    // If the chunk leaves exactly one "emit + advance" row before the final
    // emit-only row, handle it here. With even-sized hot ranges this is the
    // expected penultimate step before the final row below.
    for (; row_index + 1 < hot_row_end; ++row_index) {
        const size_t row_base =
            static_cast<size_t>(row_index) * words_per_row;
        HorizontalPartialRow& recycled_row =
            hot_row_scratch.row_partials[ring_head];
        HorizontalPartialPair2* __restrict recycled_partial_pairs =
            recycled_row.packed_partial_pairs.data();
        const HorizontalPartialPair2* __restrict outgoing_partial_pairs =
            recycled_partial_pairs;
        const uint64_t* __restrict incoming_adult_row =
            context.adults + static_cast<size_t>(row_index + 3) * words_per_row;

        int word_index = context.hot_word_begin;
        if (use_aligned_main_hot_loop) {
            const size_t flat_word_index = row_base + static_cast<size_t>(word_index);
            const HorizontalPartialWord outgoing_partial =
                scalar_backend.load_partial(recycled_row, word_index);
            const HorizontalPartialWord incoming_partial =
                scalar_backend.compute_partial(incoming_adult_row, word_index);
            ScalarBitSlicedCounts rolling_counts =
                scalar_backend.load_counts(context, word_index);

            scalar_backend.emit_next_state(context, flat_word_index, rolling_counts);
            scalar_backend.slide_window(rolling_counts, outgoing_partial, incoming_partial);

            scalar_backend.store_counts(context, word_index, rolling_counts);
            scalar_backend.store_partial(recycled_row, word_index, incoming_partial);
            ++word_index;

            const int aligned_simd_word_end =
                word_index +
                ((context.aligned_hot_word_end - word_index) / context.sve_words) *
                    context.sve_words;
            for (; word_index < aligned_simd_word_end; word_index += context.sve_words) {
                const size_t aligned_flat_word_index =
                    row_base + static_cast<size_t>(word_index);
                const size_t partial_pair_index =
                    horizontal_partial_pair_index(word_index);
                svuint64_t outgoing_b0;
                svuint64_t outgoing_b1;
                svuint64_t outgoing_b2;
                svuint64_t incoming_b0;
                svuint64_t incoming_b1;
                svuint64_t incoming_b2;
                vector_backend.load_partial(
                    outgoing_partial_pairs[partial_pair_index],
                    outgoing_b0,
                    outgoing_b1,
                    outgoing_b2);
                vector_backend.compute_partial(
                    incoming_adult_row,
                    word_index,
                    incoming_b0,
                    incoming_b1,
                    incoming_b2);

                svuint64_t count_b0;
                svuint64_t count_b1;
                svuint64_t count_b2;
                svuint64_t count_b3;
                svuint64_t count_b4;
                vector_backend.load_counts(
                    context,
                    word_index,
                    count_b0,
                    count_b1,
                    count_b2,
                    count_b3,
                    count_b4);

                vector_backend.emit_next_state(
                    context,
                    aligned_flat_word_index,
                    count_b0,
                    count_b1,
                    count_b2,
                    count_b3,
                    count_b4);
                vector_backend.slide_window(
                    count_b0,
                    count_b1,
                    count_b2,
                    count_b3,
                    count_b4,
                    outgoing_b0,
                    outgoing_b1,
                    outgoing_b2,
                    incoming_b0,
                    incoming_b1,
                    incoming_b2);
                vector_backend.store_counts(
                    context,
                    word_index,
                    count_b0,
                    count_b1,
                    count_b2,
                    count_b3,
                    count_b4);
                vector_backend.store_partial(
                    recycled_partial_pairs[partial_pair_index],
                    incoming_b0,
                    incoming_b1,
                    incoming_b2);
            }
        }

        for (; word_index < context.hot_word_end; ++word_index) {
            const size_t flat_word_index = row_base + word_index;
            const HorizontalPartialWord outgoing_partial =
                scalar_backend.load_partial(recycled_row, word_index);
            const HorizontalPartialWord incoming_partial =
                scalar_backend.compute_partial(incoming_adult_row, word_index);
            ScalarBitSlicedCounts rolling_counts =
                scalar_backend.load_counts(context, word_index);

            scalar_backend.emit_next_state(context, flat_word_index, rolling_counts);
            scalar_backend.slide_window(rolling_counts, outgoing_partial, incoming_partial);

            scalar_backend.store_counts(context, word_index, rolling_counts);
            scalar_backend.store_partial(recycled_row, word_index, incoming_partial);
        }

        ring_head = (ring_head + 1) % hot_row_scratch.row_partials.size();
    }

    const int final_row = hot_row_end - 1;
    const size_t final_row_base =
        static_cast<size_t>(final_row) * words_per_row;
    int word_index = context.hot_word_begin;
    if (use_aligned_main_hot_loop) {
        const size_t flat_word_index = final_row_base + static_cast<size_t>(word_index);
        scalar_backend.emit_next_state(
            context,
            flat_word_index,
            scalar_backend.load_counts(context, word_index));
        ++word_index;

        const int aligned_simd_word_end =
            word_index +
            ((context.aligned_hot_word_end - word_index) / context.sve_words) *
                context.sve_words;
        for (; word_index < aligned_simd_word_end; word_index += context.sve_words) {
            const size_t aligned_flat_word_index =
                final_row_base + static_cast<size_t>(word_index);
            svuint64_t count_b0;
            svuint64_t count_b1;
            svuint64_t count_b2;
            svuint64_t count_b3;
            svuint64_t count_b4;
            vector_backend.load_counts(
                context,
                word_index,
                count_b0,
                count_b1,
                count_b2,
                count_b3,
                count_b4);
            vector_backend.emit_next_state(
                context,
                aligned_flat_word_index,
                count_b0,
                count_b1,
                count_b2,
                count_b3,
                count_b4);
        }
    } else {
        for (; word_index < simd_word_end; word_index += context.sve_words) {
            const size_t flat_word_index = final_row_base + static_cast<size_t>(word_index);
            svuint64_t count_b0;
            svuint64_t count_b1;
            svuint64_t count_b2;
            svuint64_t count_b3;
            svuint64_t count_b4;
            vector_backend.load_counts(
                context,
                word_index,
                count_b0,
                count_b1,
                count_b2,
                count_b3,
                count_b4);
            vector_backend.emit_next_state(
                context,
                flat_word_index,
                count_b0,
                count_b1,
                count_b2,
                count_b3,
                count_b4);
        }
    }

    for (; word_index < context.hot_word_end; ++word_index) {
        const size_t flat_word_index = final_row_base + word_index;
        scalar_backend.emit_next_state(
            context,
            flat_word_index,
            scalar_backend.load_counts(context, word_index));
    }
}

// ---------------------------------------------------------------------------
// Edge and wrapping fallbacks
// ---------------------------------------------------------------------------

// Clears the next-egg scratch cells that are not fully overwritten by the hot
// interior kernel before the edge/wrapping passes update them.
static void initialize_next_generation_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    BitPlane& next_eggs)
{
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
                next_eggs_ptr[flat_word_index] = 0ULL;
            }
        } else {
            const size_t left_word_index = row_base;
            const size_t right_word_index = row_base + last_word_offset;
            next_eggs_ptr[left_word_index] = 0ULL;
            next_eggs_ptr[right_word_index] = 0ULL;
        }
    }
}

// Adds one full horizontal contributor row (five column offsets) into the edge
// counter accumulator.
ALWAYS_INLINE void add_full_neighbor_row_to_counts(
    ScalarBitSlicedCounts& counts,
    uint64_t left_2,
    uint64_t left_1,
    uint64_t center,
    uint64_t right_1,
    uint64_t right_2)
{
    const ScalarHotWordBackend scalar_backend;
    scalar_backend.add_mask(counts, left_2);
    scalar_backend.add_mask(counts, left_1);
    scalar_backend.add_mask(counts, center);
    scalar_backend.add_mask(counts, right_1);
    scalar_backend.add_mask(counts, right_2);
}

// Adds the center row's four non-self contributors into the edge accumulator.
ALWAYS_INLINE void add_center_neighbor_row_to_counts(
    ScalarBitSlicedCounts& counts,
    uint64_t left_2,
    uint64_t left_1,
    uint64_t right_1,
    uint64_t right_2)
{
    const ScalarHotWordBackend scalar_backend;
    scalar_backend.add_mask(counts, left_2);
    scalar_backend.add_mask(counts, left_1);
    scalar_backend.add_mask(counts, right_1);
    scalar_backend.add_mask(counts, right_2);
}

// Applies the scalar transition rules to an already-assembled edge-word count,
// while masking off the invalid wrapping columns handled elsewhere.
ALWAYS_INLINE void finalize_masked_edge_word(
    const ScalarBitSlicedCounts& counts,
    uint64_t adult_word,
    uint64_t juvenile_word,
    uint64_t blocked_word,
    uint64_t valid_mask,
    uint64_t& next_adult_word,
    uint64_t& next_egg_word)
{
    const uint64_t adult_valid = adult_word & valid_mask;
    const uint64_t juvenile_valid = juvenile_word & valid_mask;
    const uint64_t empty_valid = ~(adult_word | blocked_word) & valid_mask;
    const uint64_t ge3 = counts.b4 | counts.b3 | counts.b2 | (counts.b1 & counts.b0);
    const uint64_t ge4 = counts.b4 | counts.b3 | counts.b2;
    const uint64_t ge6 = counts.b4 | counts.b3 | (counts.b2 & counts.b1);
    const uint64_t ge10 = counts.b4 | (counts.b3 & (counts.b2 | counts.b1));

    const uint64_t next_adult_valid = juvenile_valid | (adult_valid & ge4 & ~ge10);
    const uint64_t next_egg_valid = empty_valid & ge3 & ~ge6;

    next_adult_word = (next_adult_word & ~valid_mask) | next_adult_valid;
    next_egg_word = (next_egg_word & ~valid_mask) | next_egg_valid;
}

// Handles the valid non-wrapping columns in the leftmost word of each interior row.
__attribute__((noinline)) static void run_one_generation_left_edge_valid_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict adults = current_adults.data();
    uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs = current_eggs.data();
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

        ScalarBitSlicedCounts counts{};

        add_full_neighbor_row_to_counts(
            counts,
            top_left_2,
            top_left_1,
            top_center,
            top_right_1,
            top_right_2);
        add_full_neighbor_row_to_counts(
            counts,
            upper_left_2,
            upper_left_1,
            upper_center,
            upper_right_1,
            upper_right_2);
        add_center_neighbor_row_to_counts(
            counts,
            center_left_2,
            center_left_1,
            center_right_1,
            center_right_2);
        add_full_neighbor_row_to_counts(
            counts,
            lower_left_2,
            lower_left_1,
            lower_center,
            lower_right_1,
            lower_right_2);
        add_full_neighbor_row_to_counts(
            counts,
            bottom_left_2,
            bottom_left_1,
            bottom_center,
            bottom_right_1,
            bottom_right_2);

        const uint64_t juvenile_word = juveniles[row_base];
        finalize_masked_edge_word(
            counts,
            adults[row_base],
            juvenile_word,
            juvenile_word | eggs[row_base],
            valid_mask,
            juveniles[row_base],
            next_eggs_ptr[row_base]);
    }
}

// Handles the valid non-wrapping columns in the rightmost word of each interior row.
__attribute__((noinline)) static void run_one_generation_right_edge_valid_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict adults = current_adults.data();
    uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs = current_eggs.data();
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

        ScalarBitSlicedCounts counts{};

        add_full_neighbor_row_to_counts(
            counts,
            top_left_2,
            top_left_1,
            top_center,
            top_right_1,
            top_right_2);
        add_full_neighbor_row_to_counts(
            counts,
            upper_left_2,
            upper_left_1,
            upper_center,
            upper_right_1,
            upper_right_2);
        add_center_neighbor_row_to_counts(
            counts,
            center_left_2,
            center_left_1,
            center_right_1,
            center_right_2);
        add_full_neighbor_row_to_counts(
            counts,
            lower_left_2,
            lower_left_1,
            lower_center,
            lower_right_1,
            lower_right_2);
        add_full_neighbor_row_to_counts(
            counts,
            bottom_left_2,
            bottom_left_1,
            bottom_center,
            bottom_right_1,
            bottom_right_2);

        const uint64_t juvenile_word = juveniles[edge_word_index];
        finalize_masked_edge_word(
            counts,
            adults[edge_word_index],
            juvenile_word,
            juvenile_word | eggs[edge_word_index],
            valid_mask,
            juveniles[edge_word_index],
            next_eggs_ptr[edge_word_index]);
    }
}

// Handles the rows/columns whose 5x5 neighbourhood wraps around the torus.
__attribute__((noinline)) static void run_one_generation_wrapping_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_eggs)
{
    const uint64_t* __restrict adults = current_adults.data();
    uint64_t* __restrict juveniles = current_juveniles.data();
    const uint64_t* __restrict eggs = current_eggs.data();
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
                    juveniles[flat_word_index] |= cell_mask;
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
                    juveniles[left_edge_word_index] |= cell_mask;
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
                    juveniles[right_edge_word_index] |= cell_mask;
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
                    juveniles[flat_word_index] |= cell_mask;
                }
            } else if (adult_neighbor_count >= 3 && adult_neighbor_count <= 5) {
                next_eggs_ptr[flat_word_index] |= cell_mask;
            }
        }
    }
}

// Dispatches all edge/wrapping fallback handlers for one worker-owned chunk.
__attribute__((noinline)) static void run_one_generation_edge_rows(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    BitPlane& current_juveniles,
    const BitPlane& current_eggs,
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
        next_eggs);

    run_one_generation_right_edge_valid_rows(
        row_begin,
        row_end,
        grid_size,
        words_per_row,
        current_adults,
        current_juveniles,
        current_eggs,
        next_eggs);

    run_one_generation_wrapping_rows(
        row_begin,
        row_end,
        grid_size,
        words_per_row,
        current_adults,
        current_juveniles,
        current_eggs,
        next_eggs);
}

// ---------------------------------------------------------------------------
// Generation orchestration
// ---------------------------------------------------------------------------

// Processes one worker-owned row chunk for one generation.
static void process_generation_chunk(
    int row_begin,
    int row_end,
    int grid_size,
    int words_per_row,
    const BitPlane& current_adults,
    BitPlane& current_juveniles,
    const BitPlane& current_eggs,
    BitPlane& next_eggs,
    HotRowScratch& hot_row_scratch)
{
    initialize_next_generation_rows(
        row_begin,
        row_end,
        grid_size,
        words_per_row,
        next_eggs);

    run_one_generation_hot_rows(
        row_begin,
        row_end,
        grid_size,
        words_per_row,
        current_adults,
        current_juveniles,
        current_eggs,
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
        next_eggs);
}

// Converts the three packed state planes back into the byte-per-cell output grid.
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

// Entry point: loads the input grid, runs the requested generations, and writes the result.
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
            next_eggs,
            main_hot_row_scratch);

        generation_done_barrier.arrive_and_wait();

        current_adults.swap(current_juveniles);
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
