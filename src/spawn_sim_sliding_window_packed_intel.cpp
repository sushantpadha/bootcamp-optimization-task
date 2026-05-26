// spawn_sim_packed_x86.cpp  —  Monster Spawning Grid, Intel AVX2, 2-bit packed grid.
//
// Same algorithm and packing format as spawn_sim_packed_arm.cpp.
// Only the SIMD layer differs.  See that file for the full rationale.
//
// PACK / UNPACK WITH SSE2 + SSSE3
// ────────────────────────────────
// Intel lacks vld4q_u8 / vst4q_u8, so we build the 4-way interleave manually.
//
// UNPACK (16 packed bytes → 64 sequential state bytes):
//   Extract 4 stride-4 sub-arrays with shifts + masks (using __m128i).
//   Two rounds of SSE2 unpack reconstruct sequential order:
//     Round 1: _mm_unpacklo/hi_epi8(s0,s1) and _mm_unpacklo/hi_epi8(s2,s3)
//              → pairs  [c0,c1,c4,c5,...] and [c2,c3,c6,c7,...]
//     Round 2: _mm_unpacklo/hi_epi16(pair01, pair23)
//              → quads  [c0,c1,c2,c3,c4,...] ✓
//
// PACK (64 sequential state bytes → 16 packed bytes):
//   Uses _mm_maddubs_epi16 (SSSE3) to accumulate pairs:
//     [c0+4·c1, c2+4·c3, ...]  (weights [1,4] → bits 1:0 and 3:2)
//   Then _mm_madd_epi16 to accumulate groups of four:
//     [c0+4·c1+16·c2+64·c3, ...]  (weights [1,16] → completes 2-bit packing)
//   Then _mm_shuffle_epi8 (SSSE3) to extract the lower byte of each int32.
//   Both SSSE3 instructions are available on all CPUs that support AVX2.
//
// RULE APPLICATION uses full 256-bit AVX2 (32 cells per register) on the
// unpacked sequential state and V byte arrays.

#include <immintrin.h>
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

// Unpack 16 packed bytes (64 cells) → 64 sequential state bytes.
// Uses 128-bit SSE2 operations to avoid AVX2's awkward cross-lane layout.
static inline void unpack64(const uint8_t* __restrict__ packed,
                              uint8_t* __restrict__       out)
{
    __m128i p = _mm_loadu_si128((const __m128i*)packed);
    const __m128i mask = _mm_set1_epi8(0x03);

    // Extract 4 stride-4 sub-arrays
    __m128i s0 = _mm_and_si128(p, mask);                              // cells 4k+0
    __m128i s1 = _mm_and_si128(_mm_srli_epi16(p, 2), mask);          // cells 4k+1
    __m128i s2 = _mm_and_si128(_mm_srli_epi16(p, 4), mask);          // cells 4k+2
    __m128i s3 = _mm_and_si128(_mm_srli_epi16(p, 6), mask);          // cells 4k+3

    // Round 1: interleave pairs within each 128-bit register
    //   t01_lo = [c0,c1, c4,c5, c8,c9, c12,c13, c16,c17, c20,c21, c24,c25, c28,c29]
    //   t23_lo = [c2,c3, c6,c7, c10,c11, c14,c15, c18,c19, c22,c23, c26,c27, c30,c31]
    __m128i t01_lo = _mm_unpacklo_epi8(s0, s1);
    __m128i t01_hi = _mm_unpackhi_epi8(s0, s1);
    __m128i t23_lo = _mm_unpacklo_epi8(s2, s3);
    __m128i t23_hi = _mm_unpackhi_epi8(s2, s3);

    // Round 2: interleave 16-bit pairs → sequential groups of 4
    //   r0 = [c0,c1,c2,c3, c4,c5,c6,c7, ..., c12,c13,c14,c15]
    _mm_storeu_si128((__m128i*)(out),     _mm_unpacklo_epi16(t01_lo, t23_lo));
    _mm_storeu_si128((__m128i*)(out+16),  _mm_unpackhi_epi16(t01_lo, t23_lo));
    _mm_storeu_si128((__m128i*)(out+32),  _mm_unpacklo_epi16(t01_hi, t23_hi));
    _mm_storeu_si128((__m128i*)(out+48),  _mm_unpackhi_epi16(t01_hi, t23_hi));
}

// Pack 64 sequential state bytes → 16 packed bytes.
// Processes 16 sequential states → 4 packed bytes per inner iteration (×4 = 64 cells).
//
// _mm_maddubs_epi16(a, b): result[k] = uint8(a[2k])*int8(b[2k]) + uint8(a[2k+1])*int8(b[2k+1])
//   With b=[1,4,1,4,...]:  result[k] = c[2k]*1 + c[2k+1]*4  (= bits 1:0 and 3:2 accumulated)
//
// _mm_madd_epi16(a, b): result[k] = int16(a[2k])*int16(b[2k]) + int16(a[2k+1])*int16(b[2k+1])
//   With b=[1,16,1,16,...]: result[k] = pair[2k]*1 + pair[2k+1]*16
//                         = c0+4c1+16c2+64c3  (complete 8-bit packed byte, in int32)
//
// _mm_shuffle_epi8 with mask [0,4,8,12,...]: extracts byte 0 of each int32 → the 4 packed bytes.
static inline void pack64(const uint8_t* __restrict__ states,
                            uint8_t* __restrict__       packed)
{
    const __m128i mul1 = _mm_set1_epi16(0x0401);         // bytes: [1, 4, 1, 4, ...]
    const __m128i mul2 = _mm_set1_epi32(0x00100001);      // int16: [1, 16, 1, 16, ...]
    // Shuffle: extract byte 0 of each of the 4 int32 dwords → positions 0,1,2,3
    const __m128i shuf = _mm_set_epi8(-1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 12,8,4,0);

    for (int i = 0; i < 4; i++) {
        __m128i s = _mm_loadu_si128((const __m128i*)(states + i * 16));

        // Step 1: accumulate adjacent pairs [c0+4c1, c2+4c3, ...]  (→ uint16, max 15)
        __m128i pairs = _mm_maddubs_epi16(s, mul1);

        // Step 2: accumulate adjacent pair-sums [c0+4c1+16c2+64c3, ...]  (→ int32, max 255)
        __m128i quads = _mm_madd_epi16(pairs, mul2);

        // Step 3: extract lower byte of each int32 → 4 packed bytes at positions 0..3
        __m128i result = _mm_shuffle_epi8(quads, shuf);

        // Write 4 bytes (result[0..3] are the packed bytes; result[4..15] are 0)
        *reinterpret_cast<uint32_t*>(packed + i * 4) =
            (uint32_t)_mm_cvtsi128_si32(result);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Row-sum computation
// ─────────────────────────────────────────────────────────────────────────────

static inline void unpack_adult_padded(const uint8_t* __restrict__ packed_row,
                                        uint8_t* __restrict__       ap,
                                        int N)
{
    auto adult_at = [&](int x) -> uint8_t {
        return ((packed_row[x >> 2] >> ((x & 3) * 2)) & 0x03u) == 3u ? 1u : 0u;
    };
    ap[0]   = adult_at(N-2);
    ap[1]   = adult_at(N-1);
    ap[N+2] = adult_at(0);
    ap[N+3] = adult_at(1);

    const __m128i mask    = _mm_set1_epi8(0x03);
    const __m128i k3_128  = _mm_set1_epi8(3);
    const __m128i k1_128  = _mm_set1_epi8(1);

    int x = 0;
    for (; x + 64 <= N; x += 64) {
        __m128i p = _mm_loadu_si128((const __m128i*)(packed_row + (x >> 2)));

        // Extract 4 stride-4 sub-arrays and detect adults (state==3 → 0xFF; AND 1 → 0/1)
        __m128i a0 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(p, mask),                  k3_128), k1_128);
        __m128i a1 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_srli_epi16(p,2),mask), k3_128), k1_128);
        __m128i a2 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_srli_epi16(p,4),mask), k3_128), k1_128);
        __m128i a3 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_srli_epi16(p,6),mask), k3_128), k1_128);

        // Interleave adult flags to sequential order (same two-round SSE2 unpack)
        __m128i t01_lo = _mm_unpacklo_epi8(a0, a1);
        __m128i t01_hi = _mm_unpackhi_epi8(a0, a1);
        __m128i t23_lo = _mm_unpacklo_epi8(a2, a3);
        __m128i t23_hi = _mm_unpackhi_epi8(a2, a3);

        _mm_storeu_si128((__m128i*)(ap+2+x),    _mm_unpacklo_epi16(t01_lo, t23_lo));
        _mm_storeu_si128((__m128i*)(ap+2+x+16), _mm_unpackhi_epi16(t01_lo, t23_lo));
        _mm_storeu_si128((__m128i*)(ap+2+x+32), _mm_unpacklo_epi16(t01_hi, t23_hi));
        _mm_storeu_si128((__m128i*)(ap+2+x+48), _mm_unpackhi_epi16(t01_hi, t23_hi));
    }
    for (; x < N; ++x) ap[2+x] = adult_at(x);
}

static inline void sliding_sum5(const uint8_t* __restrict__ ap,
                                  uint8_t* __restrict__       rs, int N)
{
    int x = 0;
    for (; x + 32 <= N; x += 32) {
        __m256i s = _mm256_add_epi8(
            _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(ap+x)),
                            _mm256_loadu_si256((const __m256i*)(ap+x+1))),
            _mm256_add_epi8(
                _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(ap+x+2)),
                                _mm256_loadu_si256((const __m256i*)(ap+x+3))),
                _mm256_loadu_si256((const __m256i*)(ap+x+4))));
        _mm256_storeu_si256((__m256i*)(rs+x), s);
    }
    for (; x < N; ++x) rs[x] = ap[x]+ap[x+1]+ap[x+2]+ap[x+3]+ap[x+4];
}

// ─────────────────────────────────────────────────────────────────────────────
//  V maintenance
// ─────────────────────────────────────────────────────────────────────────────

static inline void add_rows(uint8_t* __restrict__       V,
                             const uint8_t* __restrict__ rs, int N)
{
    int x = 0;
    for (; x + 32 <= N; x += 32)
        _mm256_storeu_si256((__m256i*)(V+x),
            _mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(V+x)),
                            _mm256_loadu_si256((const __m256i*)(rs+x))));
    for (; x < N; ++x) V[x] = (uint8_t)(V[x] + rs[x]);
}

static inline void slide_V(uint8_t* __restrict__       V,
                            const uint8_t* __restrict__ rs_new,
                            const uint8_t* __restrict__ rs_old, int N)
{
    int x = 0;
    for (; x + 32 <= N; x += 32) {
        __m256i v=_mm256_loadu_si256((const __m256i*)(V+x));
        __m256i n=_mm256_loadu_si256((const __m256i*)(rs_new+x));
        __m256i o=_mm256_loadu_si256((const __m256i*)(rs_old+x));
        _mm256_storeu_si256((__m256i*)(V+x), _mm256_add_epi8(_mm256_sub_epi8(v,o),n));
    }
    for (; x < N; ++x) V[x] = (uint8_t)(V[x]-rs_old[x]+rs_new[x]);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rule application
// ─────────────────────────────────────────────────────────────────────────────

static inline __m256i in_range_u8(const __m256i v, const __m256i base,
                                   const __m256i width)
{
    __m256i d = _mm256_sub_epi8(v, base);
    return _mm256_cmpeq_epi8(_mm256_min_epu8(d, width), d);
}

static inline void apply_rules_packed(
    const uint8_t* __restrict__ cur_packed,
    const uint8_t* __restrict__ V,
    uint8_t* __restrict__       nxt_packed, int N)
{
    const __m256i k0=_mm256_setzero_si256(), k1=_mm256_set1_epi8(1);
    const __m256i k2=_mm256_set1_epi8(2),    k3=_mm256_set1_epi8(3);
    const __m256i b3=_mm256_set1_epi8(3), w35=_mm256_set1_epi8(2);
    const __m256i b4=_mm256_set1_epi8(4), w49=_mm256_set1_epi8(5);

    alignas(64) uint8_t states[64];
    alignas(64) uint8_t next_s[64];

    for (int x = 0; x < N; x += 64) {
        unpack64(cur_packed + (x >> 2), states);

        // Apply rules using AVX2 (32 cells per register, two iterations per chunk)
        for (int i = 0; i < 64; i += 32) {
            __m256i A = _mm256_loadu_si256((const __m256i*)(V+x+i));
            __m256i c = _mm256_loadu_si256((const __m256i*)(states+i));

            A = _mm256_sub_epi8(A, _mm256_and_si256(_mm256_cmpeq_epi8(c,k3),k1));

            __m256i is0=_mm256_cmpeq_epi8(c,k0), is1=_mm256_cmpeq_epi8(c,k1);
            __m256i is2=_mm256_cmpeq_epi8(c,k2), is3=_mm256_cmpeq_epi8(c,k3);

            __m256i n0=_mm256_and_si256(_mm256_and_si256(is0,in_range_u8(A,b3,w35)),k1);
            __m256i n1=_mm256_and_si256(is1,k2);
            __m256i n2=_mm256_and_si256(is2,k3);
            __m256i n3=_mm256_and_si256(_mm256_and_si256(is3,in_range_u8(A,b4,w49)),k3);

            _mm256_storeu_si256((__m256i*)(next_s+i),
                _mm256_or_si256(_mm256_or_si256(n0,n1),_mm256_or_si256(n2,n3)));
        }

        pack64(next_s, nxt_packed + (x >> 2));
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
    const int N   = wc->N;
    const int NPK = N >> 2;

    const int base    = N / nthreads;
    const int rem     = N % nthreads;
    const int y_start = tid * base + std::min(tid, rem);
    const int y_end   = y_start + base + (tid < rem ? 1 : 0);

    std::vector<uint8_t> ring_buf((size_t)RING * N, 0u);
    std::vector<uint8_t> V  (N, 0u);
    std::vector<uint8_t> ap (static_cast<size_t>(N) + 4, 0u);

    auto ring_slot = [&](int d) -> uint8_t* {
        return ring_buf.data() + (size_t)(d & RMASK) * N;
    };

    for (int gen = 0; gen < wc->generations; ++gen) {
        const uint8_t* cur = wc->grids[ gen      & 1];
        uint8_t*       nxt = wc->grids[(gen + 1) & 1];

        std::fill(V.begin(), V.end(), 0u);
        for (int d = y_start - 2; d <= y_start + 2; ++d) {
            int gr = ((d % N) + N) % N;
            uint8_t* slot = ring_slot(d);
            unpack_adult_padded(cur + (size_t)gr * NPK, ap.data(), N);
            sliding_sum5(ap.data(), slot, N);
            add_rows(V.data(), slot, N);
        }

        for (int y = y_start; y < y_end; ++y) {
            apply_rules_packed(cur + (size_t)y * NPK, V.data(),
                               nxt + (size_t)y * NPK, N);

            int      new_gr = (y + 3) % N;
            uint8_t* rs_new = ring_slot(y + 3);
            unpack_adult_padded(cur + (size_t)new_gr * NPK, ap.data(), N);
            sliding_sum5(ap.data(), rs_new, N);
            slide_V(V.data(), rs_new, ring_slot(y - 2), N);
        }

        bar->arrive_and_wait();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Grid pack / unpack  (I/O only, not per-generation)
// ─────────────────────────────────────────────────────────────────────────────

static void pack_grid(const uint8_t* bytes, uint8_t* packed, size_t cells)
{
    size_t i = 0;
    for (; i + 16 <= cells; i += 16) {
        // Reuse the same pack64 infrastructure: process 16 cells → 4 bytes
        __m128i s = _mm_loadu_si128((const __m128i*)(bytes + i));
        const __m128i mul1 = _mm_set1_epi16(0x0401);
        const __m128i mul2 = _mm_set1_epi32(0x00100001);
        const __m128i shuf = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,12,8,4,0);
        __m128i q = _mm_shuffle_epi8(_mm_madd_epi16(_mm_maddubs_epi16(s, mul1), mul2), shuf);
        *reinterpret_cast<uint32_t*>(packed + (i >> 2)) = (uint32_t)_mm_cvtsi128_si32(q);
    }
    for (; i < cells; i += 4)
        packed[i>>2] = (bytes[i]&3u)|((bytes[i+1]&3u)<<2)|((bytes[i+2]&3u)<<4)|((bytes[i+3]&3u)<<6);
}

static void unpack_grid(const uint8_t* packed, uint8_t* bytes, size_t cells)
{
    size_t i = 0;
    for (; i + 64 <= cells; i += 64) {
        __m128i p = _mm_loadu_si128((const __m128i*)(packed + (i>>2)));
        const __m128i mask = _mm_set1_epi8(0x03);
        __m128i s0=_mm_and_si128(p,mask);
        __m128i s1=_mm_and_si128(_mm_srli_epi16(p,2),mask);
        __m128i s2=_mm_and_si128(_mm_srli_epi16(p,4),mask);
        __m128i s3=_mm_and_si128(_mm_srli_epi16(p,6),mask);
        __m128i t01_lo=_mm_unpacklo_epi8(s0,s1), t01_hi=_mm_unpackhi_epi8(s0,s1);
        __m128i t23_lo=_mm_unpacklo_epi8(s2,s3), t23_hi=_mm_unpackhi_epi8(s2,s3);
        _mm_storeu_si128((__m128i*)(bytes+i),    _mm_unpacklo_epi16(t01_lo,t23_lo));
        _mm_storeu_si128((__m128i*)(bytes+i+16), _mm_unpackhi_epi16(t01_lo,t23_lo));
        _mm_storeu_si128((__m128i*)(bytes+i+32), _mm_unpacklo_epi16(t01_hi,t23_hi));
        _mm_storeu_si128((__m128i*)(bytes+i+48), _mm_unpackhi_epi16(t01_hi,t23_hi));
    }
    for (; i < cells; i+=4) {
        uint8_t b=packed[i>>2];
        bytes[i]=(b)&3u; bytes[i+1]=(b>>2)&3u; bytes[i+2]=(b>>4)&3u; bytes[i+3]=(b>>6)&3u;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,"Usage: %s <input.bin> <output.bin> [generations]\n",argv[0]);
        return 1;
    }
    int generations = 10000;
    if (argc == 4) {
        char* end; long g = std::strtol(argv[3],&end,10);
        if (*end||g<=0){std::fprintf(stderr,"Error: bad generations\n");return 1;}
        generations = (int)g;
    }

    FILE* fin = std::fopen(argv[1],"rb");
    if (!fin){std::fprintf(stderr,"Error: cannot open '%s'\n",argv[1]);return 2;}

    uint64_t width=0,height=0;
    if (std::fread(&width,8,1,fin)!=1||std::fread(&height,8,1,fin)!=1||
        width==0||width!=height) {
        std::fprintf(stderr,"Error: invalid header\n"); std::fclose(fin); return 3;
    }

    const int    N      = (int)width;
    const size_t CELLS  = (size_t)N*N;
    const size_t PACKED = CELLS>>2;

    auto alloc=[](size_t n)->uint8_t*{
        void*p=nullptr;
        if(posix_memalign(&p,64,n)!=0||!p){std::fprintf(stderr,"Fatal: alloc\n");std::exit(1);}
        return (uint8_t*)p;
    };

    std::vector<uint8_t> input_bytes(CELLS);
    if (std::fread(input_bytes.data(),1,CELLS,fin)!=CELLS) {
        std::fprintf(stderr,"Error: truncated\n"); std::fclose(fin); return 4;
    }
    std::fclose(fin);

    uint8_t* grid_a = alloc(PACKED);
    uint8_t* grid_b = alloc(PACKED);
    pack_grid(input_bytes.data(), grid_a, CELLS);
    input_bytes.clear();

    WorkCtx wc{{grid_a,grid_b},N,generations};
    std::barrier<> bar(NTHREADS);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(NTHREADS-1);
    for (int t=1;t<NTHREADS;++t)
        threads.emplace_back(worker,t,NTHREADS,&wc,&bar);
    worker(0,NTHREADS,&wc,&bar);
    for (auto& th:threads) th.join();
    auto t1 = std::chrono::steady_clock::now();
    std::printf("%.3f ms\n", std::chrono::duration<double,std::milli>(t1-t0).count());

    std::vector<uint8_t> output_bytes(CELLS);
    unpack_grid(wc.grids[generations&1], output_bytes.data(), CELLS);

    FILE* fout = std::fopen(argv[2],"wb");
    if (!fout){std::fprintf(stderr,"Error: cannot open '%s'\n",argv[2]);return 5;}
    const bool ok=(std::fwrite(&width,8,1,fout)==1)&&(std::fwrite(&height,8,1,fout)==1)&&
                  (std::fwrite(output_bytes.data(),1,CELLS,fout)==CELLS);
    std::fclose(fout);
    if(!ok){std::fprintf(stderr,"Error: write\n");return 6;}

    std::free(grid_a); std::free(grid_b);
    return 0;
}