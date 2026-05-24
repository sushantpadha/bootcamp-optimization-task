// spawn_sim.cpp — Reference implementation of the Monster Spawning Grid.
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
//   Prints the wall-clock time of the simulation (excluding file I/O) to stdout
//   as a single line:  "<N.NNN> ms"
//   Only the simulation loop is timed; reading and writing files are not counted.
//
// EXIT CODES:
//   0  success
//   1  wrong number of arguments
//   2  cannot open input file
//   3  input header invalid (not square, or zero dimensions)
//   4  input file too short (cell data truncated)
//   5  cannot open output file
//   6  write error on output file

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int RANGE = 2;    // range-2 Moore neighbourhood

static const uint8_t EMPTY    = 0;
static const uint8_t EGG      = 1;
static const uint8_t JUVENILE = 2;
static const uint8_t ADULT    = 3;

// Count ADULT cells in the range-2 Moore neighbourhood of (cx, cy).
// The neighbourhood is the 24 cells in the 5×5 block centred on (cx, cy),
// excluding (cx, cy) itself. The grid wraps toroidally.
static int count_adults(const std::vector<uint8_t>& grid, int grid_size, int cx, int cy)
{
    int count = 0;
    for (int dy = -RANGE; dy <= RANGE; ++dy) {
        int ny = (cy + dy + grid_size) % grid_size;
        for (int dx = -RANGE; dx <= RANGE; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = (cx + dx + grid_size) % grid_size;
            if (grid[(size_t)ny * grid_size + nx] == ADULT) ++count;
        }
    }
    return count;
}

// Apply one step of the Monster Spawning Grid rules.
// Reads from src, writes to dst. All cells transition simultaneously.
static void step(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int grid_size)
{
    for (int y = 0; y < grid_size; ++y) {
        for (int x = 0; x < grid_size; ++x) {
            int A = count_adults(src, grid_size, x, y);
            uint8_t cell = src[(size_t)y * grid_size + x];
            uint8_t next;
            switch (cell) {
                case EMPTY:    next = (A >= 3 && A <= 5) ? EGG      : EMPTY; break;
                case EGG:      next = JUVENILE;                               break;
                case JUVENILE: next = ADULT;                                  break;
                case ADULT:    next = (A >= 4 && A <= 9) ? ADULT    : EMPTY; break;
                default:       next = EMPTY;                                  break;
            }
            dst[(size_t)y * grid_size + x] = next;
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
        char* end;
        long g = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || g <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        generations = (int)g;
    }

    // -------------------------------------------------------------------------
    // Read input
    // -------------------------------------------------------------------------
    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]);
        return 2;
    }

    uint64_t width, height;
    if (std::fread(&width,  sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, fin) != 1) {
        std::fprintf(stderr, "Error: input file too short (cannot read header)\n");
        std::fclose(fin);
        return 3;
    }

    if (width == 0 || width != height) {
        std::fprintf(stderr,
            "Error: grid must be square and non-empty, got %" PRIu64 " × %" PRIu64 "\n",
            width, height);
        std::fclose(fin);
        return 3;
    }

    int grid_size = (int)width;
    const size_t N = (size_t)grid_size * grid_size;

    std::vector<uint8_t> grid_a(N);
    if (std::fread(grid_a.data(), 1, N, fin) != N) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    // -------------------------------------------------------------------------
    // Simulate
    // -------------------------------------------------------------------------
    std::vector<uint8_t> grid_b(N);

    std::vector<uint8_t>* cur  = &grid_a;
    std::vector<uint8_t>* next = &grid_b;

    auto t0 = std::chrono::steady_clock::now();

    for (int gen = 0; gen < generations; ++gen) {
        step(*cur, *next, grid_size);
        std::swap(cur, next);
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%.3f ms\n", elapsed_ms);

    // -------------------------------------------------------------------------
    // Write output
    // -------------------------------------------------------------------------
    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }

    if (std::fwrite(&width,       sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(&height,      sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(cur->data(),  1, N, fout) != N) {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(fout);
        return 6;
    }

    std::fclose(fout);
    return 0;
}
