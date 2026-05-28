# Guide to `spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp`

This note explains **only the current file**:

- [spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:1)

The goal is to let a new person understand the structure and the key performance idea without needing any prior context.

## What the program does

The program simulates a square grid over many generations.

Each cell is in one of four states:

- `EMPTY`
- `EGG`
- `JUVENILE`
- `ADULT`

The rules that matter in this file are:

- A `JUVENILE` always becomes an `ADULT` next generation.
- A new `EGG` can be created only in an `EMPTY` cell.
- The birth/survival decisions depend only on the number of nearby `ADULT` cells in a **5 x 5 neighborhood** centered on the cell.
- The center cell itself is not counted as a neighbor for adult survival in the logical rule, but the hot kernel intentionally keeps the center bit inside its running total and compensates for that in the threshold logic.

In plain English:

- Existing adults survive if they have **4 through 9 adult neighbors**.
- Empty cells produce a new egg if they have **3 through 5 adult neighbors**.
- Juveniles promote automatically.

## The main performance idea

The expensive part is counting nearby adults for every cell.

This file avoids repeating the same work over and over by doing two things:

1. It processes **64 cells at a time** using one `uint64_t` word.
2. It reuses work when moving from one output row to the next.

The key observation is:

- When you compute the next state for row `r`, you need source rows `r-2` through `r+2`.
- When you move to row `r+1`, you need source rows `r-1` through `r+3`.
- That means 4 of the 5 source rows are the same.

So instead of rebuilding the full 5-row neighborhood count from scratch for every row, the code:

- keeps a cached summary for the current 5 source rows
- keeps a rolling total for the current output row
- moves down by **removing the top outgoing source row** and **adding the new bottom incoming source row**

That rolling update is the central optimization in this file.

## How the grid is represented

Read:

- [BitPlane](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:92)

Each grid state is stored as a flat array of `uint64_t` words.

- One bit = one cell
- One `uint64_t` = 64 cells in the same row
- One row contains `grid_size / 64` words

The program keeps separate bit-planes for:

- current adults
- current juveniles
- current eggs
- next adults
- next eggs

There is **no separate next-juveniles array**. That is intentional.

At the end of each generation, the planes are rotated like this:

- `next_adults` becomes the new `current_adults`
- `current_eggs` becomes the new `current_juveniles`
- `next_eggs` becomes the new `current_eggs`

So the model is:

- eggs age into juveniles one generation later
- juveniles age into adults one generation later

## The most important idea: counting many cells in parallel

The code does **not** keep one integer neighbor count per cell.

Instead, it uses several `uint64_t` masks to represent the bits of those counts for 64 cells at once.

Example:

- suppose you have 64 cells packed into one word
- for each of those 64 cells, you want a count like `0`, `1`, `2`, ..., `25`
- instead of storing 64 separate integers, the code stores the binary digits of all 64 counters in parallel

So:

- `b0` holds the lowest bit of the count for all 64 cells
- `b1` holds the next bit
- `b2`, `b3`, `b4` hold the higher bits

This lets the code update many counters with bitwise logic instead of scalar loops.

## The two-step counting strategy

The file counts nearby adults in two stages.

### Stage 1: summarize one source row horizontally

For one source row, and for each 64-cell word in that row, the code computes:

- for every cell lane in that word,
- how many adults are present in the 5 horizontal positions:
  - column `c-2`
  - column `c-1`
  - column `c`
  - column `c+1`
  - column `c+2`

That number is between `0` and `5`, so it fits in **3 bits**.

This 3-bit result is stored in:

- [HorizontalPartialWord](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:107)

and a full source row of those summaries is stored in:

- [HorizontalPartialRow](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:116)

Important detail:

- this file stores the three bits for one word **together** as a small struct:
  - `b0`
  - `b1`
  - `b2`

That is helpful because the hot rolling update later wants all three bits of the same word together.

### Stage 2: combine five source rows vertically

Once you have those 3-bit row summaries for five consecutive source rows, you add them together.

Now the total count is between `0` and `25`, so it needs **5 bits**.

That rolling 5-bit total for one output row is stored in:

- [VerticalAccumulatorRow](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:127)

The code keeps one such rolling total and updates it row-by-row as it moves downward.

## The scratch space used by one worker

Read:

- [HotRowScratch](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:146)

Each worker thread gets its own scratch object containing:

- a ring of 5 cached source-row summaries
- one temporary slot for the new incoming source-row summary
- one rolling 5-bit total for the current output row

Why 5 cached source rows?

Because an output row depends on exactly these 5 source rows:

- `r-2`
- `r-1`
- `r`
- `r+1`
- `r+2`

When moving from row `r` to row `r+1`:

- `r-2` drops out
- `r+3` becomes the new incoming row

So the ring buffer always represents the current 5-row vertical window.

## The hot interior kernel

Read this function first:

- [run_one_generation_hot_rows](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:607)

This is the main performance-critical function.

It only handles the easy interior region:

- rows `2` through `grid_size - 3`
- words `1` through `words_per_row - 2`

That means:

- no top/bottom wraparound
- no left/right edge masking

Those awkward edge cases are handled elsewhere so this hot function can stay clean and fast.

### English pseudocode for the hot kernel

```text
for each worker-owned chunk of rows:
    decide the interior rows that are safe to process with the fast path

    compute the 5 cached source-row summaries needed for the first output row
    build the first rolling 5-bit total from those 5 cached rows

    for each interior output row except the last one:
        compute the summary for the new incoming source row

        for each interior word in the row:
            read the current 5-bit total for this word
            produce the next-state output bits for this word
            subtract the outgoing cached source-row summary from the total
            add the incoming source-row summary to the total
            write the updated total back

        replace the old cached row with the newly computed one
        advance the logical head of the 5-row ring buffer

    emit the final interior row using the current rolling total
```

That is the whole optimization in one pass:

- build once
- emit one output row
- roll forward by remove/add
- repeat

## How one source-row summary is computed

Read:

- [compute_horizontal_partial_row](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:513)

This function takes one source row of adult bits and creates the 3-bit horizontal count summary for each word in that row.

For each word, it forms five aligned masks:

- two cells to the left
- one cell to the left
- the center
- one cell to the right
- two cells to the right

Because 64 cells are packed into one word, the left and right shifts sometimes need bits from the previous or next word.

That is why the function keeps:

- `prev`
- `curr`
- `next`

for the current sliding position.

### English pseudocode for this function

```text
for each interior word in the source row:
    build 5 aligned adult masks:
        left by 2
        left by 1
        center
        right by 1
        right by 2

    combine those 5 one-bit masks into a 3-bit count
    store the 3 bits for this word together

    slide forward one word:
        previous = current
        current = next
        next = load the next word from memory
```

The actual “combine those 5 masks into a 3-bit count” logic uses carry-save-adder style bit tricks:

- [carry_save_add](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:280)

You do **not** need to understand every boolean identity immediately. The important mental model is:

- each of the 5 masks says “which of the 64 cells have an adult in this offset position”
- the function merges those 5 masks into a 3-bit count for all 64 cells at once

## How the first rolling total is built

Read:

- [build_vertical_accumulator_row](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:565)

This function is simple:

- take the 5 cached source-row summaries
- for each word
- add those five 3-bit values together
- store the 5-bit result into the rolling total arrays

This happens only once for the first output row in the chunk.

After that, the code updates the total incrementally instead of rebuilding it.

## How the rolling update works

Inside the hot kernel, after one output row is emitted, the total is updated like this:

```text
new total
    = old total
    - summary of the source row that just left the 5-row window
    + summary of the new source row that just entered the 5-row window
```

The helpers that do that are:

- [subtract_horizontal_partial_from_total](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:437)
- [add_horizontal_partial_to_total](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:359)

This is where most of the saved work comes from.

Without the rolling update, the code would have to rebuild the 5-row total again for every output row.

## How the file turns a total count into output states

Read:

- [store_hot_result_from_total_with_center](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:487)

Inputs:

- the 5-bit total count for 64 cells
- the current adult bits for those cells
- the current juvenile bits
- the current egg bits

Outputs:

- the next adult bits
- the next egg bits

Important detail:

- in the hot path, the rolling total still includes the center adult bit
- so the threshold masks inside this function are written for that representation:
  - an adult survives when the running total is `5` through `10`, because that total includes the adult itself
  - an empty cell creates an egg when the running total is `3` through `5`, because the center bit is zero there anyway

That is why the threshold names inside the function look slightly shifted from the plain logical rule.

You can treat this function as:

```text
take the count representation
turn it into boolean masks like "count is at least X"
apply the adult survival rule
apply the empty-cell egg creation rule
```

## Why the edges are handled separately

Read:

- [initialize_next_generation_rows](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:741)
- [run_one_generation_left_edge_valid_rows](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:778)
- [run_one_generation_right_edge_valid_rows](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:945)
- [run_one_generation_wrapping_rows](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:1112)

The interior fast path assumes:

- there is a full 5-row neighborhood above and below
- there are neighboring words on the left and right

That is false near:

- the top 2 rows
- the bottom 2 rows
- the leftmost 2 columns
- the rightmost 2 columns

So the file splits the work:

- fast bit-sliced rolling kernel for the safe interior
- separate code for left/right valid edges
- simple wrapped scalar logic for the truly awkward wrapping rows and the first/last 2 columns

This is a common performance pattern:

- keep the hot path simple
- push awkward corner cases into separate, colder code

## What happens in one worker for one generation

Read:

- [process_generation_chunk](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:1302)

This function is the best “one-screen summary” of the per-generation logic.

### English pseudocode

```text
prepare next-state rows for the worker's owned chunk:
    juveniles are copied into next_adults
    edge words are initialized safely

run the fast interior kernel on the safe middle region

run the separate edge handlers for:
    left edge
    right edge
    top/bottom wrapping rows and corner cells
```

This is the function to read if you want to understand how the file is divided into hot code and edge code.

## How the threading works

Read:

- [compute_row_range](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:159)
- [main](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:1374)

The threading model is straightforward:

- up to 8 worker threads
- fixed ownership of row ranges
- one thread per CPU core
- worker threads are pinned to cores
- the main thread also acts as a worker
- threads are created after timing starts and then reused across generations

Per generation:

1. all workers wait at the start barrier
2. each worker processes its own rows
3. all workers wait at the done barrier
4. the bit-planes are rotated for the next generation

### English pseudocode for the outer loop

```text
pack input bytes into bit-planes
start timing
create worker threads

for each generation:
    release all workers
    main thread processes its own row chunk too
    wait until all workers finish
    rotate the planes so next becomes current

stop workers
join threads
stop timing
unpack bit-planes back into output bytes
```

## The best order to read the file

If you want the shortest path to understanding:

1. [process_generation_chunk](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:1302)
2. [run_one_generation_hot_rows](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:607)
3. [compute_horizontal_partial_row](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:513)
4. [build_vertical_accumulator_row](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:565)
5. [store_hot_result_from_total_with_center](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:487)
6. [main](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:1374)

If you read just those six entry points and keep the pseudocode above in mind, the rest of the file becomes much easier to follow.

## What is most likely to matter for future optimization

If someone wants to optimize this file further, the most important places to look are:

- [compute_horizontal_partial_row](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:513)
  This still does a lot of work per word.

- [run_one_generation_hot_rows](/home/sushil/Files/bootcamp/bootcamp-optimization-task-sushil-branch/reference/spawn_sim_csa_tree_sliding_window_threadpool_rowcache_rolling_aos.cpp:607)
  This is the core rolling kernel and still moves a lot of running-count state.

Everything else is secondary unless the optimization changes the whole architecture.
