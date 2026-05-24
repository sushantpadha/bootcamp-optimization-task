# Dev Environment — c8g.2xlarge Setup

Your dev instance is an AWS `c8g.2xlarge` running Ubuntu 24.04 LTS. 

## Hardware

| Property      | Value |
|---------------|-------|
| Instance type | c8g.2xlarge |
| vCPUs         | 8 (no hyperthreading) |
| CPU           | AWS Graviton4, ARM Neoverse-V2, ARMv9.0-a |
| RAM           | 16 GiB |
| Storage       | NVMe SSD |
| Architecture  | aarch64 |


## Installed Software

This is your instance. Install any required software

## Building

From your project root:

```bash
bash reference/build.sh          # build the reference
bash build.sh                    # build your submission
```

The build script compiles to a binary named `spawn_sim` in the project root.

## Running the Harness

```bash
# Single run (quick correctness check)
bash harness/run.sh -n 1 ./spawn_sim \
    test_grids/public_1_random_low.bin \
    test_grids/public_1_random_low.expected.bin

# Full grading run (10 iterations, statistics)
bash harness/run.sh -n 10 ./spawn_sim \
    test_grids/public_1_random_low.bin \
    test_grids/public_1_random_low.expected.bin
```

The harness requires `sudo` to drop filesystem caches between runs. If sudo is
not available, it will still run but will print a warning about cache warmth.

To grant passwordless sudo for cache drops only:

```bash
echo "ubuntu ALL=(ALL) NOPASSWD: /bin/sh -c echo 3 > /proc/sys/vm/drop_caches" \
    | sudo tee /etc/sudoers.d/drop_caches
```

## CPU Frequency

The c8g.2xlarge instance runs at a fixed hardware frequency — AWS Graviton does not
expose the cpufreq sysfs interface, so `/sys/devices/system/cpu/cpu0/cpufreq/` does
not exist. This is normal and expected. There is no dynamic frequency scaling to worry
about; your benchmark timings are stable by default.

## Profiling Tools

Basic profiling with `perf`:

```bash
# Record a profile
sudo perf record -g -F 1000 \
    taskset -c 0-7 ./spawn_sim test_grids/public_1_random_low.bin /dev/null

# View report
sudo perf report

# Hardware counters (cache misses, IPC, etc.)
sudo perf stat -e cycles,instructions,cache-misses,cache-references \
    taskset -c 0-7 ./spawn_sim test_grids/public_1_random_low.bin /dev/null
```

## Storage Notes

The double-buffered grid is 2 × 1 GiB ≈ 2 GiB. On 16 GiB RAM, this is
comfortable, but keep an eye on:

- OS page cache from repeated runs
- Your own scratch buffers if you add any
- The output file (another 1 GiB)

Use `free -h` and `vmstat 1` to monitor memory pressure.
