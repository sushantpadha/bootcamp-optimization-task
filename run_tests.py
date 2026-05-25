#!/usr/bin/env python3
"""
Test runner for Monster Spawning Grid implementations.

Usage:
    python3 run_tests.py <cpp_file> [max_grid_size]

Arguments:
    cpp_file          Path to the C++ file to test (will be compiled)
    max_grid_size     Maximum grid size to test (default: largest available)
                      Common sizes: 512, 2048, 8192, 32768

The script will:
    1. Compile the C++ file
    2. Find all .bin test files up to max_grid_size
    3. Run the compiled binary on each test file
    4. Verify output matches .expected.bin (if it exists)
    5. Log results to logs/ folder with:
       - Test name and status (PASS/FAIL)
       - Elapsed time reported by the program
       - Total execution time
"""

import os
import sys
import subprocess
import struct
import json
import tempfile
from pathlib import Path
from datetime import datetime
import time

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
TEST_GRIDS_DIR = SCRIPT_DIR / "test_grids"
LOGS_DIR = SCRIPT_DIR / "logs"
BUILD_DIR = SCRIPT_DIR / "build"
SRC_DIR = SCRIPT_DIR / "src"

# Available grid sizes in test_grids (by examining .bin files)
AVAILABLE_SIZES = [512, 2048, 8192, 32768]


def parse_arguments():
    """Parse and validate command-line arguments."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cpp_file> [max_grid_size] [build_script]", file=sys.stderr)
        print(f"\nExample: {sys.argv[0]} my_solution.cpp 8192", file=sys.stderr)
        print(f"Example: {sys.argv[0]} my_solution.cpp 8192 ./build.sh", file=sys.stderr)
        sys.exit(1)

    cpp_file = Path(sys.argv[1])
    if not cpp_file.exists():
        print(f"Error: C++ file not found: {cpp_file}", file=sys.stderr)
        sys.exit(1)

    max_size = None
    if len(sys.argv) > 2:
        try:
            max_size = int(sys.argv[2])
            if max_size not in AVAILABLE_SIZES:
                print(f"Warning: Grid size {max_size} may not be available.", file=sys.stderr)
        except ValueError:
            print(f"Error: Invalid grid size: {sys.argv[2]}", file=sys.stderr)
            sys.exit(1)

    build_script = None
    if len(sys.argv) > 3:
        build_script = Path(sys.argv[3])
        if not build_script.exists():
            print(f"Error: Build script not found: {build_script}", file=sys.stderr)
            sys.exit(1)

    return cpp_file, max_size, build_script


def compile_cpp(cpp_file, build_script=None):
    """Compile the C++ file and return the executable path."""
    print(f"\n{'='*70}")
    print(f"COMPILING: {cpp_file}")
    print(f"{'='*70}")

    # Create build directory if it doesn't exist
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Create a unique binary name in the build folder
    cpp_path = cpp_file.resolve()
    import hashlib
    path_hash = hashlib.md5(str(cpp_path).encode()).hexdigest()[:8]
    executable = BUILD_DIR / f"{cpp_file.stem}_{path_hash}"

    # Determine which build script to use
    if build_script is None:
        # Look for build.sh in the same directory as cpp_file
        build_script = cpp_file.parent / "build.sh"
        if not build_script.exists():
            print(f"Error: No build script found!", file=sys.stderr)
            print(f"  Looked for: {build_script}", file=sys.stderr)
            print(f"  Use: {sys.argv[0]} <cpp_file> [max_grid_size] <build_script>", file=sys.stderr)
            sys.exit(1)

    # Run build script with OUTPUT environment variable
    try:
        env = os.environ.copy()
        env["OUTPUT"] = str(executable)
        
        result = subprocess.run(
            ["bash", str(build_script)],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            print(f"Compilation failed!", file=sys.stderr)
            print(f"STDERR: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        
        if not executable.exists():
            print(f"Compilation failed: executable not created at {executable}", file=sys.stderr)
            sys.exit(1)
        
        print(f"✓ Successfully compiled to: {executable}")
        return executable
    except subprocess.TimeoutExpired:
        print(f"Compilation timed out!", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Compilation error: {e}", file=sys.stderr)
        sys.exit(1)


def find_test_files(max_size):
    """
    Find all .bin test files up to max_size.
    Returns list of (test_name, input_file, expected_file or None).
    """
    tests = []

    if not TEST_GRIDS_DIR.exists():
        print(f"Error: test_grids directory not found: {TEST_GRIDS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Find all .bin files
    bin_files = sorted(TEST_GRIDS_DIR.glob("*.bin"))

    for bin_file in bin_files:
        # Skip .expected.bin files (we'll find them through .bin files)
        if bin_file.name.endswith(".expected.bin"):
            continue

        # Extract size from filename
        filename = bin_file.stem
        size = None
        for available_size in AVAILABLE_SIZES:
            if f"_{available_size}" in filename or filename.endswith(str(available_size)):
                size = available_size
                break

        # Skip if size exceeds max_size
        if max_size is not None and (size is None or size > max_size):
            continue

        test_name = bin_file.stem
        expected_file = bin_file.parent / f"{filename}.expected.bin"
        if not expected_file.exists():
            expected_file = None

        tests.append((test_name, bin_file, expected_file))

    if not tests:
        print(f"Error: No test files found in {TEST_GRIDS_DIR}", file=sys.stderr)
        sys.exit(1)

    return tests


def files_are_identical(file1, file2):
    """Check if two binary files are bit-identical."""
    try:
        with open(file1, "rb") as f1, open(file2, "rb") as f2:
            return f1.read() == f2.read()
    except Exception as e:
        print(f"Error comparing files: {e}", file=sys.stderr)
        return False


def run_test(executable, input_file, output_file, test_name):
    """
    Run a single test and return (success, elapsed_ms, execution_time_ms).
    - success: True if program ran successfully
    - elapsed_ms: Time printed by program (from stdout)
    - execution_time_ms: Total wall-clock time including I/O
    """
    print(f"\n  Running: {test_name}")

    start_time = time.time()
    try:
        result = subprocess.run(
            [str(executable), str(input_file), str(output_file)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        execution_time_ms = (time.time() - start_time) * 1000

        if result.returncode != 0:
            print(f"    ✗ Program exited with code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(f"      STDERR: {result.stderr}", file=sys.stderr)
            return False, None, execution_time_ms

        # Parse elapsed time from stdout
        elapsed_ms = None
        for line in result.stdout.strip().split("\n"):
            # Expected format: "123.456 ms"
            if "ms" in line:
                try:
                    elapsed_ms = float(line.split()[0])
                    break
                except (ValueError, IndexError):
                    pass

        if elapsed_ms is None:
            print(f"    ✗ Could not parse elapsed time from output", file=sys.stderr)
            print(f"      Program output: {result.stdout}", file=sys.stderr)
            return False, None, execution_time_ms

        return True, elapsed_ms, execution_time_ms

    except subprocess.TimeoutExpired:
        execution_time_ms = (time.time() - start_time) * 1000
        print(f"    ✗ Test timed out after {execution_time_ms:.0f}ms", file=sys.stderr)
        return False, None, execution_time_ms
    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        print(f"    ✗ Error running test: {e}", file=sys.stderr)
        return False, None, execution_time_ms


def run_all_tests(executable, tests, log_file):
    """
    Run all tests and write results to log file on the fly.
    Returns list of test result dictionaries.
    """
    results = []

    print(f"\n{'='*70}")
    print(f"RUNNING TESTS ({len(tests)} total)")
    print(f"{'='*70}")

    for i, (test_name, input_file, expected_file) in enumerate(tests, 1):
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            output_file = Path(tmp.name)

        try:
            # Run the test
            success, elapsed_ms, execution_time_ms = run_test(
                executable, input_file, output_file, test_name
            )

            result = {
                "test_name": test_name,
                "input_file": str(input_file),
                "output_file": str(output_file),
                "expected_file": str(expected_file) if expected_file else None,
                "success": success,
                "elapsed_ms": elapsed_ms,
                "execution_time_ms": execution_time_ms,
                "output_matches": None,
                "status": "SKIP"
            }

            if not success:
                result["status"] = "FAIL"
                status_icon = "✗"
            else:
                # Check if output matches expected
                if expected_file:
                    matches = files_are_identical(output_file, expected_file)
                    result["output_matches"] = matches
                    result["status"] = "PASS" if matches else "FAIL"
                    status_icon = "✓" if matches else "✗"
                else:
                    result["status"] = "SKIP"
                    status_icon = "⊘"

            # Write result to log file immediately
            with open(log_file, "a") as f:
                f.write(f"\n[{i}/{len(tests)}] {test_name}\n")
                f.write(f"  Status: {result['status']}\n")
                if result['elapsed_ms'] is not None:
                    f.write(f"  Elapsed: {result['elapsed_ms']:.3f} ms (total: {result['execution_time_ms']:.0f} ms)\n")
                if result['output_matches'] is not None:
                    f.write(f"  Output matches: {result['output_matches']}\n")

            # Brief console output
            print(f"  {status_icon} {i:2}/{len(tests)} {test_name:40}", end="")
            if result['elapsed_ms'] is not None:
                print(f" {result['elapsed_ms']:8.2f} ms")
            else:
                print()

            results.append(result)

        finally:
            # Clean up temp output file
            if output_file.exists():
                try:
                    output_file.unlink()
                except:
                    pass

    return results


def save_logs(results, cpp_file, log_file):
    """Append summary analysis to log file and save JSON."""
    # Prepare summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    total = len(results)

    # Append analysis to log file
    with open(log_file, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"ANALYSIS SUMMARY\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total Tests:  {total}\n")
        f.write(f"Passed:       {passed}\n")
        f.write(f"Failed:       {failed}\n")
        f.write(f"Skipped:      {skipped}\n\n")

        # Timing analysis
        test_with_timing = [r for r in results if r['elapsed_ms'] is not None]
        if test_with_timing:
            elapsed_times = [r['elapsed_ms'] for r in test_with_timing]
            execution_times = [r['execution_time_ms'] for r in test_with_timing]

            f.write(f"Timing Statistics (Program-reported simulation time):\n")
            f.write(f"  Min:     {min(elapsed_times):>10.2f} ms\n")
            f.write(f"  Max:     {max(elapsed_times):>10.2f} ms\n")
            f.write(f"  Average: {sum(elapsed_times)/len(elapsed_times):>10.2f} ms\n")
            f.write(f"  Total:   {sum(elapsed_times):>10.2f} ms\n\n")

            f.write(f"Execution Time (including I/O):\n")
            f.write(f"  Min:     {min(execution_times):>10.0f} ms\n")
            f.write(f"  Max:     {max(execution_times):>10.0f} ms\n")
            f.write(f"  Average: {sum(execution_times)/len(execution_times):>10.0f} ms\n\n")

        # Failed tests list
        if failed > 0:
            f.write(f"Failed Tests:\n")
            for r in results:
                if r["status"] == "FAIL":
                    f.write(f"  ✗ {r['test_name']}\n")
            f.write(f"\n")

    # Save JSON format for easy parsing
    json_file = log_file.with_suffix(".json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved:")
    print(f"  Log: {log_file}")
    print(f"  JSON: {json_file}")
    print(f"{'='*70}")

    return passed, failed, skipped


def print_analysis(results):
    """Print brief analysis to console."""
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    total = len(results)

def print_analysis(results):
    """Print brief analysis to console."""
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")

    print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")


def main():
    """Main entry point."""
    cpp_file, max_size, build_script = parse_arguments()

    print(f"\nMonster Spawning Grid Test Runner")
    print(f"C++ File: {cpp_file}")
    if build_script:
        print(f"Build Script: {build_script}")
    else:
        print(f"Build Script: (auto-detect build.sh in source directory)")
    if max_size:
        print(f"Max Grid Size: {max_size}")
    else:
        print(f"Max Grid Size: All available")

    # Compile
    executable = compile_cpp(cpp_file, build_script)

    # Find tests
    tests = find_test_files(max_size)
    print(f"\nFound {len(tests)} test files:")
    for test_name, input_file, expected_file in tests:
        status = "✓" if expected_file else "⊘"
        print(f"  {status} {test_name}")

    # Create log file upfront
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cpp_name = cpp_file.stem
    
    cpp_path = cpp_file.resolve()
    if cpp_file.is_absolute() or str(cpp_file).startswith('/'):
        import hashlib
        path_hash = hashlib.md5(str(cpp_path).encode()).hexdigest()[:6]
        log_file = LOGS_DIR / f"{cpp_name}_{path_hash}_{timestamp}.log"
    else:
        log_file = LOGS_DIR / f"{cpp_name}_{timestamp}.log"

    # Write header to log file
    with open(log_file, "w") as f:
        f.write(f"Monster Spawning Grid - Test Run Report\n")
        f.write(f"{'='*70}\n")
        f.write(f"Timestamp:    {datetime.now().isoformat()}\n")
        f.write(f"C++ File:     {cpp_path}\n")
        f.write(f"Tests:        {len(tests)}\n")
        f.write(f"{'='*70}\n")
        f.write(f"\nTest Results:\n")

    # Run tests (passes log_file for on-the-fly writing)
    results = run_all_tests(executable, tests, log_file)

    # Save analysis and JSON
    passed, failed, skipped = save_logs(results, cpp_file, log_file)

    # Print brief summary
    print_analysis(results)

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
