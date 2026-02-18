# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is Anthropic's Original Performance Engineering Take-Home Challenge. The goal is to optimize a simulated machine kernel to minimize clock cycles. The baseline starts at **147,734 cycles**.

**CRITICAL WARNING**: Do NOT modify anything in the `tests/` directory. The tests are frozen and must remain unchanged. Do NOT modify `N_CORES` in `problem.py` — multicore is intentionally disabled.

## Commands

```bash
# Run the main test (reports cycle count and correctness)
python perf_takehome.py

# Run a specific test
python perf_takehome.py Tests.test_kernel_cycles

# Official submission scoring — shows which performance thresholds you pass
python tests/submission_tests.py

# Validate the tests folder hasn't been modified (must be empty)
git diff origin/main tests/

# Interactive trace visualization (run both in separate terminals)
python perf_takehome.py Tests.test_kernel_trace
python watch_trace.py  # then open http://localhost:8000
```

## Architecture

The project has three layers:

**`problem.py`** — The frozen machine simulator (do not modify for cheating purposes). Defines a custom VLIW SIMD architecture:
- `Machine` class: simulates execution with parallel engine slots: 12 ALU, 6 VALU (vector), 2 Load, 2 Store, 1 Flow, 64 Debug
- `VLEN = 8` (vector width), `N_CORES = 1` (intentionally disabled multicore)
- `SCRATCH_SIZE = 1536` bytes of scratch memory
- `reference_kernel()` / `reference_kernel2()`: ground-truth Python implementations for correctness comparison
- `HASH_STAGES`: 6-stage hash function used inside the kernel

**`perf_takehome.py`** — The file you optimize. Contains:
- `KernelBuilder` class: your workspace for generating optimized machine instruction bundles
- `build_kernel()`: the core method to optimize — emits instruction bundles that implement the kernel
- `build_hash()`: builds the 6-stage hash computation
- `build()`: packs instructions into bundles (currently one slot per bundle — a major optimization opportunity)
- `do_kernel_test()`: test harness that builds memory image, runs simulator, verifies correctness, reports cycles

**`tests/`** — Frozen evaluation harness (do not touch):
- `submission_tests.py`: official scoring with cycle thresholds
- `frozen_problem.py`: immutable copy of the simulator used for official scoring

## The Kernel Being Optimized

The kernel performs a parallel tree traversal over 16 rounds × 256 inputs per batch:

1. Load current index and value for each input
2. Load tree node value at that index
3. Compute `val = hash(val ^ node_val)` (6-stage hash)
4. Compute next index: `2*idx + (1 if val%2==0 else 2)`
5. Bounds check: clamp to 0 if `next_idx >= n_nodes`
6. Store updated index and value

## Performance Thresholds

These are the benchmark targets from `tests/submission_tests.py`:

| Cycles | Milestone |
|--------|-----------|
| < 147,734 | Any speedup over baseline |
| < 18,532 | 2-hour take-home starting point |
| < 2,164 | Claude Opus 4 (many hours) |
| < 1,790 | Claude Opus 4.5 (casual session) |
| < 1,579 | Claude Opus 4.5 (2-hour harness) |
| < 1,548 | Claude Sonnet 4.5 (many hours) |
| < 1,487 | Claude Opus 4.5 (11.5-hour harness) |
| < 1,363 | Claude Opus 4.5 (improved harness) |

Beating 1,487 cycles is the target to impress Anthropic. See `Readme.md` for submission instructions.

## Key Optimization Opportunities

- **Instruction-level parallelism**: `build()` currently emits one instruction per bundle; packing multiple instructions into a single bundle (using all available engine slots in parallel) is the primary optimization lever
- **VALU/vector ops**: Use VLEN=8 vector instructions to process 8 inputs simultaneously
- **Scratch space layout**: Efficient scratch memory allocation reduces load/store pressure
- **Loop unrolling**: Reduce branch overhead across the 16 rounds
