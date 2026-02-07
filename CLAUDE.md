# CLAUDE.md

## Project Overview

Anthropic's Original Performance Engineering Take-Home challenge. The task is to optimize a kernel (`KernelBuilder.build_kernel` in `perf_takehome.py`) running on a simulated VLIW SIMD processor, minimizing simulated clock cycles while maintaining correctness.

- **Language**: Python 3 (standard library only, no external dependencies)
- **Baseline**: 147,734 cycles
- **Hiring threshold**: < 1,487 cycles (Claude Opus 4.5's best at launch)

## Repository Structure

```
perf_takehome.py          # Main file: KernelBuilder (optimize this), Tests, reference kernels
problem.py                # Machine simulator, instruction set, data structures (DO NOT MODIFY for submission)
watch_trace.py            # Hot-reloading Perfetto trace viewer server
watch_trace.html          # Perfetto UI integration page
tests/
  submission_tests.py     # Official submission validation (DO NOT MODIFY)
  frozen_problem.py       # Frozen copy of simulator used by submission tests (DO NOT MODIFY)
```

## Critical Rules

1. **NEVER modify anything in the `tests/` folder.** Submission tests use a frozen simulator copy to prevent cheating. Run `git diff origin/main tests/` to verify.
2. **NEVER change `N_CORES`** in `problem.py`. Multicore is intentionally disabled.
3. **Only modify `perf_takehome.py`** — specifically `KernelBuilder.build_kernel()` and supporting methods.
4. **Do not modify `problem.py`** if you want a valid submission, since the submission tests use `frozen_problem.py`.

## Commands

```bash
# Run all tests (includes trace generation and cycle count)
python perf_takehome.py

# Run only the cycle count test
python perf_takehome.py Tests.test_kernel_cycles

# Run trace test (for Perfetto debugging)
python perf_takehome.py Tests.test_kernel_trace

# Start hot-reloading trace viewer (run in separate terminal, use Chrome)
python watch_trace.py

# Official submission validation — the authoritative benchmark
python tests/submission_tests.py
```

## Architecture: Simulated VLIW SIMD Machine

### Instruction Format

Each instruction is a VLIW bundle — a dict mapping engine names to lists of operation slots. Multiple engines execute in parallel within one cycle.

### Engines and Slot Limits

| Engine  | Max Slots | Purpose                                    |
|---------|-----------|--------------------------------------------|
| `alu`   | 12        | Scalar arithmetic (`+`, `-`, `*`, `//`, `<`, `==`, bitwise, etc.) |
| `valu`  | 6         | Vector arithmetic (same ops, operates on VLEN=8 elements) |
| `load`  | 2         | Memory loads (`load`, `load_offset`, `vload`, `const`) |
| `store` | 2         | Memory stores (`store`, `vstore`)          |
| `flow`  | 1         | Control flow (`jump`, `cond_jump`, `select`, `vselect`, `halt`, `pause`, `trace_write`) |
| `debug` | 64        | Debug comparisons (assert-like, no cycle cost) |

### Key Constants

- `VLEN = 8` — vector width (8 elements per vector operation)
- `N_CORES = 1` — single core only
- `SCRATCH_SIZE = 1536` — scratch register file size (32-bit words)

### Memory Model

Flat array of 32-bit unsigned integers. Header at offset 0:
- `[0]` rounds, `[1]` n_nodes, `[2]` batch_size, `[3]` forest_height
- `[4]` forest_values pointer, `[5]` indices pointer, `[6]` values pointer, `[7]` extra room pointer

### Algorithm (what the kernel computes)

For each round, for each batch item:
1. Load current index and value from memory
2. Load tree node value at that index
3. XOR value with node value
4. Apply 6-stage hash function (multiply, shift, XOR sequence)
5. Determine left/right child based on hash parity (bit 0)
6. Update index and value in memory

## Performance Thresholds (from submission_tests.py)

| Test Name | Cycles | Description |
|-----------|--------|-------------|
| `test_kernel_speedup` | < 147,734 | Beat naive baseline |
| `test_kernel_updated_starting_point` | < 18,532 | 2-hour version start |
| `test_opus4_many_hours` | < 2,164 | Opus 4 after many hours |
| `test_opus45_casual` | < 1,790 | Opus 4.5 casual session |
| `test_opus45_2hr` | < 1,579 | Opus 4.5 after 2 hours |
| `test_sonnet45_many_hours` | < 1,548 | Sonnet 4.5 many hours |
| `test_opus45_11hr` | < 1,487 | Opus 4.5 after 11.5 hours |
| `test_opus45_improved_harness` | < 1,363 | Opus 4.5 improved harness |

## Optimization Strategy Guide

Key optimization avenues (in rough order of impact):

1. **VLIW packing**: Pack multiple independent operations into the same instruction bundle to exploit instruction-level parallelism (all engines execute in parallel within one cycle)
2. **Vectorization**: Use `valu`/`vload`/`vstore` to process 8 batch items per cycle instead of 1
3. **Loop structure**: Minimize loop overhead; unroll where beneficial
4. **Instruction scheduling**: Reorder operations to fill all engine slots each cycle
5. **Memory access patterns**: Optimize load/store placement and reduce memory bottlenecks
6. **Reduce total instruction count**: Combine or eliminate redundant operations
7. **Use `select`/`vselect`**: Branchless conditional moves avoid costly jumps

## Debugging Workflow

1. Run `python perf_takehome.py Tests.test_kernel_trace` to generate `trace.json`
2. In a separate terminal, run `python watch_trace.py`
3. Open the printed URL in Chrome — shows cycle-by-cycle execution in Perfetto UI
4. The trace hot-reloads: edit code, re-run the trace test, and the browser updates

## Testing Conventions

- Test framework: Python `unittest`
- Main development tests: `Tests` class in `perf_takehome.py`
- Submission validation: `python tests/submission_tests.py` (uses frozen simulator)
- Correctness tests run the kernel 8 times with different random seeds
- Always validate with `tests/submission_tests.py` before considering a result final

## Code Style

- No linter or formatter configured
- Uses Python type hints (`Literal`, `dataclass`, etc.)
- Standard library only — no third-party packages
- `KernelBuilder` methods use a builder pattern: `alloc()` for scratch registers, `add()` to append instructions, `const()` for constants
