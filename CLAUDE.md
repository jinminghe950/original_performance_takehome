# CLAUDE.md

## Project Overview

Anthropic's Original Performance Engineering Take-Home challenge. The goal is to optimize a kernel (`KernelBuilder.build_kernel` in `perf_takehome.py`) that runs on a simulated VLIW SIMD processor, minimizing clock cycles. The baseline starts at **147,734 cycles**.

## Repository Structure

```
├── perf_takehome.py          # Main file — optimize KernelBuilder.build_kernel() here
├── problem.py                # Machine simulator, ISA, reference kernels (DO NOT MODIFY for submission)
├── watch_trace.py            # Hot-reload trace visualization server
├── watch_trace.html          # Web UI for Perfetto trace viewer
├── Readme.md                 # Project description and benchmarks
├── tests/
│   ├── submission_tests.py   # Official validation tests (DO NOT MODIFY)
│   └── frozen_problem.py     # Frozen copy of simulator used by submission tests (DO NOT MODIFY)
└── .gitignore
```

## Critical Rules

- **Only modify `perf_takehome.py`** — never change files in `tests/` or `problem.py` for submission
- **Validate with**: `python tests/submission_tests.py`
- **Verify tests/ is unmodified**: `git diff origin/main tests/`
- Do not change `N_CORES` (fixed at 1) — multicore is intentionally disabled

## Commands

```bash
# Run official submission validation (THE authoritative test)
python tests/submission_tests.py

# Run performance test with cycle count
python perf_takehome.py Tests.test_kernel_cycles

# Run with trace output (generates trace.json)
python perf_takehome.py Tests.test_kernel_trace

# View trace in browser (run in separate terminal, Chrome only)
python watch_trace.py

# Run all tests in perf_takehome.py
python perf_takehome.py

# Run reference kernel tests
python perf_takehome.py Tests.test_ref_kernels
```

## Architecture: VLIW SIMD Machine

### Execution Model
- **VLIW (Very Long Instruction Word)**: Each instruction bundle contains slots for multiple engines that execute in parallel within one cycle
- **SIMD**: Vector operations process `VLEN=8` elements simultaneously
- Effects of all slots within a cycle apply atomically at end of cycle (reads happen before writes)

### Engines and Slot Limits (per cycle)
| Engine | Limit | Purpose |
|--------|-------|---------|
| `alu`  | 12    | Scalar arithmetic: `+`, `-`, `*`, `//`, `%`, `^`, `&`, `\|`, `<<`, `>>`, `<`, `==`, `cdiv` |
| `valu` | 6     | Vector arithmetic: same ops as ALU on VLEN elements, plus `vbroadcast`, `multiply_add` |
| `load` | 2     | Memory reads: `const`, `load`, `load_offset`, `vload` |
| `store`| 2     | Memory writes: `store`, `vstore` |
| `flow` | 1     | Control: `select`, `vselect`, `jump`, `cond_jump`, `halt`, `pause`, `coreid`, `add_imm` |
| `debug`| 64    | Testing only: `compare`, `vcompare`, `comment` (ignored by submission simulator) |

### Key Constants
- `VLEN = 8` — vector width
- `N_CORES = 1` — single core only
- `SCRATCH_SIZE = 1536` — available scratch registers (32-bit words)
- `BASELINE = 147734` — starting cycle count

### Memory Model
- **Scratch space**: 1536 registers, acts as registers + constant storage + manual cache
- **Memory**: Flat array of 32-bit words, accessed via `load`/`store`
- All values are 32-bit unsigned integers (mod 2^32)

### Instruction Format
Instructions are dicts mapping engine names to lists of slot tuples:
```python
{"alu": [("+", dest, a1, a2)], "load": [("load", dest, addr)]}
{"valu": [("*", dest, a1, a2), ("+", dest2, a3, a4)], "flow": [("select", dest, cond, a, b)]}
```

## The Kernel

The kernel performs a batched tree traversal over a binary tree for multiple rounds:
1. For each input in the batch: load index and value from memory
2. Load tree node value at current index
3. XOR input value with node value, then apply a 6-stage hash function
4. Use hash result parity to choose left (even) or right (odd) child
5. Wrap index to 0 if it exceeds tree bounds
6. Write updated index and value back to memory

### Hash Function
6 stages defined in `HASH_STAGES`, each performing: `result = op2(op1(a, const1), op3(a, const3))`
Each stage uses 3 ALU operations (implemented in `build_hash`).

### Test Parameters
The performance benchmark uses: `forest_height=10`, `rounds=16`, `batch_size=256`.

## KernelBuilder API

```python
kb = KernelBuilder()
kb.alloc_scratch("name", length=1)    # Allocate scratch register(s)
kb.scratch["name"]                     # Get address of named scratch register
kb.scratch_const(value)                # Allocate and initialize a constant (deduplicated)
kb.add(engine, slot_tuple)             # Append a single-slot instruction
kb.build(slots)                        # Convert list of (engine, slot) pairs into instruction bundles
kb.build_kernel(forest_height, n_nodes, batch_size, rounds)  # The function to optimize
```

## Optimization Strategies

Key areas for reducing cycle count:
- **Instruction packing**: The default `build()` puts one slot per cycle. Pack multiple independent operations into the same cycle using the slot limits above
- **Vectorization**: Use `valu`, `vload`, `vstore` to process VLEN=8 batch elements per slot instead of 1
- **Loop restructuring**: Process batch elements in groups to maximize parallelism within slot limits
- **Reduce memory operations**: Memory loads/stores are bottlenecked at 2 per cycle; minimize and reuse values
- **Pipeline optimization**: Overlap independent operations across different engines in the same cycle

## Performance Thresholds (cycles)

| Threshold | Description |
|-----------|-------------|
| < 147,734 | Better than baseline |
| < 18,532  | Updated take-home starting point |
| < 2,164   | Claude Opus 4 (many hours) |
| < 1,790   | Claude Opus 4.5 (casual session) |
| < 1,579   | Claude Opus 4.5 (2-hour harness) |
| < 1,548   | Claude Sonnet 4.5 (extended compute) |
| < 1,487   | Claude Opus 4.5 (11.5 hours) |
| < 1,363   | Claude Opus 4.5 (improved harness) |

## Debugging

- Use `debug` engine slots (`compare`, `vcompare`) to verify intermediate values against the reference kernel — these are ignored by the submission simulator and cost 0 cycles
- `pause` instructions in the kernel are matched with `yield` in `reference_kernel2` for step-by-step debugging in `perf_takehome.py` tests (but ignored by submission tests)
- Generate and view traces with `test_kernel_trace` + `watch_trace.py` in Chrome/Perfetto
- The `comment` debug instruction can annotate trace output

## Tech Stack

- **Python 3.10+** (uses `match` statements, modern type hints)
- **Standard library only**: `unittest`, `dataclasses`, `enum`, `typing`, `random`, `copy`
- No external dependencies, no package manager, no build system
