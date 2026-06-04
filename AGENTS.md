# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python performance take-home centered on `KernelBuilder.build_kernel`.

- `perf_takehome.py`: primary implementation and local test entry point; optimize here.
- `problem.py`: simulator, ISA behavior, and reference kernels.
- `tests/submission_tests.py`: official correctness + performance threshold checks.
- `tests/frozen_problem.py`: frozen simulator copy used by submission tests.
- `watch_trace.py` and `watch_trace.html`: local trace viewer for `trace.json`.

Keep changes focused on optimization code unless explicitly asked; avoid editing `tests/`.

## Build, Test, and Development Commands
- `python perf_takehome.py`: run local `unittest` suite in `perf_takehome.py`.
- `python perf_takehome.py Tests.test_kernel_cycles`: run the main cycle-count test quickly.
- `python tests/submission_tests.py`: run the official scoring/correctness checks.
- `git diff origin/main tests/`: verify the `tests/` directory is unchanged before submission.
- `python perf_takehome.py Tests.test_kernel_trace` then `python watch_trace.py`: generate and inspect instruction traces in Perfetto.

## Coding Style & Naming Conventions
Follow existing Python style in this repo:

- 4-space indentation, no tabs.
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants (for example `BASELINE`, `VLEN`).
- Prefer clear scratch/register naming (`tmp_idx`, `tmp_addr`) and small helper methods.
- Keep instruction tuples and slot ordering explicit; prioritize readability during aggressive optimization.

No formatter/linter config is checked in; match the surrounding file style and keep diffs minimal.

## Testing Guidelines
- Test framework: standard library `unittest`.
- Primary validation path is `python tests/submission_tests.py`; treat it as the source of truth.
- If adding tests, use `test_*` method names in `unittest.TestCase` classes.
- Preserve deterministic behavior where seeds are used; do not weaken assertions for speed.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects (for example: `Add notes about proper verification`).

- Keep commit titles concise and action-oriented.
- In PRs, include what changed and why.
- In PRs, include cycle count before/after.
- In PRs, include exact commands run (`python tests/submission_tests.py`, `git diff origin/main tests/`).
- If trace-based analysis informed changes, include a brief note or screenshot of key findings.
