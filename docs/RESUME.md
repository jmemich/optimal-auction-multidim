# Resume point — ai-performance-refactor M0

Where the AI refactor session left off (2026-05-18 session, ended at 10% session limit). This is the pick-up doc for the next session.

## Branch state

- On `ai-performance-refactor` (cut from `performance-refactor` at `41f586b`).
- `performance-refactor` is preserved as backup — do not push/rebase/force.
- `master` untouched.
- Nothing has been committed yet on `ai-performance-refactor`. All M0 work is on the working tree (`git status` lists ~25 staged + unstaged + untracked items, listed below). **Review and commit when ready.**

## What's done (M0)

1. **Reconnaissance + planning** — locked: solver=`highspy`, scope=clean-room rewrite preserving public API, distribution=GitHub-only, Python 3.12, package name `optimal_auctions`, branch `ai-performance-refactor`.
2. **Two-repo investigation** — confirmed `optimal-auction-multidim` is the public/clean descendant of `auctions-qualitative` (the research repo that produced the thesis pickle artifacts). AQ is read-only archive going forward. Details in [auctions-qualitative discovery memory](~/.claude/projects/-Users-jamesmichelson-github/memory/project_auctions_qualitative_repo.md).
3. **[docs/benchmarks.md](./benchmarks.md)** — 6 canonical thesis settings, full configs with thesis line refs, two revenue tables, reserve prices, 10 ambiguities (8 resolved during baseline extraction, 2 outstanding).
4. **[scripts/extract_baselines.py](../scripts/extract_baselines.py)** — one-shot AQ-pickle → JSON extractor. Stub-class unpickler (no AQ install needed). Handles scipy.stats frozen distributions (extracts `dist`, `args`, `kwds`).
5. **[docs/baselines.json](./baselines.json)** — 24 entries (6 settings × 4 T values). **22/24 match thesis** to ~1e-7. Two divergences:
   - **Setting 1b T=15**: confirmed thesis typo (Table duplicated T=10 value); pickle gives 2.569334, fits monotone trend. Also took 400 outer iterations vs 9–195 elsewhere — possibly hit cap.
   - **Setting 4 T=20**: small ~0.06% drift (pickle 5.896792 vs thesis 5.893113), pickle is canonical.
6. **[docs/math_reference.md](./math_reference.md)** — canonical math spec. Written from `phd/sections/auctions.tex`. Includes Belloni 2010 connection + the thesis novelty (local IIC separation oracle with star pattern + lower-left quadrant). **Math agent draft had two errors that I corrected**: (a) IIC RHS coefficient was $Q(x)$, should be $Q(\hat x)$ (allocation at *reported* type); (b) OPT$_*$ objective was missing $c_j$ (costs) and the $N$ scaling factor.
7. **Tooling** —
   - `uv` 0.11.15 installed at `~/.local/bin/uv` (curl installer; add to PATH if not already).
   - Layout reorganized: `approx/` → `src/optimal_auctions/`, `approx/tests/` → `tests/` (history preserved via `git mv`).
   - `from approx.X` → `from optimal_auctions.X` across all 8 affected files (verified clean: `grep -rn "^from approx\|^import approx" src/ tests/ scripts/ --include="*.py"` returns nothing).
   - `pyproject.toml` written: hatchling backend, deps = numpy/scipy/highspy/ortools (ortools is transitional, remove after M1), dev deps = pytest/pytest-benchmark/hypothesis/ruff/pyright/pre-commit. Python ≥3.12.
   - `.pre-commit-config.yaml`, `.github/workflows/ci.yml` (CI: ruff + format + pyright + pytest, with `continue-on-error: true` on pyright + pytest at M0 since codebase is WIP), `.gitignore` (rewritten), `README.md` (updated install path).
   - Obsolete configs deleted: `setup.py`, `requirements.txt`, `dev-requirements.txt`, `.flake8`.

## What's left in M0

8. **`uv sync --all-extras`** — installs Python 3.12 + all deps into `.venv`. **Not run yet** (user halted to save session time; will need ~minutes to download).
9. **`uv run pytest`** — confirm test collection works post-rename. Tests are expected to **fail** at M0 (user said "this codebase is in too much of a WIP state anyways"); the rewrite in M1 will make them pass. Just need to know we can collect them.
10. **`uv run ruff check`** / **`uv run ruff format --check`** — sanity-check linter passes (or fix style nits if not).
11. **Build benchmark suite** in `tests/benchmarks/` driven by [baselines.json](./baselines.json). Skeleton: parametrize over the 24 setting × T combos; each test constructs an `OptimalAuctionApproximation`, runs, compares `opt` to baseline revenue with tolerance. Mark `@pytest.mark.benchmark` + `@pytest.mark.slow`. These will be RED until M1; they're the regression contract for the rewrite.

## What's pending (M1+)

- **M1**: clean-room rewrite of `approximation.py`, `constraints.py`, `oracle.py`. HiGHS via `highspy` (incremental rows + warm-starting). Structured constraint identities replacing string IDs. Vectorized separation oracle. Public API stays: `OptimalAuctionApproximation(n_buyers, V, costs, T).run()`.
- **M2**: EBM validation oracle (port from `phd/ch_auctions_simulations/ebm_code/ebm.py` into `src/optimal_auctions/validation/`), Myerson-K1 closed-form test, hypothesis-based property tests.
- **M3**: collaborative profiling walkthrough (per [perf workflow feedback memory](~/.claude/projects/-Users-jamesmichelson-github/memory/feedback_perf_workflow.md) — exhaust simple wins first; maintain `docs/perf_backlog.md` of suggestions).

## Uncommitted git state (snapshot)

```
Staged (renames + deletes):
  D  .flake8
  D  dev-requirements.txt
  D  requirements.txt
  D  setup.py
  R  approx/__init__.py            -> src/optimal_auctions/__init__.py
  R  approx/approximation.py       -> src/optimal_auctions/approximation.py
  R  approx/constraints.py         -> src/optimal_auctions/constraints.py
  R  approx/discrete.py            -> src/optimal_auctions/discrete.py
  R  approx/oracle.py              -> src/optimal_auctions/oracle.py
  R  approx/util.py                -> src/optimal_auctions/util.py
  R  approx/tests/test_approximation.py -> tests/test_approximation.py
  R  approx/tests/test_border.py       -> tests/test_border.py
  R  approx/tests/test_ic.py           -> tests/test_ic.py

Unstaged modifications (import updates + README + .gitignore):
  M  .gitignore
  M  README.md
  M  scripts/test.py
  M  src/optimal_auctions/__init__.py
  M  src/optimal_auctions/approximation.py
  M  src/optimal_auctions/constraints.py
  M  src/optimal_auctions/oracle.py
  M  tests/test_approximation.py
  M  tests/test_border.py
  M  tests/test_ic.py

Untracked (new files):
  .github/                          # CI workflow
  .pre-commit-config.yaml
  docs/                             # benchmarks.md, baselines.json, math_reference.md, RESUME.md (this file)
  pyproject.toml
  scripts/extract_baselines.py
  uv.lock                           # (appeared during this session — may be empty/incomplete; let `uv sync` regenerate)
```

## To resume in the next session

1. `cd /Users/jamesmichelson/github/optimal-auction-multidim && git status` — confirm matches snapshot above.
2. Decide: commit current M0 work as one big commit, or split. Suggested message: `M0: layout, tooling, math reference, benchmark manifest, baselines extraction`. Co-authored-by Claude trailer per convention.
3. Run `export PATH="$HOME/.local/bin:$PATH" && uv sync --all-extras`. Allow ~minutes for Python 3.12 download + ortools wheel.
4. Run `uv run pytest --collect-only` — confirm collection succeeds. If imports fail, fix and re-try.
5. Run `uv run pytest` for full state report (red is expected at M0).
6. Build `tests/benchmarks/test_baselines.py` parameterized over `docs/baselines.json`.
7. Commit + open PR on `ai-performance-refactor`.
8. Start M1: clean-room rewrite.

## Key memories (in `~/.claude/projects/-Users-jamesmichelson-github/memory/`)

- `user.md` — domain expert, math-correctness-first
- `project_auction_refactor.md` — locked decisions
- `project_auctions_qualitative_repo.md` — the two-repo finding
- `reference_thesis_paper.md` — where to find the math
- `feedback_perf_workflow.md` — collaborative profiling norms
- `feedback_branch_workflow.md` — preserve James's branches
- `feedback_docs_format.md` — markdown reference > docstring math
- `feedback_math_vigilance.md` — flag suspected pre-existing bugs

These are read at the start of every new session. They carry the durable context.
