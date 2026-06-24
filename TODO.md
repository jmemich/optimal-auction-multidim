# TODO

## Branch: ai-performance-refactor

- [x] **Remove duplicate code** — refactor on this branch made previous functions obsolete; audit and delete dead code
- [x] **Fix CI** — GitHub Actions runs ruff + full pytest suite (incl. 24-case baselines) on push/PR to this branch; pytest now gates (continue-on-error removed). Green as of 96b2f5b. Pyright still informational (non-gating) — tighten later.
- [x] **Drop lockfile** — `uv.lock` untracked + gitignored; CI/installs resolve fresh from `pyproject.toml` bounds (conventional library approach). (Dep-bound tightening to tested versions still optional/open.)
- [x] **Stress-test for large T** — verify the solver scales for much larger numbers of type profiles T
- [ ] **Rip out multiprocessing** — revert to single-process implementation; multiprocessing may be a valid perf win but introduces complexity (pickling, process overhead, platform quirks) that isn't worth dealing with now; revisit after M3 profiling confirms it's the right lever

---

## Benchmarks integrity

- [ ] **Adversarial review of `benchmarks.json`** — investigate whether the regression suite actually guarantees what it claims; consider: are the expected values derived from the same code being tested (circular), or from an independent oracle? Can a subtly wrong solver pass all cases? Design or commission an adversarial review that tries to find inputs the suite would miss, then tighten accordingly.

---

## M2: Validation

- [x] **EBM validation oracle** — ported as the `ebm_revenue` second-price (Vickrey) simulation in `src/optimal_auctions/validation/ebm.py` (NOT `ebm.py`'s posted-price `obj`). All 6 settings pinned in `tests/validation/test_ebm.py`; 1a–4 reproduce `benchmarks.md` exactly, setting 5 pinned to the corrected normalised value (2.7011591; published 2.57921 is unnormalised — see ⚠³).
- [x] **Convergence test** — `tests/validation/test_convergence.py` Richardson-extrapolates BOTH the approximation and the (reserve-optimised) EBM from T=20,30 and asserts a common T→∞ limit for settings 1a/2/3 (agree to <1e-3; tol 3e-3). Key insight: both are grid-dependent and converge from above to the same limit (~0.585 for 1a), so we compare limits, not approx(T) vs a single EBM(T=50) point.
- [ ] **Pavlov closed-form test** — K=2, N=1, X~U[0,1]²: reserve p\*=1/√3, revenue≈0.3849
- [ ] **Myerson reserve test** — K=1 degenerate-dimension sanity check against closed-form regular-distribution reserve
- [ ] **Hypothesis property tests** — IIC (full, not just local), monotonicity of Q, Border (B) for arbitrary subsets

---

## M3: Profiling

- [ ] **Collaborative profiling walkthrough** — exhaust simple wins first; maintain `docs/perf_backlog.md` of suggestions

---

## Unrelated / Future

- [ ] **`compression` skill** — build a Claude Code skill that embodies the duplicate-removal refactor pattern from item 1 above (separate repo/project)
