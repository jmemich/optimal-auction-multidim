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
- [x] **Pavlov closed-form test** — `tests/validation/test_closed_forms.py`: EBM oracle at N=1, U[0,1]² recovers reserve p\*=1/√3 and revenue 2/(3√3)≈0.38490 (Richardson-extrapolated; L-shaped reserve kink → O(1/T)).
- [x] **Myerson reserve test** — `tests/validation/test_closed_forms.py`: EBM oracle at N=1, single good (J=1), U[0,1] recovers reserve 1/2 and revenue 0.25 (1-D reserve kink is a clean 1/T → Richardson exact).
- [x] **Direct analytic test of the *approximation* (N=1)** — `tests/validation/test_approximation_analytic.py`: the LP (not the oracle) at N=1, U[0,1]² Richardson-extrapolates to the Pavlov closed form 2/(3√3)≈0.38490. Non-circular — an analytic target, not the LP's own stored output. **Correction:** randomisation does *not* help at U[0,1]²; the LP converges to the *deterministic* optimum here (it would beat it for shifted supports such as U[2,3]², setting 1b). Worth extending to: a shifted support where the LP should strictly exceed the deterministic EBM (asserting the stochasticity gain), and a single-good Myerson target.
- [ ] **Hypothesis property tests** — *recommend DROP (low value).* IIC/Border feasibility is already the algorithm's own termination criterion (`run()` converges iff the full separation oracle reports zero IIC/Border violations), so every converged solution provably satisfies them — a property test would mostly re-assert the runtime self-check. Monotonicity of Q is *not* LP-enforced, so testing it probes a conjecture about optimal-mechanism structure (research, may not hold universally), not a regression invariant. Only marginal gain (input-space fuzzing for silent non-convergence) on top of the baseline + analytic + convergence coverage.
- [ ] **Kink-aware EBM integration** — the EBM revenue is a trapezoidal integral of a *kinked* integrand (the Vickrey payment is discontinuous across the reserve boundary and the tie-line β₁=β₂), so its discretisation error is **not** a clean power series in 1/T. Richardson therefore does not super-converge the EBM (its extrapolated limit drifts with order), capping approximation↔EBM agreement at ~1e-3 — which is why `test_convergence.py` uses tol 3e-3, not tighter. To reach a publication-grade sub-1e-3 (~1e-5) convergence claim, split the EBM integral along the discontinuity lines so each piece is smooth (then trapezoid/Richardson super-converge) or integrate analytically. See `docs/extrapolation.md` §4.

---

## M3: Profiling

- [ ] **Collaborative profiling walkthrough** — exhaust simple wins first; maintain `docs/perf_backlog.md` of suggestions

---

## Unrelated / Future

- [ ] **`compression` skill** — build a Claude Code skill that embodies the duplicate-removal refactor pattern from item 1 above (separate repo/project)
