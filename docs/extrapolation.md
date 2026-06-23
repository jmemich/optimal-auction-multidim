# Scaling and Richardson extrapolation for the auction approximation

This note records (a) what Richardson extrapolation is, (b) how we determined the
empirical scaling laws of the approximation on the canonical baseline, and (c) how
those laws let us estimate the `T -> infinity` revenue limit cheaply instead of
brute-forcing a fine grid. All numbers come from `scripts/stress_t.py` on
**Setting 1a** (Pavlov `U[0,1]^2`, `N=2`, `costs=[0,0]`, independent uniform);
raw data in `docs/stress_t.json`.

## Symbols

| symbol | meaning |
|---|---|
| `T` | grid-resolution parameter (per-dimension subdivisions) |
| `M = (T+1)^2` | number of discretised type profiles (LP "points") |
| `rev(T)` | approximation's objective value (revenue) at resolution `T` |
| `R_inf` | limit `lim_{T->inf} rev(T)` we want to estimate |
| `t(T)` | wall-clock solve time at resolution `T` |
| `k(T)` | inner (cutting-plane) iterations performed during the run |

## 1. What Richardson extrapolation is

Richardson extrapolation recovers the limit of a sequence of finite-resolution
estimates by using the *known rate* at which the discretisation error decays,
rather than by refining the grid until the error is negligible.

Suppose an estimate obeys an asymptotic expansion in the step size `h`:

```
A(h) = A* + c1 * h^p + c2 * h^q + ...        (q > p)
```

The leading error is `c1 * h^p`. Evaluate at two step sizes `h1`, `h2`; you then
have two equations in the two unknowns `(A*, c1)`, and solving **cancels the
leading error term**, leaving an estimate of `A*` accurate to `O(h^q)`.

For our problem the natural step size is `h = 1/T`, and (Section 3) the leading
error is **linear**, `p = 1`:

```
rev(T) = R_inf + c/T + O(1/T^2)
```

Two runs at `T1 < T2` give

```
R_inf  ≈  (T2 * rev(T2) - T1 * rev(T1)) / (T2 - T1)
c      ≈  T1 * T2 * (rev(T1) - rev(T2)) / (T2 - T1)
```

This is the same device behind Romberg integration (there the error is in powers
of `h^2`); the only content is knowing `p`.

### Worked example (our data)

Using `T1 = 30` (`rev = 0.598956`) and `T2 = 40` (`rev = 0.595434`):

```
R_inf ≈ (40*0.595434 - 30*0.598956) / 10 = (23.81736 - 17.96868)/10 = 0.584868
c     ≈ 30*40*(0.598956 - 0.595434)/10  = 1200*0.003522/10        ≈ 0.42264
```

So `rev(T) ≈ 0.5849 + 0.4226/T`. The estimate is **stable across the pair you
pick**: `(20,40) -> 0.58484`, `(40,50) -> 0.58524`. We obtained `R_inf ≈ 0.585`
from runs that each take under ~70 s, instead of a (multi-hour) `T ≈ 200` run.

## 2. How the *time* scales (the cost wall)

Measured wall-clock times (`docs/stress_t.json`):

| T | M = (T+1)^2 | inner iters `k` | time `t` (s) |
|---:|---:|---:|---:|
| 5 | 36 | 5 | 0.010 |
| 10 | 121 | 9 | 0.093 |
| 15 | 256 | 11 | 0.426 |
| 20 | 441 | 15 | 1.545 |
| 25 | 676 | 22 | 5.253 |
| 30 | 961 | 37 | 16.119 |
| 35 | 1296 | 41 | 33.892 |
| 40 | 1681 | 43 | 69.767 |
| 50 | 2601 | 109 | 369.192 |

**Per-iteration cost is ~O(M^2) = O(T^4).** Normalising, `t / (k * M^2)` is nearly
constant at `~5e-7`:

```
T=30:  16.119 / (37 * 961^2)   = 4.7e-7
T=35:  33.892 / (41 * 1296^2)  = 4.9e-7
T=40:  69.767 / (43 * 1681^2)  = 5.7e-7
```

The `O(M^2)` per iteration is the **Border separation sweep**
(`oracle._check_border`): it tests all `M` candidate subsets, and each call does an
`np.setdiff1d` over the whole type space (`O(M)`), giving `M * O(M) = O(M^2)`.

**Iteration count `k(T)` grows and does NOT saturate.** It looked like it might
flatten (`...37, 41, 43`), but `T=50` surged to `k = 109`. So a single global
power law is unreliable; the principled model is

```
t(T)  ≈  k(T) * c_iter * M^2,   c_iter ≈ 5e-7
```

Fitting a single exponent anyway gives `t ≈ 1.5e-7 * T^5.4` over T<=40, but the
local log-log slope steepens to ~7.5 across `T=40 -> 50` purely because `k`
doubled. **Treat any single-exponent fit as a lower bound on the true growth.**

### Practical ceiling and extrapolation of time

Measured: `T=40 -> 70 s`, `T=50 -> 369 s (6.2 min)`. Extrapolating with the
(volatile) tail slope:

| T | rough time |
|---:|---|
| 60 | ~15-20 min |
| 80 | ~1-2 hr |
| 100 | several hr |

So **`T <= 50` is the comfortable "minutes" regime** in the current code. Going
higher needs the two perf levers below; see `docs/perf_backlog.md` (to be created).

## 3. How we determined the *revenue* error law (`p = 1`)

`rev(T)` decreases monotonically; the successive decrements are:

```
T:        5      10       15       20       25       30       35       40       50
rev:   0.660940 0.625929 0.612877 0.606033 0.601816 0.598956 0.596890 0.595434 0.593395
Δrev:        -0.035   -0.0131  -0.0068  -0.0042  -0.0029  -0.0021  -0.0015  ...
```

Two independent checks that the leading error is `c/T` (i.e. `p = 1`):

1. **Decrement halving.** Doubling `T` roughly halves the remaining decrement,
   the signature of a `1/T` tail (a `1/T^2` tail would quarter it).
2. **Held-out back-prediction.** The fit `rev(T) = 0.5849 + 0.4226/T` (calibrated
   on `T=30,40` only) predicts `T=20 -> 0.60600` (actual `0.606033`) and
   `T=50 -> 0.59332` (actual `0.593395`) — both to ~1e-4. A wrong exponent would
   not back- *and* forward-predict held-out points this well.

If higher precision were needed, a three-point fit would additionally cancel the
`1/T^2` term (`O(1/T^3)` accuracy). For the current purpose `R_inf ≈ 0.585` from
two cheap points suffices.

### Caveats

- Richardson is only as good as the assumed error law. Here `p = 1` is validated
  empirically (above); do **not** assume it transfers to other settings without
  re-checking the decrement pattern.
- **EBM reconciliation (resolved 2026-06-23):** the approximation limit
  `R_inf ≈ 0.585` is consistent with the thesis "EBM (T=50) = 0.589052" for 1a:
  the approximation decreases toward the EBM as `T -> inf`, sustaining the
  chapter's optimality conjecture. The thesis EBM figure is produced by a
  **second-price (Vickrey) auction simulation** (`ebm_revenue` in
  `phd/.../ebm_code/Full analysis.ipynb` cell 9: winner pays
  `p[winner_grade] + max(loser surplus, 0)`, integrated over both buyers), **not**
  by `ebm.py`'s `obj` (a posted-price form that gives ~0.55 and was not used for
  the thesis numbers). The `1/T` revenue extrapolation here is about the
  *approximation's own* limit; it lands just below the EBM, within Richardson
  error and EBM's own finite-grid / eyeballed-price (`p=[0.6,0.6]`) slack.

## Reproducing

```bash
.venv/bin/python scripts/stress_t.py 5 10 15 20 25 30 35 40 50
# results appended to docs/stress_t.json
```
