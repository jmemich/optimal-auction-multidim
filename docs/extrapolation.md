# Scaling and Richardson extrapolation for the auction approximation

This note records (a) what Richardson extrapolation is, (b) how we determined the
empirical scaling laws of the approximation on the canonical baseline, and (c) how
those laws let us estimate the $T \to \infty$ revenue limit cheaply instead of
brute-forcing a fine grid. All numbers come from `scripts/stress_t.py` on
**Setting 1a** (Pavlov `U[0,1]^2`, `N=2`, `costs=[0,0]`, independent uniform);
raw data in `docs/stress_t.json`.

## Symbols

| symbol | meaning |
|---|---|
| $T$ | grid-resolution parameter (per-dimension subdivisions) |
| $M = (T+1)^2$ | number of discretised type profiles (LP "points") |
| $\mathrm{rev}(T)$ | approximation's objective value (revenue) at resolution $T$ |
| $R_\infty$ | limit $\lim_{T\to\infty} \mathrm{rev}(T)$ we want to estimate |
| $t(T)$ | wall-clock solve time at resolution $T$ |
| $k(T)$ | inner (cutting-plane) iterations performed during the run |

## 1. What Richardson extrapolation is

Richardson extrapolation recovers the limit of a sequence of finite-resolution
estimates by using the *known rate* at which the discretisation error decays,
rather than by refining the grid until the error is negligible. (Reference:
[Richardson extrapolation](https://en.wikipedia.org/wiki/Richardson_extrapolation).)

**General form.** Suppose an estimate $A(h)$ at step size $h$ has an asymptotic
expansion

$$A(h) = A^* + c_1 h^p + c_2 h^q + \cdots \qquad (q > p),$$

where $A^*$ is the limit we want (as $h \to 0$), $p$ is the **order** of the
leading error, and the $c_i$ are unknown constants. Evaluate at two step sizes
$h_1 \ne h_2$ and drop the $O(h^q)$ remainder:

$$
A(h_1) = A^* + c_1 h_1^p, \qquad
A(h_2) = A^* + c_1 h_2^p .
$$

These are two linear equations in the two unknowns $A^*$ and $c_1$. Multiply the
first by $h_2^p$ and the second by $h_1^p$ and subtract; the common
$c_1 h_1^p h_2^p$ term cancels, **eliminating the leading error**:

$$A^* \approx \frac{h_2^p\, A(h_1) - h_1^p\, A(h_2)}{h_2^p - h_1^p}.$$

The result is accurate to $O(h^q)$ — one order better than either input. (Romberg
integration is the same device with $p = 2$.)

**Specialising to our problem.** The step size is $h = 1/T$ and (Section 3) the
leading error is **linear**, $p = 1$, so with $A \equiv \mathrm{rev}$,
$A^* \equiv R_\infty$, $c_1 \equiv c$:

$$\mathrm{rev}(T) = R_\infty + \frac{c}{T} + O(1/T^2).$$

Rather than substitute into the general formula, it is clearest to redo the two
steps directly in $T$. The two equations are

$$
\mathrm{rev}(T_1) = R_\infty + \frac{c}{T_1}, \qquad
\mathrm{rev}(T_2) = R_\infty + \frac{c}{T_2}.
$$

**Step 1 — subtract** to isolate $c$ (the $R_\infty$ terms cancel):

$$
\mathrm{rev}(T_1) - \mathrm{rev}(T_2) = c\left(\frac{1}{T_1} - \frac{1}{T_2}\right)
= c\,\frac{T_2 - T_1}{T_1 T_2}
\;\;\Longrightarrow\;\;
c \approx \frac{T_1 T_2\,\bigl(\mathrm{rev}(T_1) - \mathrm{rev}(T_2)\bigr)}{T_2 - T_1}.
$$

**Step 2 — back-substitute** $c$ into $\mathrm{rev}(T_2) = R_\infty + c/T_2$ and
simplify:

$$
R_\infty = \mathrm{rev}(T_2) - \frac{c}{T_2}
\;\;\Longrightarrow\;\;
R_\infty \approx \frac{T_2\,\mathrm{rev}(T_2) - T_1\,\mathrm{rev}(T_1)}{T_2 - T_1}.
$$

(These are exactly the general formulas with $p = 1$, $h_i = 1/T_i$.) The only
content is knowing $p$; everything else is solving a $2 \times 2$ linear system.

### Worked example (our data)

Using $T_1 = 30$ ($\mathrm{rev} = 0.598956$) and $T_2 = 40$ ($\mathrm{rev} = 0.595434$):

$$
\begin{aligned}
R_\infty &\approx \frac{40 \cdot 0.595434 - 30 \cdot 0.598956}{10} = \frac{23.81736 - 17.96868}{10} = 0.584868 \\[4pt]
c        &\approx \frac{30 \cdot 40\,(0.598956 - 0.595434)}{10} = \frac{1200 \cdot 0.003522}{10} \approx 0.42264
\end{aligned}
$$

So $\mathrm{rev}(T) \approx 0.5849 + 0.4226/T$. The estimate is **stable across the
pair you pick**: $(20,40) \to 0.58484$, $(40,50) \to 0.58524$. We obtained
$R_\infty \approx 0.585$ from runs that each take under ~70 s, instead of a
(multi-hour) $T \approx 200$ run.

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

**Per-iteration cost is $\sim O(M^2) = O(T^4)$.** Normalising, $t / (k M^2)$ is
nearly constant at $\sim 5\times10^{-7}$:

$$
\begin{aligned}
T=30:&\quad \frac{16.119}{37 \cdot 961^2}  = 4.7\times10^{-7} \\[2pt]
T=35:&\quad \frac{33.892}{41 \cdot 1296^2} = 4.9\times10^{-7} \\[2pt]
T=40:&\quad \frac{69.767}{43 \cdot 1681^2} = 5.7\times10^{-7}
\end{aligned}
$$

The $O(M^2)$ per iteration is the **Border separation sweep**
(`oracle._check_border`): it tests all $M$ candidate subsets, and each call does an
`np.setdiff1d` over the whole type space ($O(M)$), giving $M \cdot O(M) = O(M^2)$.

**Iteration count $k(T)$ grows and does NOT saturate.** It looked like it might
flatten (37, 41, 43), but $T=50$ surged to $k = 109$. So a single global power law
is unreliable; the principled model is

$$t(T) \approx k(T)\, c_{\mathrm{iter}}\, M^2, \qquad c_{\mathrm{iter}} \approx 5\times10^{-7}$$

Fitting a single exponent anyway gives $t \approx 1.5\times10^{-7}\,T^{5.4}$ over
$T \le 40$, but the local log-log slope steepens to ~7.5 across $T = 40 \to 50$
purely because $k$ doubled. **Treat any single-exponent fit as a lower bound on the
true growth.**

### Practical ceiling and extrapolation of time

Measured: $T=40 \to 70$ s, $T=50 \to 369$ s (6.2 min). Extrapolating with the
(volatile) tail slope:

| T | rough time |
|---:|---|
| 60 | ~15-20 min |
| 80 | ~1-2 hr |
| 100 | several hr |

So **$T \le 50$ is the comfortable "minutes" regime** in the current code. Going
higher needs the two perf levers below; see `docs/perf_backlog.md` (to be created).

## 3. How we determined the *revenue* error law ($p = 1$)

$\mathrm{rev}(T)$ decreases monotonically; the successive decrements are:

```
T:        5      10       15       20       25       30       35       40       50
rev:   0.660940 0.625929 0.612877 0.606033 0.601816 0.598956 0.596890 0.595434 0.593395
Δrev:        -0.035   -0.0131  -0.0068  -0.0042  -0.0029  -0.0021  -0.0015  ...
```

Two independent checks that the leading error is $c/T$ (i.e. $p = 1$):

1. **Decrement halving.** Doubling $T$ roughly halves the remaining decrement,
   the signature of a $1/T$ tail (a $1/T^2$ tail would quarter it).
2. **Held-out back-prediction.** The fit $\mathrm{rev}(T) = 0.5849 + 0.4226/T$
   (calibrated on $T=30,40$ only) predicts $T=20 \to 0.60600$ (actual $0.606033$)
   and $T=50 \to 0.59332$ (actual $0.593395$) — both to ~1e-4. A wrong exponent
   would not back- *and* forward-predict held-out points this well.

If higher precision were needed, a three-point fit would additionally cancel the
$1/T^2$ term ($O(1/T^3)$ accuracy). For the current purpose $R_\infty \approx 0.585$
from two cheap points suffices.

### Caveats

- Richardson is only as good as the assumed error law. Here $p = 1$ is validated
  empirically (above) **for the approximation**; do **not** assume it transfers to
  other quantities (see Section 4 — the EBM does *not* have a clean power-series
  error) or to other settings without re-checking the decrement pattern.

## 4. Convergence to the EBM, and why agreement floors near $10^{-3}$

The chapter's optimality conjecture is that the approximation converges to the
exclusive-buyer mechanism (EBM) revenue as $T \to \infty$. Both quantities are
grid-dependent and approach a common limit *from above*, so the right comparison
is between their **extrapolated limits**, not between $\mathrm{rev}(T)$ and a
single $\mathrm{EBM}(T{=}50)$ point. `tests/validation/test_convergence.py` does
exactly this: it Richardson-extrapolates both sequences from $T = 20, 30$ and
checks they agree (settings 1a/2/3; tolerance $3\times10^{-3}$).

The EBM figure is the `ebm_revenue` **second-price (Vickrey) simulation**
(`phd/.../ebm_code/Full analysis.ipynb` cell 9: winner pays
`p[winner_grade] + max(loser surplus, 0)`, integrated over both buyers; ported to
`src/optimal_auctions/validation/ebm.py`) — *not* the posted-price `obj` in the
old `ebm.py`. See `docs/benchmarks.md`.

For Setting 1a the two methods agree to $\sim10^{-3}$ — and, perhaps
surprisingly, **higher-order extrapolation does not tighten this** (EBM at its
optimal reserve $p^* = 0.6$):

| extrapolation | approx $R_\infty$ | EBM $R_\infty$ | $\lvert\Delta\rvert$ |
|---|---|---|---|
| 2-pt $(20,30)$    | 0.584802 | 0.584643 | $1.6\times10^{-4}$ |
| 2-pt $(30,40)$    | 0.584868 | 0.584336 | $5.3\times10^{-4}$ |
| 3-pt $(20,30,40)$ | 0.584934 | 0.584029 | $9.1\times10^{-4}$ |

The **approximation** column is stable ($R_\infty \approx 0.5849$ regardless of
order) — its discretisation error really is a smooth $c/T$. The **EBM** column
*drifts* as points/order are added, so Richardson cannot super-converge it.

Why: the EBM revenue is a trapezoidal integral of a **kinked** integrand — the
Vickrey payment is discontinuous across the reserve boundary and the tie-line
$\beta_1 = \beta_2$ (where $\beta_j = x_j - p_j$). Trapezoidal error for a
function with kinks is $O(1/T)$ but with *irregular* sub-leading terms (depending
on where each kink falls between grid nodes), not a clean power series — so the
expansion the Richardson step assumes does not hold for the EBM.

**Consequence.** The $\sim10^{-3}$ agreement is set by the EBM's quadrature
resolution, not by our extrapolation order — so the convergence test stays at
$3\times10^{-3}$. Pushing below it (e.g. for a publication-grade sub-$10^{-3}$
claim) needs a **kink-aware EBM integrator**: split the integral along the
reserve and tie-line discontinuities so each piece is smooth (then
trapezoid/Richardson super-converge), or integrate the EBM analytically. Tracked
in `TODO.md` (M2).

## Reproducing

```bash
.venv/bin/python scripts/stress_t.py 5 10 15 20 25 30 35 40 50
# results appended to docs/stress_t.json
```
