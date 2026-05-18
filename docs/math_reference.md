# Math Reference

This document is the canonical mathematical specification for the optimal multidimensional auction approximation algorithm implemented in this repository — drawn from Chapter 3 of James Michelson's PhD thesis ([sections/auctions.tex](../../phd/sections/auctions.tex)) and the working paper [Kushnir & Michelson 2022](https://arxiv.org/abs/2207.01664). It is the math-correctness contract for any rewrite: every refactored routine should be traceable to an equation or definition here.

It is not a tutorial, a proofs reference, or a substitute for the thesis. For the full economic intuition, history, and conjecture-supporting discussion, read the thesis chapter directly.

## Setting

A seller offers a single indivisible good with $K$ quality grades (e.g. *standard* / *premium*) to $N$ risk-neutral bidders. Bidder $i$'s valuation for grade $j$ is a real number $x_j^i \in [\underline{x}_j^i,\, \overline{x}_j^i]$; their complete (multidimensional) type is the vector $x^i = (x_1^i, \ldots, x_K^i) \in X^i := \prod_j [\underline{x}_j^i, \overline{x}_j^i]$. The joint type profile is $x = (x^1, \ldots, x^N) \in X := \prod_i X^i$ drawn from a known joint distribution $F$ over $X$.

In the symmetric settings of interest, bidder types are i.i.d. across bidders; within a bidder, valuations across grades may be correlated (Setting 3 in [benchmarks.md](./benchmarks.md) makes this explicit). The seller has constant per-unit production costs $c_j \ge 0$ per grade.

The problem: design a direct mechanism — an allocation rule $q : X \to [0,1]^{KN}$ and payment rule $p : X \to \mathbb{R}^N$ — that maximizes expected revenue net of costs, subject to feasibility (only one good is sold), interim individual rationality (IIR), and interim incentive compatibility (IIC).

Setup is at [auctions.tex:68-127](../../phd/sections/auctions.tex#L68).

## Formal objects

| Symbol | Meaning | Type | Primitive / Designed |
|--------|---------|------|----------------------|
| $N$ | Number of bidders | int $\ge 1$ | primitive |
| $K$ | Number of quality grades | int $\ge 1$ | primitive |
| $X^i_j = [\underline{x}_j^i, \overline{x}_j^i]$ | Valuation range for bidder $i$, grade $j$ | interval | primitive |
| $X^i = \prod_j X^i_j$ | Bidder $i$'s type space | hyper-rectangle | derived |
| $X = \prod_i X^i$ | Joint type space | hyper-rectangle | derived |
| $x^i$ | Bidder $i$'s realized type | vector $\in X^i$ | random (drawn from $F^i$) |
| $x_j^i$ | Bidder $i$'s valuation for grade $j$ | scalar | random |
| $F$ | Joint type distribution | CDF on $X$ | primitive |
| $F^i$ | Marginal distribution of $x^i$ | CDF on $X^i$ | primitive (= same $F^0$ across $i$ when symmetric) |
| $F^{-i}$ | Joint distribution of $x^{-i}$ | CDF on $X^{-i}$ | primitive |
| $c_j$ | Seller's per-unit cost of grade $j$ | scalar $\ge 0$ | primitive |
| $r_j$ | Reserve price for grade $j$ (used by EBM) | scalar $\ge 0$ | choice variable (optimized over) |
| $q_j^i(x)$ | Ex-post allocation: probability bidder $i$ wins grade $j$ at profile $x$ | function $X \to [0,1]$ | designed |
| $p^i(x)$ | Ex-post payment from bidder $i$ at profile $x$ | function $X \to \mathbb{R}$ | designed |
| $Q_j^i(x^i) = \int_{X^{-i}} q_j^i(x^i, x^{-i})\,dF^{-i}$ | Interim allocation | function $X^i \to [0,1]$ | derived from $q$ |
| $P^i(x^i) = \int_{X^{-i}} p^i(x^i, x^{-i})\,dF^{-i}$ | Interim payment | function $X^i \to \mathbb{R}$ | derived from $p$ |
| $U^i(x^i) = \sum_j Q_j^i(x^i)\, x_j^i - P^i(x^i)$ | Interim utility (truthful) | function $X^i \to \mathbb{R}$ | derived from $Q, P$ |
| $\beta_j^i = x_j^i - r_j$ | Linear virtual value (EBM only) | scalar | derived |
| $X_T \subset X$ | Discretization of $X$ | finite grid | designed (approximation parameter) |
| $T$ | Discretization granularity (points per dimension $\approx T+1$) | int | approximation parameter |
| $\hat f(v)$ | Discretized density at $v \in X_T$ (trapezoidal-rule weights at boundaries) | scalar | derived from $F$ and $T$ |
| $L$ | "Local region size" in the IIC separation oracle | int (grows over outer iterations) | algorithm internal |

Note on the *interim* formulation: the thesis works with $(Q^i, U^i)$ as decision variables rather than $(q, p)$, exploiting the fact that the joint $(Q, U)$ pair is sufficient for both the objective and all incentive constraints. This drops the dependence on $x^{-i}$ from the LP and makes the problem polynomial in $|X_T|$ instead of $|X_T|^N$. See [auctions.tex:129-154](../../phd/sections/auctions.tex#L129).

## Constraint classes

### (F) — Feasibility

Only one good is sold across all bidders and grades:
$$\sum_i \sum_j q_j^i(x) \;\le\; 1, \qquad q_j^i(x) \;\ge\; 0 \qquad \forall x \in X,\; i,\; j. \tag{F}$$

Stated this way, (F) has uncountably-many (or, in discrete, exponentially-many in $|X_T|$) constraints. **Border (1991)** shows that for the *interim* allocation $Q$, feasibility is equivalent to a polynomial-size family of subset inequalities. In discrete form: for all subsets $A \subseteq X_T$ and all grades $j$,
$$N \sum_{v \in A} \hat f(v)\, Q_j(v) \;\le\; 1 - \Bigl(\sum_{v \notin A} \hat f(v)\Bigr)^{\!N}. \tag{B}$$
This is the inequality implemented by `border_lhs_minus_rhs` in [approx/constraints.py](../approx/constraints.py). The number of binding subsets is small in practice; the separation oracle generates only those.

### (IIR) — Interim individual rationality

Every bidder must have non-negative expected utility at every type:
$$U^i(x^i) \;\ge\; 0 \qquad \forall x^i \in X^i,\; i. \tag{IIR}$$

In the discrete LP these are $|X_T|$ trivial lower-bound constraints. The boundary condition $U^i(\underline{x}^i) = 0$ (utility at the worst type is exactly zero) tightens the constraint and is sufficient to pin the LP value.

### (IIC) — Interim incentive compatibility

No bidder can benefit from misreporting. For every bidder $i$ and every pair of types $x^i, \hat x^i \in X^i$:
$$U^i(x^i) \;-\; U^i(\hat x^i) \;\ge\; \sum_j Q_j^i(\hat x^i)\,(x_j^i - \hat x_j^i). \tag{IIC}$$

Note carefully: the coefficient on $(x_j^i - \hat x_j^i)$ is $Q_j^i(\hat x^i)$ — the allocation under the **reported** type, not the true type. (Derivation: the utility of bidder $i$ with true type $x^i$ when reporting $\hat x^i$ is $\hat U^i(\hat x^i \mid x^i) = \sum_j Q_j^i(\hat x^i)\, x_j^i - P^i(\hat x^i) = U^i(\hat x^i) + \sum_j Q_j^i(\hat x^i)\,(x_j^i - \hat x_j^i)$. The truth-telling condition $U^i(x^i) \ge \hat U^i(\hat x^i \mid x^i)$ gives the inequality above.)

In the discrete LP there are $O(|X_T|^2)$ such pairs per bidder — the dominant constraint family. The local separation oracle (next section) is the key optimization for handling these.

This is `ic_lhs_minus_rhs` in [approx/constraints.py](../approx/constraints.py); the code computes `U[i] - U[j] - sum(Q[k][j] * (v_i[k] - v_j[k]))` where index `i` indexes the true type and `j` the reported type — exactly $Q_j^i(\hat x^i)$ as required.

## The optimization — (OPT$_*$)

The seller's objective is total expected net revenue (payments minus production costs). Using $P^i(x^i) = \sum_j Q_j^i(x^i)\, x_j^i - U^i(x^i)$ and the i.i.d.-symmetric assumption (all $N$ bidders contribute the same expectation), the objective reduces to:

$$\max_{Q,\, U} \quad N \int_{X^0} \!\Bigl[\sum_j (x_j - c_j)\, Q_j(x) - U(x)\Bigr]\, dF^0(x) \quad \text{subject to (B), (IIR), (IIC).} \tag{OPT$_*$}$$

In discrete form:
$$\max_{Q,\, U} \quad N \sum_{v \in X_T} \hat f(v)\,\Bigl[\sum_j (v_j - c_j)\, Q_j(v) - U(v)\Bigr].$$

This is implemented in `_make_obj` in [approx/approximation.py](../approx/approximation.py).

For *asymmetric* settings (Settings 4 and 5 in [benchmarks.md](./benchmarks.md)), the $N$ scaling factors out per bidder and the objective is a sum over $i$ — the code path handles this via the `force_symmetric` flag.

## Predecessor: Belloni et al. 2010

The thesis approximation algorithm extends **Belloni, Lopomo, Wang 2010** ("Multidimensional Mechanism Design: Finite-Dimensional Approximations and Efficient Computation," available as [MMD.pdf](https://faculty.fuqua.duke.edu/~abn5/MMD.pdf)). Belloni's contributions that this code inherits:

1. **Discretize** $X$ into a uniform grid $X_T$ at granularity $T$.
2. **Use Border's characterization (B)** so feasibility constraints scale polynomially in $|X_T|$ rather than exponentially.
3. **Iterative constraint generation**: solve the LP with a small initial constraint set; run a separation oracle that identifies violated (IIC) and (B) constraints; add them and re-solve; repeat until no violations are found.
4. **Random sampling in the separation oracle**: at each iteration, check a random subset of the $O(|X_T|^2)$ IIC pairs.

The class name `BelloniAuctionApproximation` in [auctions-qualitative](../../auctions-qualitative/auction/approximation.py) (the historical research code) and `OptimalAuctionApproximation` in this repo both reference this lineage. The Belloni paper PDF is cited in this repo's [README](../README.md).

## Thesis algorithmic novelty: local IIC separation oracle

The thesis's algorithmic contribution is replacing Belloni's *random* IIC sampling with a **structured, local** separation oracle, exploiting the empirical observation that optimal $Q$ is monotone (verified by the post-solve tests in [approx/tests/test_ic.py](../approx/tests/test_ic.py)) and hence IIC violations cluster between *nearby* types in $X_T$.

**Outer / inner loop structure.**

*Outer loop* — accumulates the active IIC constraint set across iterations. Each outer iteration:
1. Solves the LP with the current constraint set.
2. Runs the **full** separation oracle: checks every $(x^i, \hat x^i)$ pair across $X_T$ for an IIC violation.
3. If any violations exist, adds them to the constraint set, grows the local-region parameter $L$, and starts a new outer iteration.
4. Terminates when no global IIC violations remain.

*Inner loop* — within an outer iteration, the LP is re-solved repeatedly as Border violations are added one batch at a time. Each inner iteration:
1. Solves the LP with the current Border constraint set.
2. Runs the **Border** separation oracle (over all subsets of $X_T$, sorted to test most-violated first).
3. If any Border violations exist, adds them and re-solves.
4. Terminates when Border is fully satisfied.

**Local IIC region structure (the load-bearing optimization).**

When checking IIC during an inner iteration (the cheap, non-global check), only pairs $(x^i, \hat x^i)$ in the **local region** around each $x^i$ are tested. The local region is:

- A **star pattern** of immediately-adjacent grid neighbors (the codebase agent counted 8 neighbors per type: 4 cardinal + 4 diagonal in 2D; for general $K$, this is up to $3^K - 1$ neighbors with boundary truncation).
- A **lower-left quadrant** of size $L$: all $\hat x^i$ with $\hat x_j^i \le x_j^i$ for every $j$ and $|x_j^i - \hat x_j^i| \le L \cdot \epsilon_j$ (where $\epsilon_j$ is the per-dimension grid spacing). This is the "downward-sloping" region where IIC tends to bind.

Local-region checks are implemented in `_check_ic` in [approx/oracle.py](../approx/oracle.py) (the `check_local` flag and surrounding logic). The full check (over all pairs) runs only at the boundary of the outer loop.

**Why this is a speedup, in plain terms.** If $Q$ is monotone, the IIC constraint $U(x) - U(\hat x) \ge \sum_j Q_j(\hat x)(x_j - \hat x_j)$ is hardest to satisfy when $(x, \hat x)$ are "nearby and downward-sloping" — sampling random pairs (Belloni) wastes work on pairs that are slack by a wide margin. Empirically, this allows convergence in 9–195 outer iterations per setting at $T=20$ (see iteration counts in [benchmarks.md](./benchmarks.md)) versus what would be many more under random sampling. The one outlier (Setting 1b, $T=15$, 400 iterations) is the case where this assumption interacts badly with the discretization — flagged for refactor-time investigation.

**Refactor implication.** Each outer iteration in the current code rebuilds the LP from scratch using `solver.Add(make_*_expr_from_name(...))`, which string-parses constraint names like `"ic_3_47"` to reconstruct LP expressions ([approx/constraints.py:67](../approx/constraints.py#L67) onwards). This dominates wall-clock per the PR's own profiling. The refactor's HiGHS-with-persistent-rows design eliminates this overhead at the source: constraints are added once and persist with stable row indices, no string parsing, no full rebuild.

## Discretization

The continuous type space is approximated by a uniform tensor-product grid:
$$X_T = \prod_{i,\, j} \{\underline{x}_j^i + k \cdot \epsilon_j^i\;:\; k = 0, 1, \ldots, T\}, \qquad \epsilon_j^i := \frac{\overline{x}_j^i - \underline{x}_j^i}{T}.$$

Per buyer $i$, this gives $(T+1)^K$ points; the joint $|X_T|$ scales as $(T+1)^{KN}$, but the *interim* formulation factors this so the LP variables are $O((T+1)^K)$ per buyer.

The discretized density $\hat f(v)$ uses a trapezoidal-rule adjustment: boundary grid points are weighted by $1/2$ per boundary dimension (so corners in 2D get $1/4$). See `f_hat.numerical_adjustment` in [approx/discrete.py](../approx/discrete.py).

**Convergence claim.** As $T \to \infty$, the discrete LP's optimum approaches the continuous LP's optimum. The thesis tests $T \in \{5, 10, 15, 20\}$; revenues are monotone decreasing in $T$ across all six benchmarked settings (after correction of the Setting 1b $T=15$ thesis typo — see [benchmarks.md](./benchmarks.md)), consistent with this convergence. No closed-form convergence rate is given.

## Exclusive Buyer Mechanism (EBM)

The EBM is a *conjectured-optimal* mechanism in most settings of interest in the thesis — used as an independent oracle to validate the LP's revenue.

**Allocation rule.** Let $\beta_j^i = x_j^i - r_j$ be the linear virtual value for bidder $i$ on grade $j$. Define $\beta^i = \max_j \beta_j^i$ (the bidder's "strength" — their best virtual value) and $j^*(i) = \arg\max_j \beta_j^i$ (their preferred grade given the reserves). The set of winning bidders is
$$M(x) = \{\, i \;:\; \beta^i = \max_{i'} \beta^{i'} \text{ and } \beta^i \ge 0 \,\}.$$
The allocation:
$$q_j^i(x) = \begin{cases} 1 / |M(x)| & \text{if } i \in M(x) \text{ and } j = j^*(i), \\ 0 & \text{otherwise.} \end{cases}$$

If all $\beta^i < 0$, the good is not allocated (the "exclusion region"). Ties are broken uniformly.

Payments follow from integrating the allocation along type-space paths (the Myerson envelope), with reserve prices $r_j$ chosen to maximize the resulting expected revenue. The exclusion region's geometry — particularly its conjectured invariance to $N$ — is one of the empirical conjectures the thesis investigates ([auctions.tex:865-905](../../phd/sections/auctions.tex#L865)).

EBM revenue at $T=50$ is computed by numerical integration in [ebm.py](../../phd/ch_auctions_simulations/ebm_code/ebm.py) and serves as the row-`EBM (T=50)` column in [benchmarks.md](./benchmarks.md)'s revenue table.

## Invariants the refactor must preserve

Properties already pinned by tests in [approx/tests/](../approx/tests/):

- **Monotonicity of $Q$**: for each grade $j$, $Q_j^i(x^i)$ is non-decreasing in each component $x_k^i$. ([test_ic.py — `test_Q_monotone`](../approx/tests/test_ic.py).) Necessary for IIC + sufficient given the envelope condition.
- **Full (IIC)**: $U(x) - U(\hat x) \ge \sum_j Q_j(\hat x)(x_j - \hat x_j)$ holds for every pair, not just the local-region pairs. ([test_ic.py — `test_basic_ic`](../approx/tests/test_ic.py).)
- **Border (B)**: $N \sum_{v \in A} \hat f(v) Q_j(v) \le 1 - (\sum_{v \notin A} \hat f(v))^N$ for every subset $A \subseteq X_T$ and every grade $j$. ([test_border.py](../approx/tests/test_border.py).)
- **Pickle serialization** roundtrip: state can be saved and resumed. ([test_approximation.py](../approx/tests/test_approximation.py).) — will be replaced by JSON-based serialization in the refactor.

Properties the refactor must additionally preserve (not currently tested, candidates for new tests in M2):

- **Convergence to EBM** as $T \to \infty$ in the settings where EBM is conjectured optimal (Settings 1a, 2, 3 — see [benchmarks.md](./benchmarks.md) for the revenue gap table).
- **Pavlov closed form** for $K=2, N=1$, $X \sim U[0,1]^2$: reserve $p^* = 1/\sqrt{3}$, revenue $\approx 0.3849$.
- **Myerson reserve** for $K=1$: closed-form regular-distribution reserve. Degenerate-dimension sanity check.

## Code ↔ math mapping

Identifiers below are in the current `approx/` package; the refactor renames the package to `optimal_auctions/` but keeps most identifiers stable for traceability.

| Math object | Code identifier | Module |
|-------------|-----------------|--------|
| $X^i$ as `[[a₁,b₁], …, [a_K,b_K]]` | `V` | [approximation.py](../approx/approximation.py) |
| $T$ | `T` | [approximation.py](../approx/approximation.py) |
| $N$ | `n_buyers` | [approximation.py](../approx/approximation.py) |
| $K$ | `n_grades` | [approximation.py](../approx/approximation.py) |
| $c_j$ | `costs` | [approximation.py](../approx/approximation.py) |
| Per-dim grid for $X_T$ | `V_T_list` | [approximation.py](../approx/approximation.py), [discrete.py](../approx/discrete.py) |
| Full discretized $X_T$ | `V_T` | [approximation.py](../approx/approximation.py) |
| $\epsilon_j$ (grid spacing) | `eps` | [discrete.py](../approx/discrete.py) |
| $\hat f$ | `f_hat` class instance | [discrete.py](../approx/discrete.py) |
| $f^i(x)$ (PDFs) | `f` (list of callables) | [approximation.py](../approx/approximation.py) |
| Joint $F$ structure | `corr` (callable combining PDFs) | [approximation.py](../approx/approximation.py) |
| $Q_j^i(v)$ (LP variables) | `Q_vars[j][v_index]` | [approximation.py](../approx/approximation.py) |
| $U^i(v)$ (LP variables) | `U_vars[v_index]` | [approximation.py](../approx/approximation.py) |
| Solved $Q$ values | `_Q_values` or property `Q` | [approximation.py](../approx/approximation.py) |
| Solved $U$ values | `_U_values` or property `U` | [approximation.py](../approx/approximation.py) |
| Border constraint | `make_border_expr_from_name` | [constraints.py](../approx/constraints.py) |
| IIC constraint | `make_ic_expr_from_name` | [constraints.py](../approx/constraints.py) |
| Local region size $L$ | `net_size` | [approximation.py](../approx/approximation.py), [oracle.py](../approx/oracle.py) |
| Star-pattern neighbors | hardcoded offsets in `_check_ic` | [oracle.py](../approx/oracle.py) |
| Optimal value | `opt` attribute | [approximation.py](../approx/approximation.py) |
| Convergence flag | `converged` attribute | [approximation.py](../approx/approximation.py) |

## References

- **PhD thesis chapter (canonical)**: [auctions.tex](../../phd/sections/auctions.tex). Included as Chapter 3 of [thesis.tex](../../phd/thesis.tex).
- **Working paper (publication form)**: Kushnir & Michelson 2022, "On the Asymptotic Optimality of the Exclusive Buyer Mechanism." [arXiv:2207.01664](https://arxiv.org/abs/2207.01664).
- **Belloni, Lopomo, Wang 2010** — the foundational discretized-LP separation-oracle algorithm this work extends. [PDF](https://faculty.fuqua.duke.edu/~abn5/MMD.pdf).
- **Border 1991** — the feasibility characterization that makes (B) polynomial in $|X_T|$. *Implementation of Bayesian Equilibrium and the Separation of Market Functions*, Cambridge University Press.
- **Pavlov 2011** — closed-form optimal mechanism for the $K=2, N=1$ symmetric uniform case (Setting 1a's $N=1$ degenerate case). *"Optimal Mechanism for Selling Two Goods,"* The B.E. Journal of Theoretical Economics.
- **Myerson 1981** — virtual values and the envelope/revelation principle ($K=1$ case). *Mathematics of Operations Research*.
- **Daskalakis, Deckelbaum, Tzamos 2017** — multi-item mechanism design complexity. Referenced in the thesis for Setting 2 (Beta(1,2) example).
- **Exclusive Buyer Mechanism code (independent oracle)**: [phd/ch_auctions_simulations/ebm_code/ebm.py](../../phd/ch_auctions_simulations/ebm_code/ebm.py).
- **Sister doc**: [benchmarks.md](./benchmarks.md) — 6 canonical settings, expected revenue values, and open ambiguities surfaced during baseline extraction.
