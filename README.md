# optimal-auction-multidim

Python package implementing the LP-based approximation algorithm for optimal
multi-dimensional auctions — selling a single object of several quality grades to
multiple buyers with multi-dimensional private types. It computes the
revenue-maximizing mechanism by discretizing the type space and solving the
resulting linear program with a structured, local incentive-compatibility
separation oracle (an extension of the Belloni–Lopomo–Wang 2010 method), using
the HiGHS solver via `highspy`.

The algorithm and the economic results it produces are from:

- **Kushnir & Michelson**, *Optimal Multi-Dimensional Auctions: Conjectures and
  Simulations* (working paper) — [arXiv:2207.01664](https://arxiv.org/abs/2207.01664)
- **James Michelson**, *Essays in Design Economics* (PhD thesis, Carnegie Mellon
  University; the auction chapter) —
  [kilthub.cmu.edu/.../Essays_in_Design_Economics](https://kilthub.cmu.edu/articles/thesis/Essays_in_Design_Economics/27196314)

The package (`optimal_auctions`) ships with the math + benchmarks specification in
[docs/](./docs/), a regression suite pinning the six canonical thesis settings, and
the convergence/extrapolation analysis described below.

## State of the conjectures

The central conjecture is the **asymptotic optimality of the exclusive buyer
mechanism (EBM)**: in the settings where the EBM is conjectured optimal (Settings
1a, 2, 3), the approximation's revenue should converge to the EBM revenue as the
grid resolution $T \to \infty$. Both quantities are grid-dependent and approach a
common limit *from above*, so the right test compares their **extrapolated limits**,
not raw revenues at a single $T$.

The extrapolation work makes this test cheap and quantitative. On the canonical
Pavlov setting (1a), the approximation's discretization error is a clean $c/T$
(verified by decrement-halving and held-out back-prediction), so Richardson
extrapolation from two modest runs ($T = 20, 30$, each well under a minute) recovers
$R_\infty \approx 0.585$ — matching the EBM's own extrapolated limit to
$\sim 10^{-3}$, which **sustains the optimality conjecture for 1a** without a
multi-hour fine-grid run. `tests/validation/test_convergence.py` codifies this
(Settings 1a/2/3, tolerance $3\times10^{-3}$). The agreement *floors* near
$10^{-3}$ — not because of the approximation, but because the EBM revenue is a
trapezoidal integral of a *kinked* Vickrey-payment integrand, whose error is
$O(1/T)$ with irregular sub-leading terms that Richardson cannot super-converge.
Tightening below $10^{-3}$ would need a kink-aware EBM integrator (tracked in
[TODO.md](./TODO.md)). See [docs/extrapolation.md](./docs/extrapolation.md) for the
full derivation, scaling laws, and the cost wall ($T \lesssim 50$ is the "minutes"
regime in the current code).

The other settings behave as the thesis predicts: 1b (and likely the asymmetric
Belloni setting 4) require *randomization*, so the EBM is strictly beaten there, and
the $N$-invariance of the exclusion-region reserves (Settings 2, 3, 5) holds across
$N \in \{1, 2, 3\}$.

## Installation

Requires Python 3.12+. The package is distributed via GitHub (not PyPI).

### As a library (use it in your own project)

Install straight from GitHub into any environment — no uv required:

```bash
pip install git+https://github.com/jmemich/optimal-auction-multidim
```

Or clone first, then install (editable shown; drop `-e` for a fixed install):

```bash
git clone https://github.com/jmemich/optimal-auction-multidim
pip install -e optimal-auction-multidim
```

Then import as usual:

```python
from optimal_auctions import OptimalAuctionApproximation
```

### For development (hacking on this repo)

Install [uv](https://docs.astral.sh/uv/) (one-time):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then from the repo root:

```bash
uv sync --all-extras   # creates .venv, installs runtime + dev deps
uv run pytest          # run tests
```

## Quick example

```python
from optimal_auctions import OptimalAuctionApproximation as Approximation

approx = Approximation(n_buyers=2, V=[[0, 1], [0, 1]], costs=[0, 0], T=10)
approx.run()
print(approx.opt)        # revenue
print(approx.converged)  # bool
```

## Repository layout

```
src/optimal_auctions/   # the package
tests/                  # pytest test suite
scripts/                # one-shot utilities (e.g. extract_baselines.py)
docs/
  math_reference.md     # canonical math spec for the LP and algorithm
  benchmarks.md         # the 6 canonical settings + thesis revenue tables
  extrapolation.md      # scaling laws + Richardson convergence analysis
  baselines.json        # frozen historical-truth contract (24 entries)
```

## References

- **Kushnir & Michelson** — *Optimal Multi-Dimensional Auctions: Conjectures and Simulations* (working paper): <https://arxiv.org/abs/2207.01664>
- **Michelson** — *Essays in Design Economics* (PhD thesis, Carnegie Mellon University): <https://kilthub.cmu.edu/articles/thesis/Essays_in_Design_Economics/27196314>
- **Belloni, Lopomo, Wang 2010** — foundational algorithm: <https://faculty.fuqua.duke.edu/~abn5/MMD.pdf>
- **Daskalakis, Deckelbaum, Tzamos** — multi-item mechanism design: <https://onlinelibrary.wiley.com/doi/epdf/10.3982/ECTA12618>
