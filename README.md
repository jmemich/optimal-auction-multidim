# optimal-auction-multidim

Python package implementing the LP-based approximation algorithm for optimal multi-dimensional auctions from [Kushnir & Michelson 2022](https://arxiv.org/abs/2207.01664) (working paper) and Chapter 3 of James Michelson's PhD thesis.

> The repo is **under active refactor** on branch `ai-performance-refactor`. The current `master` reflects the pre-refactor state (`approx` package, ortools backend). The refactor renames the package to `optimal_auctions`, switches the LP backend to HiGHS via `highspy`, and adds regression baselines + reference docs. See [docs/](./docs/) for the math + benchmarks spec; see commit history on `ai-performance-refactor` for progress.

## Installation

Requires Python 3.12+. Install [uv](https://docs.astral.sh/uv/) (one-time):

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
  baselines.json        # frozen historical-truth contract (24 entries)
```

## References

- **Kushnir & Michelson 2022** — working paper: <https://arxiv.org/abs/2207.01664>
- **Belloni, Lopomo, Wang 2010** — foundational algorithm: <https://faculty.fuqua.duke.edu/~abn5/MMD.pdf>
- **Daskalakis, Deckelbaum, Tzamos** — multi-item mechanism design: <https://onlinelibrary.wiley.com/doi/epdf/10.3982/ECTA12618>
