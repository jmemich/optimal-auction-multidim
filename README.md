# Approximating optimal multi-dimensional auctions

This repository contains the code for the approximation algorithm developed in [Kushnir & Michelson](https://arxiv.org/abs/2207.01664).

## Installation

From this directory run:

```bash
pip install -r requirements.txt
python setup.py install
```

## Examples

```python
from auction import OptimalAuctionApproximation as Approximation
approx = Approximation(n_buyers=2, V=[[0,1],[0,1]], costs=[0,0], T=10)
approx.run()
```

## Additional References

- [Belloni et al.](https://faculty.fuqua.duke.edu/~abn5/MMD.pdf)
- [Daskalakis et al.](https://sci-hub.se/https://onlinelibrary.wiley.com/doi/epdf/10.3982/ECTA12618)
