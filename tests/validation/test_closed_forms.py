"""Closed-form sanity checks for the EBM oracle at the degenerate N=1 cases.

These validate the oracle's allocation/integration against analytic results, in
the no-competition (single-buyer) limit:

* Pavlov (K=2, N=1, X~U[0,1]^2): the single buyer takes their better good at a
  posted price p; revenue p*(1-p^2) is maximised at p* = 1/sqrt(3) with value
  2/(3*sqrt(3)) ~= 0.38490.
* Myerson (K=1, N=1, X~U[0,1]): the monopoly / regular-distribution reserve is
  p* = 1/2 with revenue p*(1-p*) = 0.25.

Both integrands are kinked at the reserve boundary, so the finite-T oracle value
sits above the limit with an O(1/T) error; we Richardson-extrapolate it away (see
docs/extrapolation.md). The Myerson error is a single clean 1/T (one-dimensional
step), so its extrapolation is exact; Pavlov's L-shaped kink leaves a ~3e-4
residual.
"""

import numpy as np
import pytest

from optimal_auctions.validation import ExclusiveBuyerMechanism


def _richardson(t1, r1, t2, r2):
    return (t2 * r2 - t1 * r1) / (t2 - t1)


def test_pavlov_single_buyer_unit_square():
    p_star = 1 / np.sqrt(3)  # 0.57735
    rev_star = 2 / (3 * np.sqrt(3))  # 0.38490

    def optimum(T):
        ebm = ExclusiveBuyerMechanism(X=[[0, 1], [0, 1]], c=[0, 0], T=T, f=lambda x: 1.0, N=1)
        diagonal = [[v, v] for v in np.linspace(0, 1, 8 * T + 1)]
        return ebm.optimal_revenue(price_grid=diagonal)

    r30, _ = optimum(30)
    r50, p50 = optimum(50)

    # optimal reserve recovered to grid resolution
    assert abs(p50[0] - p_star) < 0.02
    # revenue: extrapolate the O(1/T) reserve-kink error to the closed form
    assert _richardson(30, r30, 50, r50) == pytest.approx(rev_star, abs=1.5e-3)


def test_myerson_single_buyer_single_good():
    def optimum(T):
        ebm = ExclusiveBuyerMechanism(X=[[0, 1]], c=[0], T=T, f=lambda x: 1.0, N=1)
        grid = [[v] for v in np.linspace(0, 1, 2 * T + 1)]
        return ebm.optimal_revenue(price_grid=grid)

    r20, _ = optimum(20)
    r50, p50 = optimum(50)

    # Myerson reserve for the regular U[0,1] distribution
    assert p50[0] == pytest.approx(0.5, abs=1e-9)
    # 1-D reserve kink is a clean 1/T error -> Richardson is exact
    assert _richardson(20, r20, 50, r50) == pytest.approx(0.25, abs=1e-6)
