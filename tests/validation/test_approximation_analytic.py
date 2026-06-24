"""Direct analytic tests of the approximation LP against closed forms.

Unlike the baseline regression (which compares the LP to its own stored historical
output — partly circular) these pin the approximation to an *independent* analytic
result, so a subtly wrong solver cannot pass by reproducing its own past mistake.

Pavlov single buyer (N=1, K=2, X~U[0,1]^2): the optimal mechanism revenue is
2/(3*sqrt(3)) ~= 0.38490, attained by a deterministic posted price (the buyer
takes their better good at p* = 1/sqrt(3)). The LP is free to randomise (Q in
[0,1]) but does not beat the deterministic value at this support — randomisation
would only help for a shifted support such as U[2,3]^2 (cf. setting 1b).

Myerson single good (N=1, X~U[0,1]): the regular-distribution reserve is 1/2 with
revenue 0.25. The approximation is hard-coded to K=2, so we degenerate the second
good by making it unprofitable (cost > max value); the LP never allocates it and
the problem reduces to single-good Myerson on good 1.

Both LP revenues carry the same O(1/T) reserve-kink error as the EBM, so we
Richardson-extrapolate them (see docs/extrapolation.md).
"""

import logging

import numpy as np
import pytest

from optimal_auctions import OptimalAuctionApproximation


@pytest.mark.slow
def test_approximation_recovers_pavlov_single_buyer():
    rev_star = 2 / (3 * np.sqrt(3))  # 0.3849001794597505

    def approx_revenue(T):
        a = OptimalAuctionApproximation(
            n_buyers=1, V=[[0, 1], [0, 1]], costs=[0, 0], T=T, log_level=logging.WARNING
        )
        a.run()
        return a.opt

    rev_inf = (30 * approx_revenue(30) - 20 * approx_revenue(20)) / 10
    assert rev_inf == pytest.approx(rev_star, abs=1.5e-3)


@pytest.mark.slow
def test_approximation_recovers_myerson_single_good():
    # Degenerate good 2 (cost above its max value) so the LP never sells it; the
    # problem reduces to single-good Myerson on good 1.
    def solve(T):
        a = OptimalAuctionApproximation(
            n_buyers=1,
            V=[[0, 1], [0, 1]],
            costs=[0, 2],
            T=T,
            force_symmetric=False,
            log_level=logging.WARNING,
        )
        a.run()
        return a

    a20, a30 = solve(20), solve(30)
    # good 2 must be genuinely unallocated for the reduction to hold
    assert max(q[1] for q in a30.Q) == pytest.approx(0.0, abs=1e-9)
    # 1-D reserve kink is a clean 1/T -> Richardson is exact at 0.25
    rev_inf = (30 * a30.opt - 20 * a20.opt) / 10
    assert rev_inf == pytest.approx(0.25, abs=1e-5)
