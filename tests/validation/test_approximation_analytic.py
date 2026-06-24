"""Direct analytic tests of the approximation LP against closed forms.

Unlike the baseline regression (which compares the LP to its own stored historical
output — partly circular) these pin the approximation to an *independent* analytic
result, so a subtly wrong solver cannot pass by reproducing its own past mistake.

Pavlov single buyer (N=1, K=2, X~U[0,1]^2): the optimal mechanism revenue is
2/(3*sqrt(3)) ~= 0.38490, attained by a deterministic posted price (the buyer
takes their better good at p* = 1/sqrt(3)). The LP is free to randomise (Q in
[0,1]) but does not beat the deterministic value at this support — randomisation
would only help for a shifted support such as U[2,3]^2 (cf. setting 1b). The LP's
revenue carries the same O(1/T) reserve-kink error as the EBM, so we
Richardson-extrapolate it (see docs/extrapolation.md).
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
