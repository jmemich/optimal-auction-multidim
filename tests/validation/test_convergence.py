"""Convergence test: approximation revenue -> EBM revenue as T -> infinity.

This is the chapter's central optimality conjecture: the LP approximation (an
upper bound on the optimal-mechanism revenue, decreasing in T) converges to the
exclusive-buyer mechanism (the conjectured optimum) as the grid is refined.

Both quantities are grid-dependent and approach their common continuum limit
*from above*, so it is wrong to compare the approximation at finite T against a
single EBM(T=50) point. Instead we Richardson-extrapolate BOTH sequences in
h = 1/T (leading error c/T; see docs/extrapolation.md) and check that the two
independent methods -- a cutting-plane LP and a Vickrey-auction integral -- agree
on the limit.

Empirically the two R_inf estimates agree to < 1e-3 for settings 1a/2/3 (using
the T=20,30 pair); the 3e-3 tolerance here leaves margin without being vacuous.
"""

import logging

import numpy as np
import pytest
from scipy.stats import beta, uniform

from optimal_auctions import OptimalAuctionApproximation
from optimal_auctions.validation import ExclusiveBuyerMechanism

# Two grids for the 1/T extrapolation. T=30 dominates the runtime (~16 s each).
T_LO, T_HI = 20, 30
RICHARDSON_TOL = 3e-3

_beta12 = beta(1, 2)


def _corr_u(v, fs):
    # setting 3 joint density f(x1, x2) = x1 + x2 (matches tests/benchmarks)
    return v[0] + v[1]


# id -> (approximation kwargs, EBM joint density). All on [0,1]^2, N=2, costs 0.
SETTINGS = {
    "1a": (dict(f=None), lambda x: 1.0),
    "2": (dict(f=[_beta12.pdf, _beta12.pdf]), lambda x: _beta12.pdf(x[0]) * _beta12.pdf(x[1])),
    "3": (dict(f=[uniform.pdf, uniform.pdf], corr=_corr_u), lambda x: x[0] + x[1]),
}


def _richardson(t1, r1, t2, r2):
    """Limit of r(T) = R_inf + c/T from two points (cancels the c/T term)."""
    return (t2 * r2 - t1 * r1) / (t2 - t1)


def _approx_revenue(T, approx_kwargs):
    approx = OptimalAuctionApproximation(
        n_buyers=2,
        V=[[0, 1], [0, 1]],
        costs=[0, 0],
        T=T,
        log_level=logging.WARNING,
        **approx_kwargs,
    )
    approx.run()
    return approx.opt


def _ebm_optimal_revenue(T, f):
    ebm = ExclusiveBuyerMechanism(X=[[0, 1], [0, 1]], c=[0, 0], T=T, f=f)
    diagonal = [[v, v] for v in np.linspace(0, 1, 4 * T + 1)]  # symmetric reserve search
    return ebm.optimal_revenue(price_grid=diagonal)[0]


@pytest.mark.slow
@pytest.mark.parametrize("sid", list(SETTINGS))
def test_approximation_converges_to_ebm(sid):
    approx_kwargs, ebm_f = SETTINGS[sid]

    approx_inf = _richardson(
        T_LO, _approx_revenue(T_LO, approx_kwargs), T_HI, _approx_revenue(T_HI, approx_kwargs)
    )
    ebm_inf = _richardson(
        T_LO, _ebm_optimal_revenue(T_LO, ebm_f), T_HI, _ebm_optimal_revenue(T_HI, ebm_f)
    )

    assert abs(approx_inf - ebm_inf) < RICHARDSON_TOL, (
        f"setting {sid}: approximation R_inf={approx_inf:.6f} and EBM R_inf={ebm_inf:.6f} "
        f"differ by {abs(approx_inf - ebm_inf):.2e} (tol={RICHARDSON_TOL:.0e})"
    )
