"""Regression tests for the EBM revenue oracle.

Pins the oracle to the published thesis EBM figures (reproduced via the
second-price simulation), so a future change to the oracle that drifts from the
thesis numbers fails loudly.
"""

import pytest

from optimal_auctions.validation import ExclusiveBuyerMechanism


def test_ebm_1a_pavlov_unit_square():
    # Setting 1a (Pavlov U[0,1]^2): N=2, costs=0, independent uniform density.
    # Thesis reserve p=[0.6, 0.6] (read off the allocation graph), T=50.
    ebm = ExclusiveBuyerMechanism(X=[[0, 1], [0, 1]], c=[0, 0], T=50, f=lambda x: 1.0)
    rev = ebm.revenue([0.6, 0.6])
    # phd/.../Full analysis.ipynb produced 0.5890515691960928 for this call.
    assert rev == pytest.approx(0.5890515691960928, abs=1e-9)


def test_ebm_optimal_reserve_near_diagonal():
    # The optimal symmetric reserve should be in a sensible interior range and
    # earn at least the thesis p=[0.6,0.6] revenue (which was eyeballed, not
    # optimised).
    ebm = ExclusiveBuyerMechanism(X=[[0, 1], [0, 1]], c=[0, 0], T=40, f=lambda x: 1.0)
    diag = [[v, v] for v in [i / 40 for i in range(41)]]
    rev, p = ebm.optimal_revenue(price_grid=diag)
    assert 0.4 < p[0] < 0.8
    assert rev >= ebm.revenue([0.6, 0.6]) - 1e-9
