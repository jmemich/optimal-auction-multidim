"""Regression tests for the EBM revenue oracle.

Pins the oracle to the thesis EBM figures (the second-price simulation
reproduced via `ExclusiveBuyerMechanism.revenue`) at the thesis reserve prices
and T=50, so a future change that drifts from these values fails loudly.

Densities, supports, costs and reserve prices are from
`phd/ch_auctions_simulations/ebm_code/Full analysis.ipynb`.

Setting 5 (truncnorm) is the one case where the thesis figure (2.57921) is the
*unnormalised* trapezoidal value the author flagged "isn't normalized... don't
use"; the correct normalised value is pinned here instead. For every other
setting the density integrates to 1 on the grid (W=1), so normalised and
continuous coincide and the published value is reproduced exactly.
"""

import pytest
from scipy.stats import beta, truncnorm

from optimal_auctions.validation import ExclusiveBuyerMechanism

_beta12 = beta(1, 2)
_tn1 = truncnorm((2 - 2.3) / 1, (3 - 2.3) / 1, loc=2.3, scale=1)
_tn2 = truncnorm((2 - 2.8) / 0.2, (3 - 2.8) / 0.2, loc=2.8, scale=0.2)

# (id, X, costs, density f(x), reserve p, expected revenue, note)
SETTINGS = [
    ("1a", [[0, 1], [0, 1]], [0, 0], lambda x: 1.0, [0.6, 0.6], 0.589052),
    ("1b", [[2, 3], [2, 3]], [0, 0], lambda x: 1.0, [2.15, 2.15], 2.534499),
    (
        "2",
        [[0, 1], [0, 1]],
        [0, 0],
        lambda x: _beta12.pdf(x[0]) * _beta12.pdf(x[1]),
        [0.4, 0.4],
        0.385045,
    ),
    ("3", [[0, 1], [0, 1]], [0, 0], lambda x: x[0] + x[1], [0.65, 0.65], 0.676192),
    ("4", [[6, 8], [9, 11]], [0.9, 5], lambda x: 0.25, [6, 10.2], 5.805032),
    # Setting 5: corrected (normalised) value; thesis published 2.57921 = this * W^2.
    (
        "5",
        [[2, 3], [2, 3]],
        [0, 0],
        lambda x: _tn1.pdf(x[0]) * _tn2.pdf(x[1]),
        [2.56, 2.5],
        2.7011591052647033,
    ),
]


@pytest.mark.parametrize("sid,X,c,f,p,expected", SETTINGS, ids=[s[0] for s in SETTINGS])
def test_ebm_setting_revenue(sid, X, c, f, p, expected):
    ebm = ExclusiveBuyerMechanism(X=X, c=c, T=50, f=f)
    assert ebm.revenue(p) == pytest.approx(expected, abs=1e-5)


def test_ebm_weights_normalised():
    # The discrete type distribution must sum to 1 (it is the approximation's
    # f_hat); this is what makes revenue a proper expectation.
    ebm = ExclusiveBuyerMechanism(
        X=[[2, 3], [2, 3]], c=[0, 0], T=40, f=lambda x: _tn1.pdf(x[0]) * _tn2.pdf(x[1])
    )
    assert ebm._weights.sum() == pytest.approx(1.0)


def test_ebm_optimal_reserve_near_diagonal():
    # Optimal symmetric reserve for 1a is interior and earns at least the
    # thesis (eyeballed, non-optimised) p=[0.6,0.6] revenue.
    ebm = ExclusiveBuyerMechanism(X=[[0, 1], [0, 1]], c=[0, 0], T=40, f=lambda x: 1.0)
    diag = [[v, v] for v in [i / 40 for i in range(41)]]
    rev, p = ebm.optimal_revenue(price_grid=diag)
    assert 0.4 < p[0] < 0.8
    assert rev >= ebm.revenue([0.6, 0.6]) - 1e-9
