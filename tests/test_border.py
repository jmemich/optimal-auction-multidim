from itertools import chain, combinations

import numpy as np

from optimal_auctions import OptimalAuctionApproximation as Approximation
from optimal_auctions.constraints import border_lhs_minus_rhs


def test_basic_border():
    test = Approximation(n_buyers=2, V=[[0, 1], [0, 1]], costs=[0, 0], T=2)
    test.run()

    ixs = np.arange(0, len(test.V_T)).tolist()
    # every non-empty subset of the type indices (drop the leading empty set)
    subsets = list(chain.from_iterable(combinations(ixs, r) for r in range(len(ixs) + 1)))[1:]
    for subset in subsets:
        # NOTE we re-formatted the structure of Q....
        Q_ = test._Q_values
        val = border_lhs_minus_rhs(
            test.T,
            test.V_T,
            subset,
            Q_,
            test.n_buyers,
            test.grades,
            test.f_hat,
            test.force_symmetric,
        )
        is_close = np.isclose(val, 0)
        if not is_close:
            assert val <= 0, f"Border constraint subset A=`{subset!s}` failed!"
