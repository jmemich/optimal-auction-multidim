import numpy as np

from itertools import chain, combinations

from auction import OptimalAuctionApproximation as Approximation
from auction.constraints import border_lhs_minus_rhs


def _powerset(iterable):
    s = list(iterable)
    powerset = chain.from_iterable(
        combinations(s, r) for r in range(len(s) + 1))
    return list(powerset)[1:]  # strip of leading emptyset element


def test_basic_border():
    test = Approximation(
        n_buyers=2,
        V=[[0, 1], [0, 1]],
        costs=[0, 0],
        T=2
    )
    test.run()

    ixs = np.arange(0, len(test.V_T)).tolist()
    subsets = _powerset(ixs)
    for subset in subsets:
        # NOTE we re-formatted the structure of Q....
        Q_ = [[val.solution_value() for val in Q_j] for Q_j in test.Q_vars]
        val = border_lhs_minus_rhs(
            test.T, test.V_T, subset, Q_, test.n_buyers, test.grades,
            test.f_hat, test.force_symmetric)
        is_close = np.isclose(val, 0)
        if not is_close:
            assert val <= 0, \
                "Border constraint subset A=`%s` failed!" % str(subset)
