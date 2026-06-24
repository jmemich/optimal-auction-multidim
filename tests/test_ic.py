import numpy as np
import pytest

from optimal_auctions import OptimalAuctionApproximation as Approximation
from optimal_auctions.constraints import ic_lhs_minus_rhs


@pytest.mark.parametrize("test_local", [True, False])
def test_basic_ic(test_local):
    test = Approximation(
        n_buyers=2, V=[[0, 1], [0, 1]], costs=[0, 0], T=3, check_local_ic=test_local
    )
    test.run()

    Q_ = test._Q_values
    U_ = test._U_values
    for i, v_i in enumerate(test.V_T):
        for j, v_j in enumerate(test.V_T):
            val = ic_lhs_minus_rhs(
                Q_, U_, test.T, test.grades, i, v_i, j, v_j, test.force_symmetric
            )
            is_close = np.isclose(val, 0)
            if not is_close:
                assert val >= 0, f"IC constraint violated for v_i x v_j = `{v_i!s}` x `{v_j!s}`"


def test_Q_monotone():
    test = Approximation(n_buyers=2, V=[[0, 1], [0, 1]], costs=[0, 0], T=2)
    test.run()

    Q = np.array(test.Q)

    for v_i in range(test.T + 1):
        v_i_fixed = np.zeros((test.T + 1, test.T + 1))
        v_i_fixed[v_i] = 1

        # fix v1, check Q2 is monotonic in v2
        v1_fixed_ix = np.where(v_i_fixed.astype(int).reshape(-1) == 1)[0]
        v1_fixed_Q2 = Q[v1_fixed_ix].T[1]

        current = 0
        for val in v1_fixed_Q2:
            if not np.isclose(val, current):
                assert val >= current, f"Q2s not monotonic for v_1_i = {v_i}"
                current = val

        # fix v2, check Q1 is monotonic in v1
        v2_fixed_ix = np.where(v_i_fixed.T.astype(int).reshape(-1) == 1)[0]
        v2_fixed_Q1 = Q[v2_fixed_ix].T[0]

        current = 0
        for val in v2_fixed_Q1:
            if not np.isclose(val, current):
                assert val >= current, f"Q1s not monotonic for v_2_i = {v_i}"
                current = val
