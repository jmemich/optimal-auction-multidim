import numpy as np

from optimal_auctions.util import symmetric_ix

BORDER_PREFIX = "border"
IC_PREFIX = "ic"


# we make a class to ensure we can use set-like operations on constraints
class Constraint:
    def __init__(self, name, expr=None, status=None):
        self.name = name
        self.expr = expr
        self.status = status

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return self.name


def ic_lhs_minus_rhs(Q, U, T, grades, i, v_i, j, v_j, force_symmetric):
    rhs = 0
    if force_symmetric:
        rhs += Q[0][j] * (v_i[0] - v_j[0])
        rhs += Q[0][symmetric_ix(j, T)] * (v_i[1] - v_j[1])
    else:
        for k in grades:
            rhs += Q[k][j] * (v_i[k] - v_j[k])
    lhs = U[i] - U[j]
    return lhs - rhs


def border_lhs_minus_rhs(T, V_T, V_T_subset, Q, n_buyers, grades, f_hat, force_symmetric):
    lhs = 0
    if force_symmetric:
        for v_ix in V_T_subset:
            inner = Q[0][v_ix] + Q[0][symmetric_ix(v_ix, T)]
            inner *= f_hat[v_ix]
            lhs += inner
    else:
        for v_ix in V_T_subset:
            inner = 0
            for j in grades:
                inner += Q[j][v_ix]
            inner *= f_hat[v_ix]
            lhs += inner
    lhs *= n_buyers

    rhs = 0
    V_T_setdiff = np.setdiff1d(np.arange(0, len(V_T), 1), V_T_subset)
    for v_ix in V_T_setdiff:
        rhs += f_hat[v_ix]
    rhs = np.power(rhs, n_buyers)
    rhs = 1 - rhs

    return lhs - rhs
