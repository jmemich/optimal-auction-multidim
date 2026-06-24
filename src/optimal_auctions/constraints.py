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


def lower_left_quadrant(ix, T, net_size, n_dims=2):
    # Indices of the lower-left quadrant of the type grid anchored at `ix`, used
    # to widen the local IC region as `net_size` grows (empty until net_size >= 2).
    if net_size < 2:
        return []

    if n_dims == 1:
        # 1-D type grid: the "quadrant" is just the window below ix.
        return list(range(max(0, ix - net_size), ix + 1))

    i = int(ix / (T + 1))  # row
    j = ix % (T + 1)  # col

    low_i = np.max([i - net_size, 0])
    low_j = np.max([j - net_size, 0])

    grid = np.zeros((T + 1, T + 1))
    for x in range(low_i, i + 1):
        for y in range(low_j, j + 1):
            grid[x, y] = 1

    # flatten and pull the indices we set
    ixs = np.where(grid.reshape(-1) > 0.5)[0]
    return list(ixs)


def star_indices(i, T, n_V_T, n_dims=2):
    """The immediate-neighbour 'star' around flat index `i`, clipped in-range.

    For a 1-D type grid (``n_dims == 1``) this is just ``{i-1, i+1}``. For the
    2-D ``(T+1)x(T+1)`` grid it is the 8-neighbour pattern

        x x x
        x i x
        x x x

    with offsets assuming a row stride of (T+1): e.g. below-right is -(T+1)+1 == -T.
    """
    if n_dims == 1:
        star = [i - 1, i + 1]
    else:
        star = [
            i + T,  # above-left
            i + T + 1,  # above
            i + T + 2,  # above-right
            i - 1,  # left
            i + 1,  # right
            i - T - 2,  # below-left
            i - T - 1,  # below
            i - T,  # below-right
        ]
    return [ix for ix in star if 0 <= ix < n_V_T]


def local_ic_indices(i, T, net_size, n_V_T, n_dims=2):
    """Indices of the local IC region around type `i`: the star pattern (always)
    plus the lower-left quadrant (once net_size >= 2). Returns sorted, unique,
    in-range indices. Single source of truth shared by the separation oracle and
    the incremental-row pre-seed in `OptimalAuctionApproximation.run`. `n_dims` is
    the number of quality grades (1 or 2).
    """
    return sorted(
        set(star_indices(i, T, n_V_T, n_dims) + lower_left_quadrant(i, T, net_size, n_dims))
    )


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
