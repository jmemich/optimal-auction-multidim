"""Exclusive-Buyer Mechanism (EBM) revenue oracle.

The EBM is the conjectured-optimal mechanism in most of the thesis settings. Its
revenue is the expected take of an efficient **second-price (Vickrey) auction with
reserve prices** among ``N`` buyers over ``J <= 2`` quality grades, computed by
numerical integration over the type grid.

Mechanism. At reserve prices ``p = (p_0, ..., p_{J-1})`` a buyer with type ``x``
has surplus ``beta_j = x_j - p_j`` for grade ``j`` and chooses their best grade,
``beta = max_j beta_j`` (participating only if ``beta >= 0``). The buyer with the
higher best-surplus wins, is served their preferred grade, and pays Vickrey-style

    payment = p[winner_grade] + max(loser_best_surplus, 0) - cost[winner_grade].

Revenue is the expectation of that payment over independent buyer types.

This reproduces the thesis EBM figures exactly (Setting 1a, ``p=[0.6, 0.6]``,
``T=50``: 0.5890515692, matching the published 0.5890515691960928 to 1e-11). It is
an *independent* oracle for ``OptimalAuctionApproximation`` -- no LP, no cutting
planes -- so agreement is a non-circular check on the approximation, and the
chapter's optimality conjecture is the statement that the approximation converges
to this value as ``T -> inf``.

Scope. Implemented for ``N = 2`` (every thesis setting); the explicit two-buyer
expectation is what generated the published numbers. Generalising to ``N > 2``
would require an order-statistic / ``F^{N-1}`` formulation rather than the direct
pairwise integral here.

Ported from the second-price simulation ``ebm_revenue`` in
``phd/ch_auctions_simulations/ebm_code/Full analysis.ipynb`` (cell 9). The
posted-price ``obj`` form in the original ``ebm.py`` computed a *different* (and
unused) quantity and is intentionally not reproduced here.
"""

import itertools

import numpy as np


class ExclusiveBuyerMechanism:
    """Numerical EBM (second-price-with-reserve) revenue oracle for N=2, J<=2.

    Parameters
    ----------
    X : list[list[float]]
        Per-grade type support, e.g. ``[[0, 1], [0, 1]]``. Supports must share a
        common width (square only).
    c : list[float]
        Per-grade production costs.
    T : int
        Grid resolution; the integration grid has ``(T + 1)**J`` points.
    f : callable
        Joint type density ``f(x) -> float`` over a length-J point. For
        independent ``U[0,1]**2`` this is the constant ``1.0``.
    N : int, default=2
        Number of buyers. Only ``N == 2`` is supported.
    """

    def __init__(self, X, c, T, f, N=2):
        self.N = int(N)
        if self.N != 2:
            raise NotImplementedError("EBM revenue oracle is implemented for N=2 only")

        self.X = list(X)
        self.J = len(self.X)
        if self.J != 2:
            raise NotImplementedError("EBM oracle supports exactly J=2 quality grades")
        self.c = list(c)
        assert len(self.c) == self.J, "len(c) != J"
        self.T = int(T)
        self.f = f

        widths = [Xj[1] - Xj[0] for Xj in self.X]
        assert len(np.unique(widths)) == 1, "non-square supports not supported"
        self.delta = widths[0] / T

        self.axes = [np.arange(Xj[0], Xj[1] + 1e-10, self.delta) for Xj in self.X]
        self.X_iter = np.array(list(itertools.product(*self.axes)))  # (M, J)

        # Discrete type distribution over the (T+1)**J grid: trapezoidal weights
        # (boundary nodes get half weight per dimension) NORMALISED to sum to 1.
        # This is exactly the approximation's `f_hat`. Normalisation is required
        # because EBM revenue is an expectation over a probability distribution;
        # it is a no-op when the density already integrates to 1 on the grid
        # (uniform / piecewise-linear -- settings 1a-4, where trapezoid and f_hat
        # coincide), and the correct renormalisation for a curved density whose
        # trapezoidal mass differs from 1 (truncnorm, setting 5).
        self._weights = self._type_distribution()

    def _type_distribution(self):
        n = self.T + 1
        w = np.empty(len(self.X_iter))
        for i, x in enumerate(self.X_iter):
            adj = 1.0
            if (i // n) in (0, self.T):  # first grid coordinate on a boundary
                adj *= 0.5
            if (i % n) in (0, self.T):  # second grid coordinate on a boundary
                adj *= 0.5
            w[i] = self.f(tuple(x)) * adj
        return w / w.sum()

    def revenue(self, p):
        """Expected second-price-with-reserve revenue at reserve prices ``p``.

        The expectation is taken over the normalised discrete type distribution
        (``self._weights``, i.e. ``f_hat``).
        """
        p = np.asarray(p, dtype=float)
        assert len(p) == self.J, "len(p) != J"
        w = self._weights

        beta = self.X_iter - p  # (M, J)
        best = beta.max(axis=1)  # best surplus per type
        grade = beta.argmax(axis=1)  # preferred grade per type
        price_at = p[grade]  # reserve the buyer would pay
        cost_at = np.asarray(self.c)[grade]  # cost of that grade

        # Pairwise over (buyer i, buyer k): higher best-surplus wins (ties -> i).
        bi, bk = best[:, None], best[None, :]
        win_i = bi >= bk
        best_win = np.where(win_i, bi, bk)
        best_los = np.where(win_i, bk, bi)
        price_win = np.where(win_i, price_at[:, None], price_at[None, :])
        cost_win = np.where(win_i, cost_at[:, None], cost_at[None, :])

        # Winner participates iff best surplus >= 0; pays own reserve plus the
        # loser's non-negative surplus, net of the served grade's cost.
        served = best_win >= -1e-10
        pay = price_win + np.maximum(best_los, 0.0) - cost_win
        rev = np.where(served, pay, 0.0)

        return float((rev * (w[:, None] * w[None, :])).sum())

    def optimal_revenue(self, price_grid=None):
        """Max revenue over candidate reserve-price vectors.

        ``price_grid`` defaults to the type grid ``self.X_iter`` (the search the
        thesis code used). Returns ``(revenue, argmax_price)``.
        """
        if price_grid is None:
            price_grid = self.X_iter
        best_r, best_p = -np.inf, None
        for p in price_grid:
            r = self.revenue(p)
            if r >= best_r:
                best_r, best_p = r, list(p)
        return best_r, best_p
