import numpy as np

EPS = 1e-6


def discretize(V, T):
    # NOTE see Belloni p.1082
    eps = np.inf
    for v_j in V:
        candidate = (v_j[1] - v_j[0]) / T
        if candidate < eps:
            eps = candidate
    assert eps < np.inf, "cannot discretize when `eps` is `np.inf`"
    V_T = []
    for v_j in V:
        v_j_discretized = \
            np.arange(v_j[0], v_j[1] - EPS, eps).tolist() + [v_j[1]]
        V_T.append(v_j_discretized)
    return V_T, eps


def normalize(f, V_T_j):
    sum = 0
    for v in V_T_j:
        sum += f(v)

    def f_normalized(v):
        return f(v) / sum

    return f_normalized


# NOTE here we make a class for the PDF of the type distribution because we
# can precompute values and index them, which saves considerable time
class f_hat:

    def __init__(self, pdfs, V_T, corr):
        self.pdfs = []
        for j in range(len(V_T[0])):
            V_T_j = [V[j] for V in V_T]
            pdf = normalize(pdfs[j], V_T_j)
            self.pdfs.append(pdf)
        self.V_T = V_T
        self.corr = corr

        # just in case `corr` isn't normalized by user we make sure to (re)sum
        self.V_T_sum = 0
        for v in V_T:
            f = corr(v, self.pdfs)
            self.V_T_sum += f

        # precompute and then store value
        self.precompute = []
        for v in V_T:
            f = corr(v, self.pdfs)
            self.precompute.append(f / self.V_T_sum)

    def __getitem__(self, i):
        return self.precompute[i]

    def __call__(self, v):
        # this is *much* slower than indexing
        f = self.corr(v, self.pdfs)
        return f / self.V_T_sum
