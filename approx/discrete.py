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


class f_hat:

    def __init__(self, pdfs, V_T, corr, T, J):
        self.pdfs = pdfs
        self.V_T = V_T
        self.corr = corr
        self.T = T
        self.J = J

        # 1) normalize the distribution so sum = 1 (for numerical issues)
        # 2) incorporate the adjustments for numerical integration here
        # NOTE user is ultimately repsonsible for normalizing distribution
        self.V_T_sum = 0
        for i, v in enumerate(V_T):
            self.V_T_sum += corr(v, self.pdfs) * self.numerical_adjustment(i)

        # precompute and then store value
        self.precompute = []
        for i, v in enumerate(V_T):
            f = corr(v, self.pdfs) * self.numerical_adjustment(i)
            self.precompute.append(f / self.V_T_sum)

    def numerical_adjustment(self, i):
        adjustment = 1
        if self.J == 1:
            if i in [0, self.T]:
                adjustment /= 2
        elif self.J == 2:
            xi = int(i / (self.T + 1))
            yi = i % (self.T + 1)
            if xi in [0, self.T]:
                adjustment /= 2
            if yi in [0, self.T]:
                adjustment /= 2
        else:  # J > 2
            raise ValueError("J cannot be > 2")
        return adjustment

    def __getitem__(self, i):
        return self.precompute[i]

    def __call__(self, v):
        # NOTE this is roundabout way of getting the value from the index
        # it kinda works. shouldn't really be used in prod though....
        # (this is also *much* slower than indexing)
        i = np.argmax((np.round(self.V_T, 4) == np.round(v, 4)).all(axis=1))
        f = self.corr(v, self.pdfs) * self.numerical_adjustment(i)
        return f / self.V_T_sum
