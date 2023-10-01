from scipy.stats import uniform, beta

from auction.approximation import OptimalAuctionApproximation as \
    Approximation


def make_examples(T, n_buyers=2, **kwargs):

    # belloni (uniform + independent)
    ex1 = Approximation(
        n_buyers=n_buyers,
        V=[[6, 8], [9, 11]],
        costs=[0.9, 5],
        T=T,
        **kwargs)

    # U[0,1] x U[0,1] (independent)
    # (see Pavlov 2011 for analytic result)
    ex2 = Approximation(
        n_buyers=n_buyers,
        V=[[0, 1], [0, 1]],
        costs=[0, 0],
        T=T,
        **kwargs)

    # Beta(1,2) x Beta(1,2) (independent)
    ex3 = Approximation(
        n_buyers=n_buyers,
        V=[[0, 1], [0, 1]],
        costs=[0, 0],
        T=T,
        f=[beta(1, 2).pdf, beta(1, 2).pdf],
        check_local_ic='star',
        **kwargs)

    # mixture U[0,1] x Beta(1,2) (alpha=1/2)
    alpha = 0.5

    def corr(v, fs):
        return alpha * fs[0](v[0]) + (1 - alpha) * fs[1](v[1])

    ex4 = Approximation(
        n_buyers=n_buyers,
        V=[[0, 1], [0, 1]],
        costs=[0, 0],
        T=T,
        f=[uniform(0, 1).pdf, beta(1, 2).pdf],
        corr=corr,
        **kwargs)

    return ex1, ex2, ex3, ex4
