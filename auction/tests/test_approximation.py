import pickle

from auction import OptimalAuctionApproximation as Approximation


def test_pickle_roundtrip():
    test = Approximation(
        n_buyers=2,
        V=[[0, 1], [0, 1]],
        costs=[0, 0],
        T=2
    )
    test.run()

    test = pickle.loads(pickle.dumps(test))
    test.Q  # check this works as intended
    test.run()  # check we can run it again if we want
