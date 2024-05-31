import numpy as np

from time import time
from concurrent.futures import ProcessPoolExecutor

from approx import OptimalAuctionApproximation as Approximation

N_WORKERS = 10

if __name__ == '__main__':

    res = dict()

    for T in [20, 25, 30]:

        executor = ProcessPoolExecutor(max_workers=N_WORKERS)

        start = time()
        try:
            approx = Approximation(
                n_buyers=2,
                V=[[0, 1], [0, 1]],
                costs=[0, 0],
                T=T,
                executor=executor,
                log_level='debug')
            approx.run()
        except:
            x = 1
        elapsed = time() - start
        res[T] = (elapsed, approx)

        executor.shutdown()

    for k, v in res.items():
        print('T=%s, success=%s, time=%s' % (k, v[1].converged, np.round(v[0], 5)))
