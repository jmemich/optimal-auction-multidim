import numpy as np

from time import time
from concurrent.futures import ProcessPoolExecutor

from approx import OptimalAuctionApproximation as Approximation

N_WORKERS = 2

if __name__ == '__main__':
    executor = ProcessPoolExecutor(max_workers=N_WORKERS)

    start = time()
    approx = Approximation(
        n_buyers=2,
        V=[[0,1],[0,1]],
        costs=[0,0],
        T=10,
        executor=executor,
        log_level='debug')
    approx.run()
    elapsed = time() - start
    print(np.round(elapsed,5))

    executor.shutdown()
