from concurrent.futures import ProcessPoolExecutor
from time import time

import numpy as np
from approx import OptimalAuctionApproximation as Approximation

N_WORKERS = 2

if __name__ == "__main__":
    res = dict()

    for T in [20]:
        executor = ProcessPoolExecutor(max_workers=N_WORKERS)

        start = time()
        try:
            print(f"running T={T}...")
            approx = Approximation(
                n_buyers=2,
                V=[[0, 1], [0, 1]],
                costs=[0, 0],
                T=T,
                executor=executor,
                log_level="info",
            )
            approx.run()
        except RuntimeError as err:
            print(err)
        elapsed = time() - start
        res[T] = (elapsed, approx)

        executor.shutdown()

    for k, v in res.items():
        print(f"T={k}, success={v[1].converged}, time={np.round(v[0], 5)}")

    solver = approx.solver
    constraints = [
        c.name()
        for c in solver.constraints()
        if c.name().startswith("ic") or c.name().startswith("border")
    ]
    solver2, Q2, U2 = approx._setup_solver(constraints)
    obj2 = approx._make_obj(Q2, U2)
    solver2.Maximize(obj2)
    status = solver2.Solve()
    print(status)
