from optimal_auctions import OptimalAuctionApproximation as Approx
from optimal_auctions.constraints import Constraint   # NOQA

from ortools.linear_solver import pywraplp   # NOQA


approx = Approx(
    n_buyers=2,
    V=[[0, 1], [0, 1]],
    costs=[0, 0],
    T=30,
    force_symmetric=True,
    solver_type='GLOP',
    log_level='debug')
approx.run()

try:
    solver = approx.solver
except RuntimeError:
    x = 1

constraints = [c.name() for c in solver.constraints()
               if c.name().startswith('ic') or c.name().startswith('border')]
solver2, Q2, U2 = approx._setup_solver(constraints)
obj2 = approx._make_obj(Q2, U2)
solver2.Maximize(obj2)
status = solver2.Solve()
