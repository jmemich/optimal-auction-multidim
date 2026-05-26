import itertools
import logging
import pickle
import sys
import time

import numpy as np
from ortools.linear_solver import pywraplp
from scipy.stats import uniform

from optimal_auctions.constraints import (
    BORDER_PREFIX,
    IC_PREFIX,
    Constraint,
    make_border_expr_from_name,
    make_ic_expr_from_name,
)
from optimal_auctions.discrete import discretize, f_hat
from optimal_auctions.oracle import _check_border, _make_lower_left_quadrant, separation_oracle
from optimal_auctions.util import symmetric_ix

LOG_FMT = "(%(process)d) %(asctime)s [%(levelname)s] %(message)s"


def _standardize_log_level(log_level):
    if type(log_level) is str:
        if log_level.lower() == "info":
            log_level = logging.INFO
        elif log_level.lower() == "debug":
            log_level = logging.DEBUG
        elif log_level.lower() == "warning":
            log_level = logging.WARNING
        elif log_level.lower() == "critical":
            log_level = logging.CRITICAL
        else:
            raise ValueError(f"`log_level` '{log_level}' not supported!")
    return log_level


def _independent(v, fs):
    # evaluation bidder PDFs independently (f1 * f2 * ...)
    # this exists because we can't pickle a lambda function
    val = 1
    for i, f in enumerate(fs):
        val *= f(v[i])
    return val


class OptimalAuctionApproximation:
    """This class executes the approximation algorithm for optimal
    auctions developed in (Kushnir & Michelson). It is intended to be an
    extension of (Belloni et al. 2010) which allows for the exploration of a
    number of hypotheses concerning the qualitative properties of optimal
    mechanisms in multi-dimensional settings.

    Parameters
    ----------
    n_buyers : int
        The number of buyers (bidders) in the auction

    V : list[list[float]]
        Buyers' types. Expected in form `[[a,b], [c,d], ...]` where `[a,b]`
        corresponds to the buyers' valuations for the first quality level
        (good) and `[c,d]` to the buyers' valuations for the second quality
        level, etc.

    costs : list[float]
        Costs for each quality level (good), for example, `[0,0]`

    T : int
        Parameter controlling the discretization of the type space. For finer
        discretizations, increase `T`.

    f : list[function]
        The distriubtions used to represent the designer's uncertainty about
        buyers' types.

    corr : function[list[float], list[function]], default=None
        Arbitrary correlation structure across buyers' valuations. Default is
        assummed to be independence

    solver_type : str, default='GLOP'
        Argument passed to `ortools.linear_solver`

    force_symmetric : bool, default=True
        Allow the code to assume valuation symmetry such that Q(i,j) == Q(j,i)

    check_local_ic : bool, default=True
        Toggles consideration of only local incentive-compatibility constraints
        on each iteration.

    executor : concurrent.futures.ProcessPoolExecutor, default=None
        For multiprocessing.

    seed : int, default=12345
        Seed which is used to randomly sample subsets of constraints at each
        iteration. Note, this argument is NOT passed to `ortools.linear_solver`

    log_level : str, default=logging.INFO
        Controls logging level.

    Attributes
    ----------
    Q : array
        The resulting optimal allocation from the approximation

    U : array
        The result optimal utility variables from the approximation

    opt : float
        Value of the objective function

    References
    ----------
    Kushnir & Michelson: https://arxiv.org/abs/2207.01664
    Belloni et al. (2010): https://people.duke.edu/~abn5/MMD.pdf

    Examples
    --------
    >>> from auction import OptimalAuctionApproximation as Approximation
    >>> approx = Approximation(n_buyers=2, V=[[0,1],[0,1]], costs=[0,0], T=10)
    >>> approx.run()
    """

    def __init__(
        self,
        n_buyers,
        V,
        costs,
        T,
        f=None,
        corr=None,
        solver_type="GLOP",
        force_symmetric=True,
        check_local_ic=True,
        executor=None,
        seed=12345,
        log_level=logging.INFO,
    ):
        self.n_buyers = n_buyers
        self.n_grades = len(V)
        self.grades = list(range(self.n_grades))
        self.costs = costs

        if len(V) != 2:
            raise NotImplementedError("Code doesn't support len(V) > 2!")
        self.V = V

        self.T = T
        self.V_T_list, self.eps = discretize(V, T)
        self.V_T = list(itertools.product(*self.V_T_list))  # x-product vt_i

        if f is None:  # assume uniform
            f = [uniform(*V[0]).pdf, uniform(*V[1]).pdf]
        self.f = f
        if corr is None:  # assume independent
            corr = _independent
        self.corr = corr
        self.f_hat = f_hat(self.f, self.V_T, self.corr, self.T, self.n_grades)

        self.check_local_ic = check_local_ic  # TODO remove
        self.executor = executor
        self.force_symmetric = force_symmetric
        if force_symmetric and self.n_grades > 2:
            raise NotImplementedError("`force_symmetric`=True and `grades`>2 not supported!")

        self.solver_type = solver_type
        self.rng = np.random.RandomState(seed)

        self.log_level = _standardize_log_level(log_level)
        logging.basicConfig(
            format=LOG_FMT, datefmt="%H:%M:%S", level=self.log_level, stream=sys.stdout
        )
        problem_str = (
            f"PROBLEM SETUP: n_buyers={n_buyers}, n_grades={self.n_grades}, V={V!s}, costs={costs!s}, T={T}, "
            f"solver={solver_type}, force_symmetric={force_symmetric}, ic_local={check_local_ic}"
        )
        logging.info(problem_str)

    def _create_or_warmstart_Q_U_vars(self, solver, Q_values=None, U_values=None):
        Q_vars = []
        grades = self.grades if not self.force_symmetric else [0]
        for j in grades:
            Q_j = []
            for i, _ in enumerate(self.V_T):
                var_name = f"Q_{j}_{i}"
                var = solver.NumVar(0, 1, var_name)
                Q_j.append(var)
            if Q_values is not None:
                solver.SetHint(Q_j, Q_values[j])
            Q_vars.append(Q_j)

        max_utility = max([max(j) for j in self.V_T])

        U_vars = []
        for i, _ in enumerate(self.V_T):
            var_name = f"U_{i}"
            var = solver.NumVar(0, max_utility, var_name)
            U_vars.append(var)
        if U_values is not None:
            solver.SetHint(U_vars, U_values)

        return Q_vars, U_vars

    def _make_obj(self, Q_vars, U_vars):
        obj = 0
        if self.force_symmetric:
            for v_ix, v in enumerate(self.V_T):
                inner = 0
                inner += (v[0] - self.costs[0]) * Q_vars[0][v_ix]
                symm_v_ix = symmetric_ix(v_ix, self.T)
                inner += (v[1] - self.costs[1]) * Q_vars[0][symm_v_ix]
                inner -= U_vars[v_ix]
                inner *= self.f_hat[v_ix]
                obj += inner
            obj *= self.n_buyers
        else:
            for v_ix, v in enumerate(self.V_T):
                inner = 0
                for j_ix in self.grades:
                    inner += (v[j_ix] - self.costs[j_ix]) * Q_vars[j_ix][v_ix]
                inner -= U_vars[v_ix]
                inner *= self.f_hat[v_ix]
                obj += inner
            obj *= self.n_buyers
        return obj

    def _create_base_constraints(self, Q_vars, U_vars):
        # NOTE here we define feasibility and IR constraints (IC come later)
        # NOTE 0 <= Q <= 1 (forall Q) is defined when we create variables!
        # NOTE 0 <= U <= U_max (forall U) is defined when we create variables!

        # 1. IR constraint
        ir_con = Constraint("ir", U_vars[0] <= 10**-10, None)

        # 2. Q is valid probability distribution
        prob_cons = []
        for i in range(len(self.V_T)):
            if not self.force_symmetric:
                Q_total_i = 0
                for j in self.grades:
                    Q_total_i += Q_vars[j][i]
                prob_con = Constraint(f"prob_{i}_{j}", Q_total_i <= 1, None)
                prob_cons.append(prob_con)
            else:
                j = symmetric_ix(i, self.T)
                Q_total_i = Q_vars[0][i] + Q_vars[0][j]
                prob_con = Constraint(f"prob_{i}_{j}", Q_total_i <= 1, None)
                prob_cons.append(prob_con)

        base_cons = [ir_con, *prob_cons]
        return base_cons

    def _setup_solver(self, con_names, Q_values=None, U_values=None):
        # 1. create solver, 2. create vars, 3. add base constraints
        solver = pywraplp.Solver.CreateSolver(self.solver_type)

        Q_vars, U_vars = self._create_or_warmstart_Q_U_vars(solver, Q_values, U_values)
        base_cons = self._create_base_constraints(Q_vars, U_vars)

        for con in base_cons:
            solver.Add(con.expr)

        for name in con_names:
            if name.startswith(IC_PREFIX):
                expr = make_ic_expr_from_name(
                    name, Q_vars, U_vars, self.T, self.V_T, self.grades, self.force_symmetric
                )
            elif name.startswith(BORDER_PREFIX):
                expr = make_border_expr_from_name(
                    name,
                    Q_vars,
                    self.T,
                    self.V_T,
                    self.n_buyers,
                    self.grades,
                    self.f_hat,
                    self.force_symmetric,
                )
            else:
                raise RuntimeError("constraint prefix not recognised!")
            solver.Add(expr, name)

        return solver, Q_vars, U_vars

    def run(self, warm_start=True, solver_max_time=300):
        """Run the approximation algorithm.

        Parameters
        ----------
        warm_start : bool, default=True
            Warm start the `ortools` solver with previous values at each
            iteration. Note, user needs to check `ortools` documentation
            for whether solver supports warmstarts.

        solver_max_time : int, default=300
            Solver time limit in seconds.
        """
        begin = time.time()
        self.opt = np.inf
        self.i = 1
        self.js = []

        # here we define a local region of the typespace which we check for
        # constraint violations. If we subsequently find violated constraints,
        # we incrementally grow the region of the typespace checked
        self.net_size = 0  # start only considering each point
        self.net_size_inc = max(1, int(self.T / 10))  # scales for big problems

        self.con_names = set()
        self.converged = False
        while not self.converged:
            # TODO cleanup this dupe code
            # define local region of typespace for IC constrait checks
            for i, _v_i in enumerate(self.V_T):
                # NOTE also this creates dupes! (we de-dupe below)
                # TODO this is wrong but it overcounts by only a small amount
                star_ix = [
                    i + self.T,  # above-left
                    i + self.T + 1,  # above
                    i + self.T + 2,  # above-right
                    i - 1,  # left
                    i + 1,  # right
                    i - self.T - 2,  # below-left
                    i - self.T - 1,  # below
                    i - self.T,
                ]  # below-right
                # deal with corners/edges:
                star_ix = [ix for ix in star_ix if ix >= 0 and ix < len(self.V_T)]
                quadrant_ix = _make_lower_left_quadrant(i, self.T, self.net_size)
                all_ix = set(star_ix + quadrant_ix)
                all_v_j = [self.V_T[ix] for ix in all_ix]
                inner_loop = zip(star_ix, all_v_j, strict=False)
                for j, _v_j in inner_loop:
                    name = f"{IC_PREFIX}_{i}_{j}"
                    self.con_names.add(name)

            # make solver
            self.solver, self.Q_vars, self.U_vars = self._setup_solver(self.con_names)
            # self.solver.SetSolverSpecificParametersAsString(
            #     "max_time_in_seconds:%s" % solver_max_time)
            self.solver.SetSolverSpecificParametersAsString("use_dual_simplex:1")
            obj = self._make_obj(self.Q_vars, self.U_vars)
            self.solver.Maximize(obj)

            j = 1
            n_violated = 0
            self.border_failing = True
            n_consecutive_fails = 0
            while self.border_failing:
                # 4 = "abnormal" (from numerical fail due to re-running)
                # https://github.com/google/or-tools/blob/stable/ortools/linear_solver/linear_solver.h#L464-L479
                self.solver_status = self.solver.Solve()

                if self.solver_status != 0:
                    if self.solver_status == 4 and n_consecutive_fails < 5:
                        n_consecutive_fails += 1
                        logging.warning(
                            f"solver status = {self.solver_status}! remaking solver... (#{n_consecutive_fails}/5)"
                        )
                        self.solver, self.Q_vars, self.U_vars = self._setup_solver(self.con_names)
                        self.solver.SetSolverSpecificParametersAsString("use_dual_simplex:1")
                        # self.solver.SetSolverSpecificParametersAsString(
                        #     "max_time_in_seconds:%s" % solver_max_time)
                        obj = self._make_obj(self.Q_vars, self.U_vars)
                        self.solver.Maximize(obj)
                        continue
                    self.solver, self.Q_vars, self.U_vars = self._setup_solver(self.con_names)
                    self.solver.SetSolverSpecificParametersAsString("use_dual_simplex:1")
                    obj = self._make_obj(self.Q_vars, self.U_vars)
                    self.solver.Maximize(obj)
                    self.solver.EnableOutput()
                    self.solver_status = self.solver.Solve()
                    raise RuntimeError(
                        f"solver failed with status={self.solver_status} (#{n_consecutive_fails}/5)"
                    )
                n_consecutive_fails = 0

                solver_i = self.solver.iterations()
                self.opt = self.solver.Objective().Value()

                n_ic, n_border = 0, 0
                for con in self.solver.constraints():
                    if con.name().startswith(IC_PREFIX):
                        n_ic += 1
                    if con.name().startswith(BORDER_PREFIX):
                        n_border += 1

                msg = f"(i={self.i}, j={j}, solver={solver_i}) obj={self.opt}, #IC={n_ic}, #border={n_border} ({n_violated})"
                logging.info(msg)

                # convert solver vars to raw values
                self._U_values = [U.solution_value() for U in self.U_vars]
                if not self.force_symmetric:
                    self._Q_values = []
                    for j in self.grades:
                        self._Q_values.append([Q.solution_value() for Q in self.Q_vars[j]])
                else:
                    # TODO keep this the same as `self.Q`...
                    _Q1 = [Q.solution_value() for Q in self.Q_vars[0]]
                    _Q2 = np.zeros(len(_Q1))
                    for i, _ in enumerate(_Q1):
                        symm_i = symmetric_ix(i, self.T)
                        _Q2[symm_i] = _Q1[i]
                    self._Q_values = [_Q1, list(_Q2)]

                # check border + add violations
                border_cons = _check_border(
                    self.V_T, self.T, self._Q_values, self.grades, self.n_buyers, self.f_hat
                )

                n_violated = 0
                to_add = set()
                for con in border_cons:
                    if con.status == "VIOLATED":
                        to_add.add(con)
                        n_violated += 1
                    if con.status == "BINDING":
                        to_add.add(con)

                if n_violated == 0:
                    self.border_failing = False
                    logging.info(f"successfully run inner loop (net_size={self.net_size})")
                    break

                j += 1

                for con in to_add.difference(self.con_names):
                    self.con_names.add(con.name)
                    expr = make_border_expr_from_name(
                        con.name,
                        self.Q_vars,
                        self.T,
                        self.V_T,
                        self.n_buyers,
                        self.grades,
                        self.f_hat,
                        self.force_symmetric,
                    )
                    self.solver.Add(expr, con.name)

            self.js.append(j)
            check_local = False
            logging.info("checking full solution...")
            _, A_ic, _, _, A_border, _ = separation_oracle(
                self._Q_values,
                self._U_values,
                self.V_T,
                self.T,
                self.grades,
                self.n_buyers,
                self.f_hat,
                self.force_symmetric,
                check_local,
                self.executor,
                self.net_size,
            )
            n_A_ic, n_A_border = len(A_ic), len(A_border)

            if n_A_ic + n_A_border > 0:
                logging.warning(
                    "FAILURE: final solution to optimisation problem "
                    f"fails full separation oracle! # IC violated={n_A_ic}, # "
                    f"Border violated={n_A_border}."
                )
                for con in A_ic:
                    ixs_str = con.name.lstrip(IC_PREFIX + "_").split("_")
                    ixs = [int(ix) for ix in ixs_str]
                    logging.debug(
                        f"FAILURE [IC]: v={np.round(self.V_T[ixs[0]], 4)} (ix={ixs[0]}), v'={np.round(self.V_T[ixs[1]], 4)} (ix={ixs[1]})"
                    )
                    self.con_names.add(con.name)
                for con in A_border:
                    self.con_names.add(con.name)

                self.net_size += self.net_size_inc
                if self.net_size > self.T:
                    break
                logging.info(f"Running again with previous failures and `net_size`={self.net_size}")
                self.i += 1
                j = 1
            else:
                self.converged = True

        if not self.converged:
            logging.warning("algorithm did not converge!")
        self.elapsed = time.time() - begin
        minutes = np.round(self.elapsed / 60, 1)
        logging.info(f"finished! total time: {minutes} (mins), total iterations: {np.sum(self.js)}")

    def __getstate__(self):
        init_params = {
            "n_buyers": self.n_buyers,
            "costs": self.costs,
            "V": self.V,
            "T": self.T,
            "f": self.f,
            "corr": self.corr,
            "force_symmetric": self.force_symmetric,
            "check_local_ic": self.check_local_ic,
            "executor": self.executor,
            "solver_type": self.solver_type,
            "log_level": self.log_level,
        }
        if hasattr(self, "i"):  # this indiciates it has been run!
            run_params = {
                "i": self.i,
                "js": self.js,
                "elapsed": self.elapsed,
                "opt": self.opt,
                "converged": self.converged,
                "_Q_values": self._Q_values,
                "_U_values": self._U_values,
            }
        else:
            run_params = {}
        return {"init_params": init_params, "run_params": run_params}

    def __setstate__(self, state):
        init_params = state["init_params"]
        approx = OptimalAuctionApproximation(**init_params)
        run_params = state["run_params"]
        for key in run_params:
            setattr(approx, key, run_params[key])
        self.__dict__ = approx.__dict__

    def to_file(self, path):
        with open(path, "wb") as fout:
            pickle.dump(self, fout)

    @staticmethod
    def from_file(path):
        with open(path, "rb") as fin:
            return pickle.load(fin)

    @property
    def Q(self):
        # make sure we use Q_vars from most recent iteration!
        if not hasattr(self, "_Q_i"):
            self._Q_i = self.i
        if not hasattr(self, "_Q") or self._Q_i != self.i:
            _Q = []
            for i in range(len(self.V_T)):
                _Q_v = []
                for j in self.grades:
                    _Q_v.append(self._Q_values[j][i])
                _Q.append(_Q_v)
            self._Q = _Q
        return self._Q

    @property
    def U(self):
        # make sure we use U_vars from most recent iteration!
        if not hasattr(self, "_U_i"):
            self._U_i = self.i
        if not hasattr(self, "_U") or self._U_i != self.i:
            _U = []
            for i in range(len(self.V_T)):
                _U.append(self._U_values[i])
            self._U = _U
        return self._U
