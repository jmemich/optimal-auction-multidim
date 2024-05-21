import sys
import time
import pickle
import logging
import itertools

from scipy.stats import uniform
import numpy as np

from ortools.linear_solver import pywraplp

from approx.constraints import (Constraint, IC_PREFIX, BORDER_PREFIX,
                                make_border_expr_from_name,
                                make_ic_expr_from_name)
from approx.oracle import (separation_oracle, _check_border,
                           _make_lower_left_quadrant)
from approx.discrete import discretize, f_hat
from approx.util import symmetric_ix


LOG_FMT = '(%(process)d) %(asctime)s [%(levelname)s] %(message)s'


def _standardize_log_level(log_level):
    if type(log_level) is str:
        if log_level.lower() == 'info':
            log_level = logging.INFO
        elif log_level.lower() == 'debug':
            log_level = logging.DEBUG
        elif log_level.lower() == 'warning':
            log_level = logging.WARNING
        else:
            raise ValueError("`log_level` '%s' not supported!" % log_level)
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

    I_subset_prop : float, default=0.3
        The proportion of inactive constraints checked at each iteration

    A_subset_prop : float, default=0.3
        The proportion of violated constraints checked at each iteration

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
            solver_type='GLOP',
            force_symmetric=True,
            check_local_ic=True,
            executor=None,
            I_subset_prop=0.3,
            A_subset_prop=0.3,
            seed=12345,
            log_level=logging.INFO):
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

        self.check_local_ic = check_local_ic
        self.executor = executor
        self.force_symmetric = force_symmetric
        if force_symmetric and self.n_grades > 2:
            raise NotImplementedError(
                "`force_symmetric`=True and `grades`>2 not supported!")

        self.I_subset_prop = I_subset_prop
        self.A_subset_prop = A_subset_prop

        self.solver_type = solver_type
        self.rng = np.random.RandomState(seed)

        self.log_level = _standardize_log_level(log_level)
        logging.basicConfig(format=LOG_FMT, datefmt='%H:%M:%S',
                            level=self.log_level, stream=sys.stdout)
        problem_str = \
            ('PROBLEM SETUP: n_buyers=%s, n_grades=%s, V=%s, costs=%s, T=%s, '
             'solver=%s, force_symmetric=%s, ic_local=%s') % \
            (n_buyers, self.n_grades, str(V), str(costs), T, solver_type,
             force_symmetric, check_local_ic)
        logging.info(problem_str)

    def _create_or_warmstart_Q_U_vars(
            self, solver, Q_values=None, U_values=None):
        Q_vars = []
        grades = self.grades if not self.force_symmetric else [0]
        for j in grades:
            Q_j = []
            for i, _ in enumerate(self.V_T):
                var_name = 'Q_%s_%s' % (j, i)
                var = solver.NumVar(0, 1, var_name)
                Q_j.append(var)
            if Q_values is not None:
                solver.SetHint(Q_j, Q_values[j])
            Q_vars.append(Q_j)

        max_utility = max([max(j) for j in self.V_T])

        U_vars = []
        for i, _ in enumerate(self.V_T):
            var_name = 'U_%s' % i
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
        ir_con = Constraint('ir', U_vars[0] <= 10 ** -10, None)

        # 2. Q is valid probability distribution
        prob_cons = []
        for i in range(len(self.V_T)):
            if not self.force_symmetric:
                Q_total_i = 0
                for j in self.grades:
                    Q_total_i += Q_vars[j][i]
                prob_con = Constraint(
                    'prob_%s_%s' % (i, j), Q_total_i <= 1, None)
                prob_cons.append(prob_con)
            else:
                j = symmetric_ix(i, self.T)
                Q_total_i = \
                    Q_vars[0][i] + Q_vars[0][j]
                prob_con = Constraint(
                    'prob_%s_%s' % (i, j), Q_total_i <= 1, None)
                prob_cons.append(prob_con)

        base_cons = [ir_con] + prob_cons
        return base_cons

    def _setup_solver(self, S, A, Q_values=None, U_values=None):
        # 1. create solver, 2. create vars, 3. add base constraints
        solver = pywraplp.Solver.CreateSolver(self.solver_type)

        Q_vars, U_vars = self._create_or_warmstart_Q_U_vars(
            solver, Q_values, U_values)
        base_cons = self._create_base_constraints(Q_vars, U_vars)

        for con in base_cons:
            solver.Add(con.expr)

        for con in S + A:
            if con.name.startswith(IC_PREFIX):
                expr = make_ic_expr_from_name(
                    con.name, Q_vars, U_vars, self.T, self.V_T, self.grades,
                    self.force_symmetric)
            elif con.name.startswith(BORDER_PREFIX):
                expr = make_border_expr_from_name(
                    con.name, Q_vars, self.T, self.V_T, self.n_buyers,
                    self.grades, self.f_hat, self.force_symmetric)
            else:
                raise RuntimeError("constraint prefix not recognised!")
            solver.Add(expr)

        return solver, Q_vars, U_vars

    def run(self,
            warm_start=True,
            skip_oracle_check=False):
        """Run the approximation algorithm.

        Parameters
        ----------
        warm_start : bool, default=True
            Warm start the `ortools` solver with previous values at each
            iteration. Note, user needs to check `ortools` documentation
            for whether solver supports warmstarts.

        max_iter : int, default=None
            Maximum number of iterations before approximation algorithm
            terminates. Note, algorithm may not converge if this argument is
            used.

        skip_oracle_check : bool, default=False
            Toggle skip of final check of all IC constraints. This is needed to
            guarantee convergence.
        """
        begin = time.time()
        self.opt = np.inf
        self.i = 1
        A = []

        # here we define a local region of the typespace which we check for
        # constraint violations. If we subsequently find violated constraints,
        # we incrementally grow the region of the typespace checked
        self.net_size = 0  # start only considering each point
        self.net_size_inc = max(1, int(self.T / 10))  # scales for big problems

        self.converged = False
        while not self.converged:
            self._run(A, warm_start, self.net_size)

            if not self.check_local_ic:  # TODO x2 check this with new logic
                self.converged = True
                break

            check_local = False
            logging.debug('checking final solution...')
            _, A_ic, _, _, A_border, _ = \
                separation_oracle(
                    self._Q_values, self._U_values, self.V_T, self.T,
                    self.grades, self.n_buyers, self.f_hat,
                    self.force_symmetric, check_local, self.executor,
                    self.net_size)
            self.A_ic = A_ic
            self.A_border = A_border
            n_A_ic, n_A_border = len(A_ic), len(A_border)
            if n_A_ic + n_A_border > 0:
                logging.warning(
                    "FAILURE: final solution to optimisation problem "
                    "fails full separation oracle! # IC violated=%s, # "
                    "Border violated=%s." % (n_A_ic, n_A_border))
                for con in A_ic:
                    ixs_str = con.name.lstrip(IC_PREFIX + '_').split('_')
                    ixs = [int(ix) for ix in ixs_str]
                    logging.info(
                        "FAILURE [IC]: v=%s (ix=%s), v'=%s (ix=%s)" %
                        (np.round(self.V_T[ixs[0]], 4), ixs[0],
                         np.round(self.V_T[ixs[1]], 4), ixs[1]))

                self.net_size += self.net_size_inc
                if self.net_size > self.T:
                    logging.warning("Failed to converge!")
                    break
                logging.info(
                    "Running again with previous failures and "
                    "`net_size`=%s" % self.net_size)
                _A = A_ic + A_border
                A = list(set(_A + A))  # always keep annoying failures...
            else:
                self.converged = True

        if not self.converged:
            logging.warning("algorithm did not converge!")
        self.elapsed = time.time() - begin
        minutes = np.round(self.elapsed / 60, 1)
        logging.info('finished! total time: %s (mins)' % minutes)

    @property
    def border_con_names(self):
        # NOTE we store names so as not to keep cons from old solvers around
        if not hasattr(self, '_border_con_names'):
            self._border_con_names = set()
        return self._border_con_names

    def _run(self, A_old, warm_start, net_size):
        # 0. setup solver and variables; add old violated cons
        S = []  # TODO remove this
        A = A_old.copy()
        self.solver, self.Q_vars, self.U_vars = self._setup_solver(S, A)

        # 1. make + add local ic cons to solver
        ic_con_names = set()
        for i, v_i in enumerate(self.V_T):
            # NOTE also this creates dupes! (we de-dupe below)
            # TODO this is wrong but it overcounts by only a small amount
            star_ix = [i + self.T,      # above-left
                       i + self.T + 1,  # above
                       i + self.T + 2,  # above-right
                       i - 1,           # left
                       i,
                       i + 1,           # right
                       i - self.T - 2,  # below-left
                       i - self.T - 1,  # below
                       i - self.T]      # below-right
            # deal with corners/edges:
            star_ix = [ix for ix in star_ix
                       if ix >= 0 and ix < len(self.V_T)]
            quadrant_ix = _make_lower_left_quadrant(i, self.T, net_size)
            all_ix = set(star_ix + quadrant_ix)
            all_v_j = [self.V_T[ix] for ix in all_ix]
            inner_loop = zip(star_ix, all_v_j)
            for j, v_j in inner_loop:
                name = '%s_%s_%s' % (IC_PREFIX, i, j)
                ic_con_names.add(name)

        self._ic_cons = set()
        for name in ic_con_names.difference(A):
            expr = make_ic_expr_from_name(
                name, self.Q_vars, self.U_vars, self.T, self.V_T,
                self.grades, self.force_symmetric)
            self.solver.Add(expr, name)
            con = Constraint(name, expr, None)
            self._ic_cons.add(con)

        # 2. add old border con violations
        for name in self.border_con_names:
            expr = make_border_expr_from_name(
                name, self.Q_vars, self.T, self.V_T, self.grades,
                self.f_hat, self.force_symmetric)
            self.solver.Add(expr, name)

        # 3. define obj
        obj = self._make_obj(self.Q_vars, self.U_vars)
        self.solver.Maximize(obj)

        while True:
            # 4. run solver
            status = self.solver.Solve()
            assert status == pywraplp.Solver.OPTIMAL, \
                "optimal solution not found! status=%s" % status

            solver_i = self.solver.iterations()
            self.opt = self.solver.Objective().Value()

            msg = '(i=%s, solver i=%s) obj=%s, # IC=%s, # border=%s' \
                % (self.i, solver_i, self.opt, len(self._ic_cons),
                   len(self.border_con_names))
            logging.info(msg)

            # TODO max_iter

            # 5. convert solver vars to raw values
            self._U_values = [U.solution_value() for U in self.U_vars]
            if not self.force_symmetric:
                self._Q_values = []
                for j in self.grades:
                    self._Q_values.append(
                        [Q.solution_value() for Q in self.Q_vars[j]])
            else:
                # TODO keep this the same as `self.Q`...
                _Q1 = [Q.solution_value() for Q in self.Q_vars[0]]
                _Q2 = np.zeros(len(_Q1))
                for i, _ in enumerate(_Q1):
                    symm_i = symmetric_ix(i, self.T)
                    _Q2[symm_i] = _Q1[i]
                self._Q_values = [_Q1, list(_Q2)]

            # 6. check border + add violations
            local_border_cons = _check_border(
                self.V_T, self.T, self._Q_values, self.grades, self.n_buyers,
                self.f_hat)

            to_add = set()
            n_violated_border = 0
            for con in local_border_cons:
                if con.status == 'VIOLATED':
                    to_add.add(con)
                    self._border_con_names.add(con.name)
                    n_violated_border += 1

            if n_violated_border == 0:
                logging.info("success!")
                break  # success!

            self.i += 1

            for con in to_add:
                expr = make_border_expr_from_name(
                    con.name, self.Q_vars, self.T, self.V_T, self.n_buyers,
                    self.grades, self.f_hat, self.force_symmetric)
                self.solver.Add(expr, con.name)

            # self.solver.Clear()  # TODO this segfaults

    def __getstate__(self):
        init_params = {
            'n_buyers': self.n_buyers,
            'costs': self.costs,
            'V': self.V,
            'T': self.T,
            'f': self.f,
            'corr': self.corr,
            'force_symmetric': self.force_symmetric,
            'check_local_ic': self.check_local_ic,
            'executor': self.executor,
            'I_subset_prop': self.I_subset_prop,
            'A_subset_prop': self.A_subset_prop,
            'solver_type': self.solver_type,
            'log_level': self.log_level
        }
        if hasattr(self, "i"):  # this indiciates it has been run!
            run_params = {
                'i': self.i,
                'elapsed': self.elapsed,
                'opt': self.opt,
                'converged': self.converged,
                '_Q_values': self._Q_values,
                '_U_values': self._U_values,
                'S': self.S,
                'A': self.A,
                'I': self.I
            }
        else:
            run_params = {}
        return {
            'init_params': init_params,
            'run_params': run_params
        }

    def __setstate__(self, state):
        init_params = state['init_params']
        approx = OptimalAuctionApproximation(**init_params)
        run_params = state['run_params']
        for key in run_params:
            setattr(approx, key, run_params[key])
        self.__dict__ = approx.__dict__

    def to_file(self, path):
        with open(path, 'wb') as fout:
            pickle.dump(self, fout)

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as fin:
            return pickle.load(fin)

    @property
    def Q(self):
        # make sure we use Q_vars from most recent iteration!
        if not hasattr(self, '_Q_i'):
            self._Q_i = self.i
        if not hasattr(self, '_Q') or self._Q_i != self.i:
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
        if not hasattr(self, '_U_i'):
            self._U_i = self.i
        if not hasattr(self, '_U') or self._U_i != self.i:
            _U = []
            for i in range(len(self.V_T)):
                _U.append(self._U_values[i])
            self._U = _U
        return self._U
