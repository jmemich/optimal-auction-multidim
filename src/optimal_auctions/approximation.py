import itertools
import logging
import pickle
import sys
import time

import highspy
import numpy as np
from scipy.stats import uniform

from optimal_auctions.constraints import (
    BORDER_PREFIX,
    IC_PREFIX,
    star_indices,
)
from optimal_auctions.discrete import discretize, f_hat
from optimal_auctions.oracle import _check_border, separation_oracle
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
    # evaluate bidder PDFs independently (f1 * f2 * ...); the default `corr`
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

    T : int, default=40
        Parameter controlling the discretization of the type space. For finer
        discretizations, increase `T`. Default 40 is the largest grid that
        solves in ~minutes for the symmetric U[0,1]^2 / N=2 baseline (~70s);
        revenue is within ~0.01 of the T->inf limit and Richardson-extrapolable.

    f : list[function]
        The distriubtions used to represent the designer's uncertainty about
        buyers' types.

    corr : function[list[float], list[function]], default=None
        Arbitrary correlation structure across buyers' valuations. Default is
        assummed to be independence

    force_symmetric : bool, default=True
        Allow the code to assume valuation symmetry such that Q(i,j) == Q(j,i)

    check_local_ic : bool, default=True
        Toggles consideration of only local incentive-compatibility constraints
        on each iteration.

    seed : int, default=12345
        Seed which is used to randomly sample subsets of constraints at each
        iteration. Note, this argument is NOT passed to the LP solver.

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
    >>> from optimal_auctions import OptimalAuctionApproximation as Approximation
    >>> approx = Approximation(n_buyers=2, V=[[0,1],[0,1]], costs=[0,0], T=10)
    >>> approx.run()
    """

    def __init__(
        self,
        n_buyers,
        V,
        costs,
        T=40,
        f=None,
        corr=None,
        force_symmetric=True,
        check_local_ic=True,
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
        self.force_symmetric = force_symmetric
        if force_symmetric and self.n_grades > 2:
            raise NotImplementedError("`force_symmetric`=True and `grades`>2 not supported!")

        self.rng = np.random.RandomState(seed)

        self.log_level = _standardize_log_level(log_level)
        logging.basicConfig(
            format=LOG_FMT, datefmt="%H:%M:%S", level=self.log_level, stream=sys.stdout
        )
        problem_str = (
            f"PROBLEM SETUP: n_buyers={n_buyers}, n_grades={self.n_grades}, V={V!s}, costs={costs!s}, T={T}, "
            f"solver=HiGHS, force_symmetric={force_symmetric}, ic_local={check_local_ic}"
        )
        logging.info(problem_str)

    # ------------------------------------------------------------------
    # column-index bookkeeping
    # ------------------------------------------------------------------
    def _q_col(self, j, v_ix):
        # column index of the Q_j(v_ix) variable.
        # symmetric -> a single grade-0 block of length len(V_T);
        # asymmetric -> n_grades contiguous blocks.
        if self.force_symmetric:
            return self._q_col_start + v_ix
        return self._q_col_start + j * len(self.V_T) + v_ix

    def _u_col(self, v_ix):
        return self._u_col_start + v_ix

    def _create_columns(self):
        """Add all Q and U columns to the HiGHS instance and record the column
        start indices. Q columns are bounded [0, 1]; U columns [0, U_max] (with
        column 0 -- the worst type's utility -- pinned to <= 1e-10 for IIR)."""
        n_V_T = len(self.V_T)
        n_grade_blocks = 1 if self.force_symmetric else self.n_grades

        # Q columns
        self._q_col_start = self.solver.getNumCol()
        n_q = n_grade_blocks * n_V_T
        self.solver.addVars(n_q, [0.0] * n_q, [1.0] * n_q)

        # U columns
        max_utility = max([max(j) for j in self.V_T])
        self._u_col_start = self.solver.getNumCol()
        u_lb = [0.0] * n_V_T
        u_ub = [max_utility] * n_V_T
        self.solver.addVars(n_V_T, u_lb, u_ub)

        # IIR boundary: pin worst type's utility ~0 (U[0] <= 1e-10, with
        # U[0] >= 0 from the bound). OR-Tools expressed this exact value.
        self.solver.changeColBounds(self._u_col(0), 0.0, 10**-10)

    def _set_objective(self):
        """Set the per-column objective coefficients for OPT*:

            OPT* = N * sum_v f_hat[v] * [ (v0-c0)Q0(v) + (v1-c1)Q1(v) - U(v) ].

        HiGHS stores one coefficient per column, so we accumulate each column's
        contribution into `obj_coeffs` and hand the whole vector to the solver.
        Each line below writes a variable's *coefficient* -- the solver supplies
        the variable itself when it evaluates the objective. (`self.costs` is the
        per-grade production cost vector from __init__; `obj_coeffs` is unrelated
        -- it is the LP objective row.)
        """
        n_cols = self.solver.getNumCol()
        obj_coeffs = np.zeros(n_cols)

        if self.force_symmetric:
            for v_ix, v in enumerate(self.V_T):
                f = self.f_hat[v_ix]
                # coeff on Q0(v_ix): f * (v0 - cost0)
                obj_coeffs[self._q_col(0, v_ix)] += f * (v[0] - self.costs[0])
                # coeff on Q1(v_ix), which under symmetry is carried by Q0 at the
                # swapped index symmetric_ix(v_ix): f * (v1 - cost1)
                symm_v_ix = symmetric_ix(v_ix, self.T)
                obj_coeffs[self._q_col(0, symm_v_ix)] += f * (v[1] - self.costs[1])
                # coeff on U(v_ix): -f. Utility is per-type (indexed directly), so
                # unlike Q it has no symmetric partner.
                obj_coeffs[self._u_col(v_ix)] += -f
        else:
            for v_ix, v in enumerate(self.V_T):
                f = self.f_hat[v_ix]
                for j in self.grades:
                    # coeff on Qj(v_ix): f * (vj - costj)
                    obj_coeffs[self._q_col(j, v_ix)] += f * (v[j] - self.costs[j])
                # coeff on U(v_ix): -f
                obj_coeffs[self._u_col(v_ix)] += -f

        obj_coeffs *= self.n_buyers

        col_index = list(range(n_cols))
        self.solver.changeColsCost(n_cols, col_index, obj_coeffs.tolist())
        self.solver.changeObjectiveSense(highspy.ObjSense.kMaximize)

    @staticmethod
    def _add_row(solver, lower, upper, coeffs):
        """Add a single row to the HiGHS instance.

        `coeffs` is a dict[col_index -> coefficient]. We aggregate per column
        because HiGHS's `addRow` rejects rows that list a column index more than
        once. Returns the new row's index.
        """
        cols = list(coeffs.keys())
        vals = [coeffs[c] for c in cols]
        row_ix = solver.getNumRow()
        solver.addRow(lower, upper, len(cols), cols, vals)
        return row_ix

    def _add_base_constraints(self):
        """Add the Border-feasibility ('probability') base rows once.

        Per type index i:
          symmetric  -> Q[0][i] + Q[0][symmetric_ix(i)] <= 1
          asymmetric -> sum_j Q[j][i] <= 1
        The IIR boundary is handled via column 0's bound in `_create_columns`.
        """
        inf = highspy.kHighsInf
        for i in range(len(self.V_T)):
            coeffs = {}
            if not self.force_symmetric:
                for j in self.grades:
                    col = self._q_col(j, i)
                    coeffs[col] = coeffs.get(col, 0.0) + 1.0
            else:
                # On the grid diagonal i == symmetric_ix(i) -> single column,
                # coefficient 2.0 (HiGHS would otherwise reject the duplicate).
                col_i = self._q_col(0, i)
                coeffs[col_i] = coeffs.get(col_i, 0.0) + 1.0
                col_s = self._q_col(0, symmetric_ix(i, self.T))
                coeffs[col_s] = coeffs.get(col_s, 0.0) + 1.0
            self._add_row(self.solver, -inf, 1.0, coeffs)

    # ------------------------------------------------------------------
    # IIC / Border incremental rows
    # ------------------------------------------------------------------
    def _ic_coeffs(self, i, j):
        """Column coefficients for the discrete IIC row (true type i, report j):

            U[i] - U[j] - sum_k Q_k(j)*(v_i[k]-v_j[k]) >= 0   (mirrors `ic_lhs_minus_rhs`).

        Returns {column -> coefficient}. The `*` in the formula (coefficient times
        variable) is performed by the solver; here we only record coefficients.
        """
        v_i, v_j = self.V_T[i], self.V_T[j]
        coeffs = {}

        def acc(col, c):
            # Accumulate, don't overwrite: a single column can be hit by more than
            # one term -- e.g. the sum_k landing twice on the grid diagonal. HiGHS
            # rejects a row that lists the same column twice, so colliding
            # coefficients must be summed into one entry (this is the sum_k, not
            # the coefficient*variable product).
            coeffs[col] = coeffs.get(col, 0.0) + c

        # coeffs on U[i], U[j]: +1, -1
        acc(self._u_col(i), 1.0)
        acc(self._u_col(j), -1.0)

        # coeffs on Q_k(j): -(v_i[k] - v_j[k])
        if self.force_symmetric:
            acc(self._q_col(0, j), -(v_i[0] - v_j[0]))
            acc(self._q_col(0, symmetric_ix(j, self.T)), -(v_i[1] - v_j[1]))
        else:
            for k in self.grades:
                acc(self._q_col(k, j), -(v_i[k] - v_j[k]))
        return coeffs

    def _add_ic_row(self, i, j):
        if (i, j) in self._ic_rows:
            return
        inf = highspy.kHighsInf
        coeffs = self._ic_coeffs(i, j)
        # >= 0
        row_ix = self._add_row(self.solver, 0.0, inf, coeffs)
        self._ic_rows[(i, j)] = row_ix

    def _add_border_row(self, subset):
        """Add a Border subset row:
        N*sum_{v in A} f_hat[v]*(sum_j Q_j(v)) <= 1 - (sum_{v notin A} f_hat[v])^N
        i.e. lhs <= rhs, expressed as (lhs - rhs) <= 0 with rhs moved to bound.
        `subset` is an iterable of type indices.
        """
        key = frozenset(int(v) for v in subset)
        if key in self._border_rows:
            return
        inf = highspy.kHighsInf

        coeffs = {}

        def acc(col, c):
            coeffs[col] = coeffs.get(col, 0.0) + c

        for v_ix in key:
            f = self.f_hat[v_ix]
            if self.force_symmetric:
                acc(self._q_col(0, v_ix), self.n_buyers * f)
                acc(self._q_col(0, symmetric_ix(v_ix, self.T)), self.n_buyers * f)
            else:
                for j in self.grades:
                    acc(self._q_col(j, v_ix), self.n_buyers * f)

        # rhs = 1 - (sum_{v notin A} f_hat[v])^N
        setdiff = np.setdiff1d(np.arange(0, len(self.V_T), 1), list(key))
        outside = 0.0
        for v_ix in setdiff:
            outside += self.f_hat[v_ix]
        rhs = 1.0 - np.power(outside, self.n_buyers)

        row_ix = self._add_row(self.solver, -inf, float(rhs), coeffs)
        self._border_rows[key] = row_ix

    @staticmethod
    def _ic_key_from_name(name):
        ixs = name.lstrip(IC_PREFIX + "_").split("_")
        return int(ixs[0]), int(ixs[1])

    @staticmethod
    def _border_subset_from_name(name):
        str_subset = name.lstrip(BORDER_PREFIX + "_").split("_")
        return [int(v_t) for v_t in str_subset]

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    def _solve(self):
        """Run HiGHS and raise on any non-optimal terminal status."""
        self.solver.run()
        ms = self.solver.getModelStatus()
        if ms != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"solver failed with status={ms}")
        self.opt = self.solver.getInfoValue("objective_function_value")[1]

    def _read_solution(self):
        """Read the column solution into `_Q_values` / `_U_values`."""
        vals = list(self.solver.getSolution().col_value)
        self._U_values = [vals[self._u_col(v_ix)] for v_ix in range(len(self.V_T))]
        if not self.force_symmetric:
            self._Q_values = []
            for j in self.grades:
                self._Q_values.append([vals[self._q_col(j, v_ix)] for v_ix in range(len(self.V_T))])
        else:
            _Q1 = [vals[self._q_col(0, v_ix)] for v_ix in range(len(self.V_T))]
            _Q2 = np.zeros(len(_Q1))
            for i, _ in enumerate(_Q1):
                symm_i = symmetric_ix(i, self.T)
                _Q2[symm_i] = _Q1[i]
            self._Q_values = [_Q1, list(_Q2)]

    def run(self, solver_max_time=300):
        """Run the approximation algorithm.

        A single HiGHS instance is kept alive across all solver iterations within
        a `run()`; constraint rows are added incrementally and the solver is
        pinned to dual simplex (see below), so each re-solve warm-starts from the
        retained basis.

        Parameters
        ----------
        solver_max_time : int, default=300
            Solver time limit in seconds (HiGHS `time_limit` option).
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

        # one HiGHS instance for the whole run; rows are added incrementally.
        self.solver = highspy.Highs()
        self.solver.setOptionValue("output_flag", False)
        self.solver.setOptionValue("time_limit", float(solver_max_time))
        # Pin dual simplex: it warm-starts from the retained basis as rows are
        # added incrementally (interior-point does not), and it makes the
        # `simplex_iteration_count` logged in the inner loop meaningful.
        self.solver.setOptionValue("solver", "simplex")

        self._create_columns()
        self._set_objective()

        # structured constraint identities (no string re-parsing)
        self._ic_rows = {}  # (i, j) -> row_index
        self._border_rows = {}  # frozenset[int] -> row_index

        # base "probability" Border-feasibility rows, added once.
        self._add_base_constraints()

        self.converged = False
        while not self.converged:
            # Pre-seed the local IC rows (the 8-neighbour star around each type).
            # The full separation oracle below catches violations outside this
            # region and widens it by growing `net_size`; `_add_ic_row` de-dupes
            # rows already present.
            for i in range(len(self.V_T)):
                for j in star_indices(i, self.T, len(self.V_T)):
                    self._add_ic_row(i, j)

            j = 1
            n_violated = 0
            self.border_failing = True
            while self.border_failing:
                self._solve()

                solver_i = self.solver.getInfoValue("simplex_iteration_count")[1]

                n_ic = len(self._ic_rows)
                n_border = len(self._border_rows)

                msg = f"(i={self.i}, j={j}, solver={solver_i}) obj={self.opt}, #IC={n_ic}, #border={n_border} ({n_violated})"
                logging.info(msg)

                # convert solver vars to raw values
                self._read_solution()

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

                n_border_before = len(self._border_rows)
                for con in to_add:
                    subset = self._border_subset_from_name(con.name)
                    self._add_border_row(subset)
                if len(self._border_rows) == n_border_before:
                    # No new Border rows added but violations remain: the solver
                    # already enforces these within its feasibility tolerance, so
                    # re-solving cannot make progress. Stop rather than spin.
                    self.border_failing = False
                    break

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
                    i_ix, j_ix = self._ic_key_from_name(con.name)
                    logging.debug(
                        f"FAILURE [IC]: v={np.round(self.V_T[i_ix], 4)} (ix={i_ix}), v'={np.round(self.V_T[j_ix], 4)} (ix={j_ix})"
                    )
                    self._add_ic_row(i_ix, j_ix)
                for con in A_border:
                    subset = self._border_subset_from_name(con.name)
                    self._add_border_row(subset)

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
