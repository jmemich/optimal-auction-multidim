from time import time
import logging

import numpy as np

from auction.constraints import (ic_lhs_minus_rhs, border_lhs_minus_rhs,
                                 Constraint, BORDER_PREFIX, IC_PREFIX)


logger = logging.getLogger(__name__)


def _make_lower_left_quadrant(ix, T, net_size):
    # NOTE this function generates the indexes for the local region of the
    # typespace used to check IC constraints based on `net_size`
    if net_size < 2:
        return []

    i = int(ix / (T + 1))  # row
    j = ix % (T + 1)       # col
    grid = np.zeros((T + 1, T + 1))
    for offset_i in range(net_size + 1):
        new_i = i + offset_i
        if new_i >= 0 and new_i <= T:
            for offset_j in range(net_size + 1):
                new_j = j - offset_j
                if new_j >= 0 and new_j <= T:
                    grid[new_i, new_j] = 1

    # convert matrix to 1D array and pull values where = 1
    ixs = np.where(grid.reshape((-1)) > 0.5)[0]
    return list(ixs)


def _check_ic(Q, U, V_T, T, grades, check_local, net_size):
    # the way we check for local IC violations is as follows:
    #
    #         | x x x
    #         | x o x
    #         | x x x
    #         |_,_,_,_
    #
    #           star
    #          pattern
    #
    # where `o` is `v_i` and `x`s are the `v_j`s checked for incentive-
    # compatibility. Note that if no `check_local_ic` option is provided
    # then `v_i` is checked against *all* other `v_j`s.
    n_cons = len(V_T)
    if check_local:
        n_cons *= 8  # NOTE `star` pattern above
    else:
        n_cons *= n_cons
    logger.debug('checking %s IC constraints...' % n_cons)

    start = time()
    ic_cons = []
    for i, v_i in enumerate(V_T):
        if check_local:
            star_ix = [i + T,      # above-left
                       i + T + 1,  # above
                       i + T + 2,  # above-right
                       i - 1,      # left
                       i + 1,      # right
                       i - T - 2,  # below-left
                       i - T - 1,  # below
                       i - T]      # below-right
            # deal with corners/edges:
            star_ix = [ix for ix in star_ix
                       if ix >= 0 and ix < len(V_T)]
            quadrant_ix = _make_lower_left_quadrant(i, T, net_size)
            all_ix = set(star_ix + quadrant_ix)  # get unique
            all_v_j = [V_T[ix] for ix in all_ix]
            inner_loop = zip(all_ix, all_v_j)
        else:
            inner_loop = enumerate(V_T)

        for j, v_j in inner_loop:
            con = _check_one_ic(Q, U, grades, i, v_i, j, v_j)
            ic_cons.append(con)

    logger.debug('ic constraint checks completed!')
    end = time()
    elapsed = (end - start)
    n_violated, n_binding, n_inactive = 0, 0, 0
    for c in ic_cons:
        if c.status == 'VIOLATED':
            n_violated += 1
        if c.status == 'BINDING':
            n_binding += 1
        if c.status == 'INACTIVE':
            n_inactive += 1
    logger.debug(
        ('time taken (%s mins), # violated (%s), # binding (%s), '
         '# inactive (%s)') %
        (np.round(elapsed / 60, 1), n_violated, n_binding, n_inactive))

    return ic_cons


def _check_one_ic(Q, U, grades, i, v_i, j, v_j):
    name = '%s_%s_%s' % (IC_PREFIX, i, j)

    # NOTE we `force_symmetric`=False here because we are using Q_vals so
    # the indexing switch has already been done!
    T, force_symmetric = None, False
    val = ic_lhs_minus_rhs(Q, U, T, grades, i, v_i, j, v_j, force_symmetric)

    # NOTE becase we are comparing floating point numbers we eval
    # `is_close` before we asses > / <
    is_close = np.isclose(val, 0.0)
    if is_close:
        con = Constraint(name, None, 'BINDING')
    elif val < 0:
        con = Constraint(name, None, 'VIOLATED')
    else:  # lhs_minus_rhs > 0:
        con = Constraint(name, None, 'INACTIVE')

    return con


def _check_border(V_T, T, Q, grades, n_buyers, f_hat, problem_type):
    start = time()
    border_cons = []

    # TODO how to break ties for `Q_hat_ix`?
    if problem_type == 'single_good':
        # add Qj's together for each point in V_T
        Q_upperbar = np.array(Q).sum(axis=0)

        Q_hat_ix = np.argsort(Q_upperbar)[::-1]  # descending

        E_Qs = []
        for i in range(len(V_T)):
            subset_ix = list(range(0, i + 1))
            subset = Q_hat_ix[subset_ix]

            E_Qs.append(subset.tolist())

        V_T_subsets = E_Qs

        for V_T_subset in V_T_subsets:
            con = _check_one_border(
                T, V_T, V_T_subset, Q, n_buyers, grades, f_hat)
            border_cons.append(con)

    elif problem_type == 'multi_good':
        # we use the belloni algorithm separately for each object (grade) j
        for j in grades:
            Q_hat_ix = np.argsort(Q[j])[::-1]  # to get descending

            E_Qs = []
            for i in range(len(V_T)):
                subset_ix = list(range(0, i + 1))
                subset = Q_hat_ix[subset_ix]

                E_Qs.append(subset.tolist())

            V_T_subsets = E_Qs

            for V_T_subset in V_T_subsets:
                con = _check_one_border(
                    T, V_T, V_T_subset, Q, n_buyers, [j], f_hat)
                border_cons.append(con)

    logger.debug('border constraint checks completed!')
    end = time()
    elapsed = (end - start)
    n_violated, n_binding, n_inactive = 0, 0, 0
    for c in border_cons:
        if c.status == 'VIOLATED':
            n_violated += 1
        if c.status == 'BINDING':
            n_binding += 1
        if c.status == 'INACTIVE':
            n_inactive += 1
    logger.debug(
        ('time taken (%s mins), # violated (%s), # binding (%s), '
         '# inactive (%s)') %
        (np.round(elapsed / 60, 1), n_violated, n_binding, n_inactive))

    return border_cons


def _check_one_border(T, V_T, V_T_subset, Q, n_buyers, grades, f_hat):
    name = '%s_%s' % (
        BORDER_PREFIX, '_'.join([str(i) for i in V_T_subset]))

    # NOTE we `force_symmetric`=False here because we are using Q_vals so
    # the indexing switch has already been done!
    val = border_lhs_minus_rhs(
        T, V_T, V_T_subset, Q, n_buyers, grades, f_hat, force_symmetric=False)

    # NOTE becase we are comparing floating point numbers we eval
    # `is_close` before we asses > / <
    is_close = np.isclose(val, 0.0)
    if is_close:
        con = Constraint(name, None, 'BINDING')
    elif val > 0:
        con = Constraint(name, None, 'VIOLATED')
    else:  # lhs_minus_rhs < 0:
        con = Constraint(name, None, 'INACTIVE')

    return con


def separation_oracle(
        Q, U, V_T, T, grades, n_buyers, f_hat, check_local_ic, problem_type,
        net_size):
    # TODO multiprocess...
    logger.debug('starting separation oracle...')
    I_ic, A_ic, B_ic = [], [], []
    ic_cons = _check_ic(
        Q, U, V_T, T, grades, check_local_ic, net_size)
    for ic_con in ic_cons:
        if ic_con.status == 'INACTIVE':
            I_ic.append(ic_con)
        if ic_con.status == 'VIOLATED':
            A_ic.append(ic_con)
        if ic_con.status == 'BINDING':
            B_ic.append(ic_con)

    I_border, A_border, B_border = [], [], []
    # if N=1 skip Border checks
    if n_buyers != 1:
        border_cons = _check_border(
            V_T, T, Q, grades, n_buyers, f_hat, problem_type)
        for border_con in border_cons:
            if border_con.status == 'INACTIVE':
                I_border.append(border_con)
            if border_con.status == 'VIOLATED':
                A_border.append(border_con)
            if border_con.status == 'BINDING':
                B_border.append(border_con)

    logger.debug('separation oracle completed!')
    return I_ic, A_ic, B_ic, I_border, A_border, B_border


def make_subsets(
        I_ic, A_ic, I_border, A_border, I_subset_prop, A_subset_prop, rng,
        check_local_ic):
    I, A = [], []

    logger.debug('total # inactive constraints: %s' % len(I_ic + I_border))
    logger.debug('total # violated constraints: %s' % len(A_ic + A_border))

    # TODO separate subsetting for IC constraints?
    if not check_local_ic:
        if len(I_ic) > 0:
            n_I_ic_subset = np.round(
                max(len(I_ic) * I_subset_prop, 1)).astype('int')
            I_ic_subset_ix = rng.randint(0, len(I_ic), n_I_ic_subset)
            for ic_con_ix in I_ic_subset_ix:
                I.append(I_ic[ic_con_ix])

        if len(A_ic) > 0:
            n_A_ic_subset = np.round(
                max(len(A_ic) * A_subset_prop, 1)).astype('int')
            A_ic_subset_ix = rng.randint(0, len(A_ic), n_A_ic_subset)
            for ic_con_ix in A_ic_subset_ix:
                A.append(A_ic[ic_con_ix])
    else:
        if len(I_ic) > 0:
            I.extend(I_ic)
        if len(A_ic) > 0:
            A.extend(A_ic)

    I.extend(I_border)
    A.extend(A_border)

    return I, A
