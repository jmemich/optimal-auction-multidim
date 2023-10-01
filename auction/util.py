def symmetric_ix(i, T):
    # this is used for `force_symmetric=True`
    # imagine a matrix and transpose i,j....
    i_ = int(i / (T + 1))
    j_ = i % (T + 1)
    return j_ * (T + 1) + i_
