def symmetric_ix(i, T):
    # this is used for `force_symmetric=True`
    # imagine a matrix and transpose i,j....
    i_ = int(i / (T + 1))
    j_ = i % (T + 1)
    return j_ * (T + 1) + i_


# NOTE these are here for ease of serialization
def corrU(vs, fs):
    # correlated uniform distribution `f(x,y)=x+y` on [0,1]^2
    return vs[0] + vs[1]


def belloni_f(x):
    # uniform distribution on [a,a+2]
    return 1 / 2
