from approx import OptimalAuctionApproximation as Approx

approx = Approx(
    n_buyers=2,
    V=[[0,1],[0,1]],
    costs=[0,0],
    T=20)
approx.run(max_iter=20)
