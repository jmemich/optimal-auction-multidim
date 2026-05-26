import json
import logging
from pathlib import Path

import pytest
import scipy.stats

from optimal_auctions import OptimalAuctionApproximation

_BASELINES_PATH = Path(__file__).parent.parent.parent / "docs" / "baselines.json"
_DATA = json.loads(_BASELINES_PATH.read_text())
TOLERANCE = _DATA["tolerance_used_for_match"]


def _corr_u(v, fs):
    # joint density f(x1, x2) = x1 + x2 for setting 3 (symmetric correlated)
    # integral over [0,1]^2 equals 1, so no rescaling needed
    return v[0] + v[1]


# settings whose f or corr can't be reconstructed from the JSON type="callable" entries
_SETTING_OVERRIDES = {
    "3": {"corr": _corr_u},
    # setting 4: belloni_f is U[6,8] x U[9,11]; scipy.stats.uniform(loc, scale) => support [loc, loc+scale]
    "4": {
        "f": [
            scipy.stats.uniform(loc=6, scale=2).pdf,
            scipy.stats.uniform(loc=9, scale=2).pdf,
        ]
    },
}


def _build_cases():
    cases = []
    for setting in _DATA["settings"]:
        sid = setting["id"]
        overrides = _SETTING_OVERRIDES.get(sid, {})
        for run in setting["runs"]:
            if "f" in overrides:
                f = overrides["f"]
            else:
                f = [
                    getattr(scipy.stats, fi["dist"])(*fi["args"], **fi["kwds"]).pdf
                    for fi in run["f"]
                ]
            cases.append(
                pytest.param(
                    dict(
                        n_buyers=run["n_buyers"],
                        V=run["V"],
                        costs=run["costs"],
                        T=run["T"],
                        f=f,
                        corr=overrides.get("corr"),
                        force_symmetric=run["force_symmetric"],
                        check_local_ic=run["check_local_ic"],
                        log_level=logging.WARNING,
                    ),
                    run["revenue"],
                    id=f"s{sid}_T{run['T']}",
                )
            )
    return cases


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize("kwargs,expected_revenue", _build_cases())
def test_baseline_revenue(kwargs, expected_revenue):
    approx = OptimalAuctionApproximation(**kwargs)
    approx.run()
    assert abs(approx.opt - expected_revenue) <= TOLERANCE, (
        f"revenue {approx.opt:.6f} differs from baseline {expected_revenue:.6f} "
        f"by {abs(approx.opt - expected_revenue):.2e} (tol={TOLERANCE:.2e})"
    )
