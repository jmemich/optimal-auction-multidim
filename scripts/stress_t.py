"""Stress-test the approximation as T grows on the Pavlov U[0,1]^2 baseline
(Setting 1a: N=2, V=[[0,1],[0,1]], costs=[0,0], independent uniform).

Records wall-time, revenue and iteration/constraint counts for each T and
*appends to docs/stress_t.json after every run*, so partial results survive an
interrupt. Revenue is cross-checked against docs/baselines.json where available.

Usage:
    python scripts/stress_t.py 5 10 15 20 25 30
    python scripts/stress_t.py 40 --budget 120   # stop before a T if the
                                                  # previous run exceeded budget(s)
"""

import argparse
import json
import time
from pathlib import Path

from optimal_auctions import OptimalAuctionApproximation as Approximation

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "stress_t.json"
BASELINES = ROOT / "docs" / "baselines.json"


def baseline_revenue(T):
    """Recorded Setting-1a revenue at this T, or None."""
    d = json.loads(BASELINES.read_text())
    for s in d["settings"]:
        if s["id"] == "1a":
            for r in s["runs"]:
                if r["T"] == T:
                    return r["revenue"]
    return None


def load_results():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return []


def save_results(rows):
    OUT.write_text(json.dumps(rows, indent=2))


def run_one(T):
    n_types = (T + 1) ** 2
    approx = Approximation(n_buyers=2, V=[[0, 1], [0, 1]], costs=[0, 0], T=T, log_level="warning")
    t0 = time.time()
    approx.run()
    elapsed = time.time() - t0

    base = baseline_revenue(T)
    rev = float(approx.opt)
    row = {
        "T": T,
        "n_types": n_types,
        "n_vars": int(approx.solver.getNumCol()),
        "n_rows": int(approx.solver.getNumRow()),
        "n_ic_rows": len(approx._ic_rows),
        "n_border_rows": len(approx._border_rows),
        "outer_iters": int(approx.i),
        "inner_iters_sum": int(sum(approx.js)),
        "elapsed_s": round(elapsed, 3),
        "revenue": rev,
        "converged": bool(approx.converged),
        "baseline_revenue": base,
        "revenue_abs_err": (abs(rev - base) if base is not None else None),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("T", nargs="+", type=int, help="T values to run, in order")
    ap.add_argument(
        "--budget",
        type=float,
        default=None,
        help="seconds: skip remaining T once a run exceeds this",
    )
    args = ap.parse_args()

    rows = load_results()
    done = {r["T"] for r in rows}

    hdr = f"{'T':>4} {'n_types':>8} {'rows':>7} {'outer':>6} {'inner':>6} {'elapsed_s':>10} {'revenue':>10} {'abs_err':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["T"]):
        _print_row(r)

    last_elapsed = rows[-1]["elapsed_s"] if rows else 0.0
    for T in args.T:
        if T in done:
            print(f"# T={T} already recorded, skipping")
            continue
        if args.budget is not None and last_elapsed > args.budget:
            print(
                f"# previous run {last_elapsed:.1f}s > budget {args.budget}s; stopping before T={T}"
            )
            break
        row = run_one(T)
        rows.append(row)
        rows.sort(key=lambda x: x["T"])
        save_results(rows)  # persist after every run
        _print_row(row)
        last_elapsed = row["elapsed_s"]


def _print_row(r):
    err = f"{r['revenue_abs_err']:.2e}" if r["revenue_abs_err"] is not None else "    --"
    flag = "" if r["converged"] else "  !NOCONV"
    print(
        f"{r['T']:>4} {r['n_types']:>8} {r['n_rows']:>7} {r['outer_iters']:>6} "
        f"{r['inner_iters_sum']:>6} {r['elapsed_s']:>10.3f} {r['revenue']:>10.6f} {err:>9}{flag}"
    )


if __name__ == "__main__":
    main()
