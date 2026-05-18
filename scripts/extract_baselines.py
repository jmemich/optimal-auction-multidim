"""One-shot M0: unpickle auctions-qualitative artifacts → docs/baselines.json.

Reads the 24 pickle files in `auctions-qualitative/scripts/runs/` produced by
`BelloniAuctionApproximation` runs (the historical thesis-generating code),
extracts the metadata we need as a regression baseline, and writes a single
JSON file that becomes the frozen historical-truth contract for the refactor.

After this runs successfully, optimal-auction-multidim never imports anything
from auctions-qualitative again — `docs/baselines.json` is the only artifact
that crosses repo boundaries.

Usage:
    python scripts/extract_baselines.py

The script uses a stub-class unpickler so the `auction` package does NOT need
to be installed or on PYTHONPATH. If you do have it importable, real classes
take precedence over stubs.

Exit codes:
    0 = all 24 pickles loaded and revenues match thesis within tolerance
    1 = setup error (missing pickle dir, etc.)
    2 = at least one mismatch or missing pickle (baselines.json still written)
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
OAM_ROOT = HERE.parent
AQ_ROOT = OAM_ROOT.parent / "auctions-qualitative"
PICKLE_DIR = AQ_ROOT / "scripts" / "runs"
OUT_PATH = OAM_ROOT / "docs" / "baselines.json"

# Canonical settings — mirrors docs/benchmarks.md. Order matters for output.
SETTINGS: list[dict[str, Any]] = [
    {
        "id": "1a",
        "name": "symmetric_independent_unif_01",
        "thesis_label": "Pavlov U[0,1]^2",
        "expected_revenue": {5: 0.66094, 10: 0.625929, 15: 0.612877, 20: 0.606033},
    },
    {
        "id": "1b",
        "name": "symmetric_independent_unif_23",
        "thesis_label": "Pavlov U[2,3]^2 (stochastic optimal)",
        "expected_revenue": {5: 2.622409, 10: 2.58287, 15: 2.58287, 20: 2.562314},
    },
    {
        "id": "2",
        "name": "symmetric_independent_beta",
        "thesis_label": "Symmetric Beta(1,2)^2",
        "expected_revenue": {5: 0.448709, 10: 0.418815, 15: 0.406948, 20: 0.400615},
    },
    {
        "id": "3",
        "name": "symmetric_correlated_unif",
        "thesis_label": "Symmetric correlated x1+x2",
        "expected_revenue": {5: 0.75159, 10: 0.718683, 15: 0.705905, 20: 0.698962},
    },
    {
        "id": "4",
        "name": "asymmetric_independent_belloni",
        "thesis_label": "Belloni asymmetric uniform",
        "expected_revenue": {5: 6.02496, 10: 5.941549, 15: 5.91222, 20: 5.893113},
    },
    {
        "id": "5",
        "name": "asymmetric_independent_truncnorm",
        "thesis_label": "Asymmetric truncnorm",
        "expected_revenue": {5: 2.779996, 10: 2.749451, 15: 2.736171, 20: 2.729342},
    },
]

T_VALUES = [5, 10, 15, 20]
TOLERANCE = 1e-4  # thesis reports 5-6 dp; 1e-4 catches typos without false alarms


def _make_stub(name: str, module: str) -> type:
    """Generic stub class that accepts any state dict from pickle."""
    def setstate(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._pickled_state = state

    def init(self, *args, **kwargs):
        pass

    return type(name, (object,), {
        "__setstate__": setstate,
        "__init__": init,
        "__module__": module,
    })


class StubUnpickler(pickle.Unpickler):
    """Substitutes a stub class whenever pickle asks for an `auction.*` symbol.

    If the real `auction` package is importable (PYTHONPATH set, or installed)
    the parent class's lookup runs first and returns the real class — stubs are
    only used when import fails. This makes the script work either way.
    """
    def find_class(self, module: str, name: str) -> Any:
        if module == "auction" or module.startswith("auction."):
            try:
                return super().find_class(module, name)
            except (ImportError, ModuleNotFoundError, AttributeError):
                # Lazily create a parent module if needed so future lookups
                # under the same module path also work.
                if module not in sys.modules:
                    sys.modules[module] = ModuleType(module)
                stub = _make_stub(name, module)
                setattr(sys.modules[module], name, stub)
                return stub
        return super().find_class(module, name)


def _jsonable(x: Any) -> Any:
    """Convert numpy types / non-trivial objects to JSON-friendly form."""
    if x is None:
        return None
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return repr(x)


def _describe_pdf(fi: Any) -> Any:
    """Best-effort structured description of a PDF callable.

    Recognises `scipy.stats.rv_frozen` (returned by e.g. `truncnorm(a, b, loc, scale)`)
    and pulls out (dist name, args, kwds) so we can later re-create the same
    distribution without consulting the pickle. Falls back to repr for anything else.
    """
    if fi is None:
        return None
    rv = getattr(fi, "__self__", None)  # bound .pdf method → underlying rv_frozen
    if rv is None and type(fi).__name__ == "rv_frozen":
        rv = fi
    if rv is not None and type(rv).__name__ == "rv_frozen":
        dist = getattr(rv, "dist", None)
        return {
            "type": "scipy.stats.rv_frozen",
            "dist": getattr(dist, "name", repr(dist)),
            "args": [_jsonable(a) for a in getattr(rv, "args", ())],
            "kwds": {k: _jsonable(v) for k, v in getattr(rv, "kwds", {}).items()},
        }
    return {"type": "callable", "repr": repr(fi)}


def _shape_of(obj: Any) -> list[int] | None:
    """Best-effort shape extraction for arrays / nested lists. None if unknown."""
    if obj is None:
        return None
    try:
        import numpy as np
        return list(np.asarray(obj, dtype=object).shape)
    except Exception:
        try:
            return [len(obj)]
        except Exception:
            return None


def _merged_state(approx: Any) -> dict[str, Any]:
    """Flatten an approx instance into one dict.

    AQ's `BelloniAuctionApproximation` serializes via `__getstate__` that
    returns `{'init_params': {...}, 'run_params': {...}}` rather than
    using direct attributes. Direct-attribute layouts also work as fallback.
    """
    raw = vars(approx) if hasattr(approx, "__dict__") else {}
    flat: dict[str, Any] = {}
    init = raw.get("init_params")
    run = raw.get("run_params")
    if isinstance(init, dict):
        flat.update(init)
    if isinstance(run, dict):
        flat.update(run)
    # direct attrs win only for keys not in init/run dicts (covers future layouts)
    for k, v in raw.items():
        if k not in ("init_params", "run_params") and k not in flat:
            flat[k] = v
    return flat


def _extract(approx: Any) -> dict[str, Any]:
    """Pull the regression-relevant fields out of an unpickled approx object."""
    s = _merged_state(approx)
    f_attr = s.get("f")
    return {
        "n_buyers": _jsonable(s.get("n_buyers")),
        "n_grades": _jsonable(s.get("n_grades")),
        "V": _jsonable(s.get("V")),
        "costs": _jsonable(s.get("costs")),
        "T": _jsonable(s.get("T")),
        "revenue": _jsonable(s.get("opt")),
        "converged": _jsonable(s.get("converged")),
        "elapsed_s": _jsonable(s.get("elapsed")),
        "iterations": _jsonable(s.get("i")),
        "n_constraints": (
            len(s["con_names"]) if isinstance(s.get("con_names"), (set, list)) else None
        ),
        "f": (
            [_describe_pdf(fi) for fi in f_attr]
            if f_attr is not None and hasattr(f_attr, "__iter__") else _describe_pdf(f_attr)
        ),
        "corr_repr": repr(s.get("corr")),
        "border_type": _jsonable(s.get("border_type")),
        "force_symmetric": _jsonable(s.get("force_symmetric")),
        "check_local_ic": _jsonable(s.get("check_local_ic")),
        "Q_shape": _shape_of(s.get("_Q_values") or s.get("Q")),
        "U_shape": _shape_of(s.get("_U_values") or s.get("U")),
        "state_keys": sorted(s.keys()),
    }


def main() -> int:
    if not PICKLE_DIR.exists():
        print(f"ERROR: pickle dir does not exist: {PICKLE_DIR}", file=sys.stderr)
        return 1

    n_expected = len(SETTINGS) * len(T_VALUES)
    print(
        f"Extracting {len(SETTINGS)} settings x {len(T_VALUES)} T values "
        f"= {n_expected} pickles from {PICKLE_DIR}\n"
    )

    output: dict[str, Any] = {
        "schema_version": 1,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source_pickles_dir": str(PICKLE_DIR),
        "tolerance_used_for_match": TOLERANCE,
        "notes": (
            "Generated by scripts/extract_baselines.py. Source: AQ pickle "
            "artifacts produced on the fix-distributions branch of "
            "auctions-qualitative. After this file exists, optimal-auction-multidim "
            "is the source of truth for regression baselines; AQ is read-only "
            "archive only."
        ),
        "settings": [],
    }
    n_match = n_diverge = n_missing = n_error = 0

    for setting in SETTINGS:
        s_out: dict[str, Any] = {
            "id": setting["id"],
            "name": setting["name"],
            "thesis_label": setting["thesis_label"],
            "runs": [],
        }
        for T in T_VALUES:
            pkl = PICKLE_DIR / f"{setting['name']}_{T}.pkl"
            entry: dict[str, Any] = {"T": T, "pickle_file": pkl.name}

            if not pkl.exists():
                print(f"  [MISSING] {setting['id']:3s} T={T:2d}  {pkl.name}")
                entry["status"] = "missing"
                n_missing += 1
                s_out["runs"].append(entry)
                continue

            try:
                with pkl.open("rb") as f:
                    approx = StubUnpickler(f).load()
                data = _extract(approx)
                entry.update(data)
                expected = setting["expected_revenue"].get(T)
                actual = data["revenue"]
                if expected is not None and isinstance(actual, (int, float)):
                    diff = abs(actual - expected)
                    matches = diff < TOLERANCE
                    entry["thesis_expected_revenue"] = expected
                    entry["abs_diff_from_thesis"] = diff
                    entry["matches_thesis"] = matches
                    status = "OK     " if matches else "DIVERGE"
                    print(
                        f"  [{status}] {setting['id']:3s} T={T:2d}  "
                        f"rev={actual:.6f}  thesis={expected:.6f}  |delta|={diff:.2e}"
                    )
                    if matches:
                        n_match += 1
                    else:
                        n_diverge += 1
                else:
                    print(f"  [LOADED ] {setting['id']:3s} T={T:2d}  rev={actual!r} (no expected to compare)")
                    entry["status"] = "loaded_no_compare"
            except Exception as e:
                print(f"  [ERROR  ] {setting['id']:3s} T={T:2d}  {type(e).__name__}: {e}")
                entry["status"] = "error"
                entry["error"] = f"{type(e).__name__}: {e}"
                n_error += 1
            s_out["runs"].append(entry)
        output["settings"].append(s_out)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nWrote {OUT_PATH}")
    print(
        f"Summary: {n_match} match thesis, {n_diverge} diverge, "
        f"{n_missing} missing, {n_error} error (of {n_expected} total)"
    )
    return 0 if (n_diverge == 0 and n_missing == 0 and n_error == 0) else 2


if __name__ == "__main__":
    sys.exit(main())
