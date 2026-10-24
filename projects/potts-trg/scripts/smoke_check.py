"""Small numerical controls and an audit of existing logs; no high-D campaign."""

from __future__ import annotations

import json
import math
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from critline import p_c
from dtrg import kappa_lr
from exact import houtappel, onsager, series_square, series_tri
from rc_tensor import square_tensors, triangular_block
from run_kappa import f_of_q
from trg_u1 import check_u1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def saved_log_summary() -> dict:
    pattern = re.compile(r"D=\s*(\d+)\s+f\(1\)=(\S+)\s+kappa3=(\S+)\s+kappa5=(\S+)")
    records = {}
    for name in ("log_cl_sq_0.05.txt", "log_cl_sq_0.1.txt",
                 "log_cl_tri_0.05.txt", "log_cl_tri_0.1.txt"):
        rows = [{"D": int(m[1]), "f_at_q1": float(m[2]),
                 "kappa3": float(m[3]), "kappa5": float(m[4])}
                for m in pattern.finditer((ROOT / "logs" / name).read_text())]
        require([r["D"] for r in rows] == [8, 16, 24, 32, 48], f"incomplete log: {name}")
        records[name] = rows
    square_exact = (3 * math.sqrt(3) - 5) / 2
    square48 = records["log_cl_sq_0.05.txt"][-1]["kappa5"]
    tri = records["log_cl_tri_0.05.txt"]
    return {"square_analytical_reference": square_exact,
            "square_D48_error_from_printed_log": square48 - square_exact,
            "triangular_D32_to_D48_change_from_printed_log": tri[-1]["kappa5"] - tri[-2]["kappa5"],
            "triangular_converged_error_bar_established": False,
            "saved_ladders": records}


def main() -> None:
    references = {
        ("square", 1): 0.5,
        ("triangular", 1): 2 * math.sin(math.pi / 18),
        ("square", 2): 1 - math.exp(-math.log(1 + math.sqrt(2))),
        ("triangular", 2): 1 - math.exp(-math.asinh(1 / math.sqrt(3))),
    }
    critical_errors = {f"{lattice}_q{q}": float(p_c(lattice, q) - reference)
                       for (lattice, q), reference in references.items()}
    require(max(map(abs, critical_errors.values())) < 1e-12, "critical-line check failed")
    require(abs(onsager(0, n=128) - math.log(2)) < 1e-14, "square normalization failed")
    require(abs(houtappel(0, n=32) - math.log(2)) < 1e-14, "triangular normalization failed")

    # The series are truncated: these residuals include omitted series terms.
    series_errors = {
        "square_K0.05": float(onsager(.05, n=4096) - series_square(.05)),
        "triangular_K0.05": float(houtappel(.05, n=128) - series_tri(.05)),
    }
    require(abs(series_errors["square_K0.05"]) < 1e-8, "square small-K series check failed")
    require(abs(series_errors["triangular_K0.05"]) < 3e-6, "triangular small-K series check failed")
    for q in (1.0, 2.0):
        square_a, square_b, charges = square_tensors(p_c("square", q), q)
        triangular, triangular_charges = triangular_block(p_c("triangular", q), q)
        for tensor, charge in ((square_a, charges), (square_b, charges),
                               (triangular, triangular_charges)):
            require(check_u1(tensor, charge)[0] == 0, "tensor charge conservation failed")

    # Two square steps need no rank truncation at D=64. Both methods evaluate
    # the same finite contraction, rather than an infinite-lattice derivative.
    step = 1e-5
    _, dual = kappa_lr("square", .5, D=64, nsteps=2)
    finite_difference = (f_of_q("square", .5, 1 + step, D=64, nsteps=2)
                         - f_of_q("square", .5, 1 - step, D=64, nsteps=2)) / (2 * step)
    derivative_error = float(dual - finite_difference)
    require(abs(derivative_error) < 1e-7, "small-contraction derivative check failed")

    result = {"status": "passed", "python": platform.python_version(),
              "numpy": np.__version__, "critical_line_errors": critical_errors,
              "small_K_quadrature_minus_truncated_series": series_errors,
              "tensor_charge_checks": 6,
              "square_two_step_dual_minus_finite_difference": derivative_error,
              "full_convergence_or_historical_campaign_reproduced": False,
              "saved_log_audit": saved_log_summary()}
    output = ROOT / "results/smoke-check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: critical lines, normalization, series, six charge checks and finite-contraction derivative.")
    print("Saved-log audit: triangular D32-to-D48 change is 0.0000562; no converged error bar established.")


if __name__ == "__main__":
    main()
