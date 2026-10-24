#!/usr/bin/env python3
"""Exact audit of one reversible binding pair and its Ramsey witness.

Uses only Python's standard library and fractions.Fraction. No floating-point
matrix exponentials, stochastic simulations, or external data are required.

Run:
    python3 verify_exact_witness.py
    python3 verify_exact_witness.py --output exact_witness_checks.json

This is a fixed-pair algebraic verification. It does not establish global
identifiability, finite-shot performance, physical tau sensing, or novelty.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as F
from pathlib import Path


def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def matvec(a, v):
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def dot(u, v):
    return sum(x * y for x, y in zip(u, v))


def inverse(a):
    n = len(a)
    aug = [list(row) + ident for row, ident in zip(a, eye(n))]
    for col in range(n):
        pivot = next(i for i in range(col, n) if aug[i][col])
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [x / scale for x in aug[col]]
        for row in range(n):
            if row != col:
                scale = aug[row][col]
                aug[row] = [x - scale * y for x, y in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def encounter_generator(e):
    # a = d = 1; state order B1 B2 B3 U1 U2 U3.
    n = len(e)
    q = [[F(0) for _ in range(2 * n)] for _ in range(2 * n)]
    for i in range(n):
        q[i][i] = F(-1)
        q[i][i + n] = F(1)
        q[i + n][i] = F(1)
        for j in range(n):
            q[i + n][j + n] = e[i][j] - F(i == j)
    return q


def ramsey_derivative_polynomials(q, f, pi, max_order):
    """Coefficients of pi (Q + z D)^m 1, for m = 0 ... max_order.

    Entry k of each returned row multiplies z**k. Later substitute z = -i g.
    These are derivatives at T=0, so the Taylor coefficient is divided by m!.
    """
    size = len(q)
    v = [[F(1)] for _ in range(size)]
    out = [[F(1)]]
    for order in range(1, max_order + 1):
        nxt = [[F(0)] * (order + 1) for _ in range(size)]
        for i in range(size):
            for power in range(order):
                nxt[i][power] += sum(q[i][j] * v[j][power] for j in range(size))
                nxt[i][power + 1] += f[i] * v[i][power]
        v = nxt
        out.append([sum(pi[i] * v[i][k] for i in range(size)) for k in range(order + 1)])
    return out


def stringify(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: stringify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [stringify(item) for item in value]
    return value


def audit():
    n = 3
    e = [
        [F(-21, 20), F(1), F(1, 20)],
        [F(1), F(-2), F(1)],
        [F(1, 20), F(1), F(-21, 20)],
    ]
    c = [F(-1), F(-2), F(3)]
    w = [F(5), F(-4), F(-1)]
    r = [[F(i == j) - w[i] * w[j] / 21 for j in range(n)] for i in range(n)]
    ep = matmul(matmul(r, e), transpose(r))
    q, qp = encounter_generator(e), encounter_generator(ep)
    s = [[F(0) for _ in range(2 * n)] for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            s[i][j] = s[i + n][j + n] = r[i][j]
    f = c + [F(0)] * n
    pi = [F(1, 2 * n)] * (2 * n)
    ones = [F(1)] * (2 * n)
    left_cov = [pi[i] * f[i] for i in range(2 * n)]

    checks = {
        "R_orthogonal": matmul(r, transpose(r)) == eye(n),
        "R_fixes_ones": matvec(r, [F(1)] * n) == [F(1)] * n,
        "R_fixes_c": matvec(r, c) == c,
        "Q_alias_is_S_Q_S_transpose": qp == matmul(matmul(s, q), transpose(s)),
        "Q_row_sums_zero": matvec(q, ones) == [F(0)] * (2 * n),
        "Q_alias_row_sums_zero": matvec(qp, ones) == [F(0)] * (2 * n),
        "pi_stationary": matvec(transpose(q), pi) == [F(0)] * (2 * n),
        "pi_alias_stationary": matvec(transpose(qp), pi) == [F(0)] * (2 * n),
        "Q_symmetric_detailed_balance": q == transpose(q),
        "Q_alias_symmetric_detailed_balance": qp == transpose(qp),
        "S_fixes_f": matvec(s, f) == f,
        "left_covector_fixes_S": matvec(transpose(s), left_cov) == left_cov,
        "mean_detuning_zero": dot(pi, f) == 0,
        "E_positive_off_diagonal": all(e[i][j] > 0 for i in range(n) for j in range(n) if i != j),
        "E_alias_positive_off_diagonal": all(ep[i][j] > 0 for i in range(n) for j in range(n) if i != j),
    }

    derivatives = ramsey_derivative_polynomials(q, f, pi, 6)
    derivatives_alias = ramsey_derivative_polynomials(qp, f, pi, 6)
    differences = [[b - a for a, b in zip(base, alias)] for base, alias in zip(derivatives, derivatives_alias)]
    expected_sixth = [F(0)] * 7
    expected_sixth[3] = F(-152, 21)
    checks["derivative_differences_zero_through_order_5"] = all(all(x == 0 for x in row) for row in differences[:6])
    checks["sixth_derivative_difference_exact"] = differences[6] == expected_sixth

    c_squared = [x * x for x in c]
    moment = dot(c_squared, matvec(e, c))
    moment_alias = dot(c_squared, matvec(ep, c))
    checks["third_order_encounter_contrast_exact"] = moment_alias - moment == F(-152, 7)
    checks["sixth_coefficient_equals_two_covector_words"] = differences[6][3] == (moment_alias - moment) / 3
    factorial_six = 1 * 2 * 3 * 4 * 5 * 6
    taylor_z3 = differences[6][3] / factorial_six
    checks["factorial_and_T6_coefficient_exact"] = taylor_z3 == F(-19, 1890)

    p = inverse([[F(i == j) - e[i][j] for j in range(n)] for i in range(n)])
    pp = inverse([[F(i == j) - ep[i][j] for j in range(n)] for i in range(n)])
    checks["return_alias_is_R_P_R_transpose"] = pp == matmul(matmul(r, p), transpose(r))
    checks["partner_2_return_differs"] = pp[1][1] != p[1][1]
    checks["mean_same_partner_return_invariant"] = sum(p[i][i] for i in range(n)) == sum(pp[i][i] for i in range(n))

    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
    return stringify({
        "status": "All checks passed with exact rational arithmetic",
        "scope": "Fixed pair separation only; no global identification, finite-shot performance, physical tau result, or novelty claim",
        "parameters": {"a": F(1), "d": F(1), "state_order": ["B1", "B2", "B3", "U1", "U2", "U3"], "c": c, "w": w},
        "E": e,
        "R": r,
        "E_alias": ep,
        "stationary_distribution": pi,
        "checks": checks,
        "derivative_polynomials_base_in_z": derivatives,
        "derivative_polynomials_alias_in_z": derivatives_alias,
        "derivative_polynomials_alias_minus_base_in_z": differences,
        "c_squared_E_c_base": moment,
        "c_squared_E_c_alias": moment_alias,
        "c_squared_E_c_difference": moment_alias - moment,
        "T6_coefficient_of_z3_alias_minus_base": taylor_z3,
        "substitution": "z=-i*g; z^3=+i*g^3, so delta W(T)=-(19/1890)*i*g^3*T^6+O(T^7) for fixed real g",
        "at_g_one_half": {"sixth_derivative_delta_W": "-(19/21)*i", "T6_coefficient_delta_W": "-(19/15120)*i"},
        "return_matrix_base": p,
        "return_matrix_alias": pp,
        "P22_alias_minus_base": pp[1][1] - p[1][1],
        "mean_same_partner_return": sum(p[i][i] for i in range(n)) / n,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output file; otherwise print JSON")
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"All {len(result['checks'])} exact checks passed; wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
