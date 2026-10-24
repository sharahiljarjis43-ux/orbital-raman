"""Numerical checks of the proposed sharp stationary moment-order theorem.

This is a mathematical verification script, not a simulation or finite-shot test.
The proof is algebraic; numerical checks are supplementary.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm


def compositions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for first in range(1, total - parts + 2):
            for tail in compositions(total - first, parts - 1):
                yield (first,) + tail


def generator(E, a, d):
    n = len(E)
    eye = np.eye(n)
    return np.block([[-d * eye, d * eye], [a * eye, E - a * eye]])


def moment(pi, f, q, powers, gaps):
    row = pi * f ** powers[0]
    for gap, power in zip(gaps, powers[1:]):
        row = (row @ expm(q * gap)) * f ** power
    return float(row.sum())


def one_check(c):
    c = np.asarray(c, dtype=float)
    c /= np.max(np.abs(c))
    n = len(c)
    a, d, lam, delta = 0.8, 1.1, 1.3, 0.2
    vand = np.stack([c ** k for k in range(n - 1)])
    w = np.linalg.svd(vand, full_matrices=True)[2][-1]
    j = np.ones((n, n)) / n
    e0 = lam * (j - np.eye(n))
    e1 = e0 + delta * np.outer(w, w)
    q0, q1 = generator(e0, a, d), generator(e1, a, d)
    pi = np.r_[np.full(n, a / (n * (a + d))),
               np.full(n, d / (n * (a + d)))]
    f = np.r_[c, np.zeros(n)]
    low_error = 0.0
    cases = 0
    # All compositions into one, two or three observation times, plus all
    # first-power products at the maximum subthreshold order.
    gap_pattern = (0.17, 0.43)
    for order in range(1, 2 * n - 2):
        for parts in range(1, min(order, 3) + 1):
            for powers in compositions(order, parts):
                gaps = gap_pattern[:parts - 1]
                diff = abs(moment(pi, f, q0, powers, gaps)
                           - moment(pi, f, q1, powers, gaps))
                low_error = max(low_error, diff)
                cases += 1
    powers = (1,) * (2 * n - 3)
    gaps = tuple(0.11 + 0.01 * k for k in range(len(powers) - 1))
    low_error = max(low_error, abs(moment(pi, f, q0, powers, gaps)
                                   - moment(pi, f, q1, powers, gaps)))
    cases += 1
    top0 = moment(pi, f, q0, (n - 1, n - 1), (0.9,))
    top1 = moment(pi, f, q1, (n - 1, n - 1), (0.9,))
    p0 = a * np.linalg.inv(a * np.eye(n) - e0)
    p1 = a * np.linalg.inv(a * np.eye(n) - e1)
    expected_dp = (a * delta / ((a + lam) * (a + lam - delta))
                   * np.outer(w, w))

    # Reconstruction from all mixed two-time moments up to total order2n-2.
    t = 0.9
    powers = np.arange(1, n)
    cpow = np.column_stack([c ** k for k in powers])
    means = cpow.mean(axis=0)
    v = cpow - means
    b = a / (a + d) + d / (a + d) * np.exp(-(a + d) * t)
    bb = expm(q1 * t)[:n, :n]
    measured = cpow.T @ bb @ cpow
    projected = measured - n * b * np.outer(means, means)
    vinv = np.linalg.inv(v.T @ v)
    bb_recovered = v @ vinv @ projected @ vinv @ v.T + b * j

    # E from the exact third time derivative of the reconstructed BB matrix.
    bb3 = np.linalg.matrix_power(q1, 3)[:n, :n]
    projected3 = cpow.T @ bb3 @ cpow - n * (
        -d * (a + d) ** 2) * np.outer(means, means)
    bb3_recovered = (v @ vinv @ projected3 @ vinv @ v.T
                     - d * (a + d) ** 2 * j)
    e_recovered = (bb3_recovered + d * (a + d) ** 2 * np.eye(n)) / (a * d)
    # Direct recovery from the integrated projected mixed-moment matrix.
    u = np.linalg.qr(np.column_stack([np.ones(n), v]))[0][:, 1:]
    zero = np.zeros_like(u)
    lift = np.block([[u, zero], [zero, u]])
    q_reduced = lift.T @ q1 @ lift
    g_reduced = -np.linalg.inv(q_reduced)[:n - 1, :n - 1]
    g_integral = u @ g_reduced @ u.T
    integrated_moments = v.T @ g_integral @ v
    g_recovered = v @ vinv @ integrated_moments @ vinv @ v.T
    p_recovered_direct = np.eye(n) - np.linalg.pinv(g_recovered) / d
    c0 = np.dot(pi, f * f)
    cprime0 = (pi * f) @ q1 @ f
    d_recovered = -cprime0 / c0
    bound_fraction = n * c0 / np.dot(c, c)
    a_recovered = d_recovered * bound_fraction / (1 - bound_fraction)

    assert np.min(e1[~np.eye(n, dtype=bool)]) > 0
    assert low_error < 1e-10, (n, low_error)
    assert abs(top1 - top0) > 1e-10, n
    assert np.max(np.abs(p1 - p0 - expected_dp)) < 1e-10
    assert np.max(np.abs(bb_recovered - bb)) < 1e-8
    assert np.max(np.abs(e_recovered - e1)) < 1e-8
    assert np.max(np.abs(p_recovered_direct - p1)) < 1e-8
    assert abs(a_recovered - a) < 1e-12
    assert abs(d_recovered - d) < 1e-12
    return {
        "n": n,
        "couplings": c.tolist(),
        "hidden_vector": w.tolist(),
        "parameters": {"a": a, "d": d, "lambda": lam, "delta": delta},
        "maximum_vandermonde_orthogonality_error": float(np.max(np.abs(vand @ w))),
        "minimum_offdiagonal_rate": float(np.min(e1[~np.eye(n, dtype=bool)])),
        "subthreshold_maximum_order": 2 * n - 3,
        "subthreshold_moments_checked": cases,
        "maximum_subthreshold_moment_difference": low_error,
        "first_distinguishing_mixed_moment_order": 2 * n - 2,
        "threshold_moment_difference": top1 - top0,
        "maximum_return_probability_change": float(np.max(np.abs(p1 - p0))),
        "P22_before": float(p0[1, 1]),
        "P22_after": float(p1[1, 1]),
        "rank_one_return_change_formula_error": float(np.max(np.abs(p1 - p0 - expected_dp))),
        "bound_transition_reconstruction_error": float(np.max(np.abs(bb_recovered - bb))),
        "encounter_generator_reconstruction_error": float(np.max(np.abs(e_recovered - e1))),
        "direct_integrated_moment_P_reconstruction_error": float(np.max(np.abs(p_recovered_direct - p1))),
        "a_reconstruction_error": float(abs(a_recovered - a)),
        "d_reconstruction_error": float(abs(d_recovered - d)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {
        "scope": "Stationary exact moments; calibrated distinct couplings; reversible homogeneous-rate binding model. No finite-shot or quantum-control inversion is tested.",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "examples": [one_check(c) for c in ([1, -1], [-1, -2, 3],
                                            [-3, -1, 1, 3], [-4, -2, -1, 3, 4],
                                            [-5, -3, -1, 1, 3, 5])],
    }
    target = args.output
    target.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
