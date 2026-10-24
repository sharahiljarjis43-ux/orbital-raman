"""Validate controlled coherence and recompute saved expected-data contrasts.

Checks are deterministic and use synthetic model parameters. They do not
estimate finite-shot error probabilities or validate a physical sensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm

import forward_model as fm
from profile_likelihood import bernoulli_kl, validate_forward

ROOT = Path(__file__).resolve().parent.parent


def check():
    engine = validate_forward()
    q0, pi = fm.generator(fm.TRUTH, 0)
    q1, _ = fm.generator(fm.TRUTH, 1)
    times = np.r_[0., np.geomspace(1e-3, 20., 40)]
    covariance_error = 0.
    occupancy_coherence_error = 0.
    for t in times:
        covariance_error = max(covariance_error, abs(float(
            (pi * fm.F) @ (expm(q0*t) - expm(q1*t)) @ fm.F)))
        # Occupancy-only coupling sees B versus U, with all partner tags equal.
        occ = np.diag([1., 1., 1., 0., 0., 0.])
        for g in (.1, .5, 1.):
            wa = pi @ expm((q0 - 1j*g*occ)*t) @ np.ones(6)
            wb = pi @ expm((q1 - 1j*g*occ)*t) @ np.ones(6)
            occupancy_coherence_error = max(occupancy_coherence_error, abs(wa-wb))
    assert covariance_error < 1e-12, covariance_error
    assert occupancy_coherence_error < 1e-12, occupancy_coherence_error
    mixed_derivative_delta = float((pi * fm.F**2) @ (
        np.linalg.matrix_power(q1, 3) - np.linalg.matrix_power(q0, 3)) @ fm.F)
    assert abs(mixed_derivative_delta + 76/21) < 1e-12
    mixed_at_one_delta = float((pi * fm.F**2) @ (expm(q1)-expm(q0)) @ fm.F)
    assert abs(mixed_at_one_delta) > .09

    saved = json.loads((ROOT/'results/expected_data_profiles_20261019.json').read_text())
    design_path = ROOT/'results/deterministic-design.json'
    design = json.loads(design_path.read_text())
    assert hashlib.sha256(design_path.read_bytes()).hexdigest() == saved['design_sha256']
    assert hashlib.sha256((ROOT/'scripts/forward_model.py').read_bytes()).hexdigest() == design['forward_model_sha256']
    contrasts = []
    max_kl_discrepancy = 0.
    for record in saved['profiles']:
        rows = fm.make_settings(record['scenario'] == 'synthetic_shared_calibration')
        shots = fm.shot_weights(rows)
        p = fm.predict(record['truth'], record['truth_hypothesis'], rows)
        fit = record['fit']
        q = fm.predict(fit['params'], record['fit_hypothesis'], rows)
        kl = float(np.dot(shots, bernoulli_kl(p, q)))
        discrepancy = abs(kl - fit['kl'])
        max_kl_discrepancy = max(max_kl_discrepancy, discrepancy)
        assert discrepancy < 1e-7, (record['truth_name'], discrepancy)
        contrasts.append(dict(truth=record['truth_name'], scenario=record['scenario'],
                              true_hypothesis=record['truth_hypothesis'],
                              recomputed_expected_kl=kl))
    return dict(status='all checks passed',
                runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                forward_engine=engine,
                historical_design_and_forward_model_hashes_match=True,
                covariance_grid_maximum_error=covariance_error,
                occupancy_only_coherence_grid_maximum_error=float(occupancy_coherence_error),
                nonlinear_classical_moment_third_derivative_delta=mixed_derivative_delta,
                nonlinear_classical_moment_delta_at_lag_one=mixed_at_one_delta,
                saved_expected_kl_maximum_discrepancy=max_kl_discrepancy,
                expected_data_contrasts=contrasts,
                scope='Finite deterministic checks; no global optimizer, finite-shot power, physical or novelty claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('All forward-model, classical-comparator and saved-profile checks passed.')


if __name__ == '__main__':
    main()
