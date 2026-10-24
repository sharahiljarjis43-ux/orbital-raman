"""Frozen, deterministic nuisance-profile evaluation for two fixed kinetic shapes.

Run with system Python 3 + NumPy/SciPy. Exposes profile_fit for finite-shot use:
profile_fit(empirical_fractions, totals, hypothesis, settings, ...).
It minimizes exact binomial deviance; empirical fractions may come from counts.
Finite samples must evaluate/calibrate a test independently of design selection.
"""

import argparse
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import platform
import time
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.special import xlogy

import forward_model as fm

HERE = Path(__file__).resolve().parent
SEED = 20261019
N_STARTS = 8
MAX_NFEV = 300
TRUTHS = {
    "central": fm.TRUTH.copy(),
    "offcenter_rates": fm.TRUTH | dict(a=.7, d=1.4, lam=1.6),
}
SCENARIOS = {
    "known_sensor_limit": dict(calibration=False, free_names=fm.KINETIC_NAMES),
    "science_only_free_sensor": dict(calibration=False, free_names=fm.PARAM_NAMES),
    "synthetic_shared_calibration": dict(calibration=True, free_names=fm.PARAM_NAMES),
}


def bernoulli_kl(p, q):
    p, q = np.asarray(p), np.clip(np.asarray(q), 1e-13, 1-1e-13)
    # Exact expression, with a local series to avoid cancellation close to p=q.
    raw = xlogy(p, p/q) + xlogy(1-p, (1-p)/(1-q))
    delta = q-p
    small = (np.abs(delta) < 1e-5) & (p > 1e-5) & (p < 1-1e-5)
    v = p*(1-p)
    safe_v = np.maximum(v, 1e-20)
    approx = delta**2/(2*safe_v) + delta**3*(2*p-1)/(3*safe_v**2)
    # Fourth term controls the relative error of the local expansion.
    approx += delta**4 * (1/np.maximum(p, 1e-10)**3
                           + 1/np.maximum(1-p, 1e-10)**3)/4
    return np.maximum(np.where(small, approx, raw), 0.)


def profile_fit(target_probs, shots, hypothesis, settings,
                free_names=fm.PARAM_NAMES, fixed=None, n_starts=N_STARTS,
                seed=SEED, initial_params=None, max_nfev=MAX_NFEV):
    """Find a local-profile minimum from reproducible independent starts.

    target_probs: observed counts / shots, or noiseless true probabilities.
    shots: total trials per condition (fractional expected allocation permitted
      for deterministic KL; use integer totals for a finite-shot experiment).
    fixed: full parameter dictionary giving fixed values and the first start.
    Returns best params, all-start diagnostics and KL/deviance. Multiple starts
    are a numerical safeguard, not a proof of the global profile minimum.
    """
    start_time = time.perf_counter()
    target_probs, shots = np.asarray(target_probs), np.asarray(shots)
    names = tuple(free_names)
    fixed = fm.TRUTH.copy() if fixed is None else fm.params_dict(fixed)
    bounds = fm.internal_bounds(names)
    rng = np.random.default_rng(seed)
    starts = [fm.to_internal(initial_params or fixed, names)]
    starts += [rng.uniform(bounds[0], bounds[1]) for _ in range(n_starts-1)]
    results = []
    def residual(x):
        q = fm.predict(fm.from_internal(x, names, fixed), hypothesis, settings)
        return np.sign(q-target_probs)*np.sqrt(2*shots*bernoulli_kl(target_probs, q))
    for x in starts:
        ts = time.perf_counter()
        fit = least_squares(residual, np.clip(x, bounds[0]+1e-10, bounds[1]-1e-10),
                            bounds=bounds, x_scale="jac", max_nfev=max_nfev,
                            ftol=1e-10, xtol=1e-10, gtol=1e-8)
        params = fm.from_internal(fit.x, names, fixed)
        q = fm.predict(params, hypothesis, settings)
        kl = float(np.dot(shots, bernoulli_kl(target_probs, q)))
        near = [name for j, name in enumerate(names)
                if min(fit.x[j]-bounds[0][j], bounds[1][j]-fit.x[j])
                < 1e-5 * (bounds[1][j]-bounds[0][j])]
        results.append(dict(params=params, kl=kl, deviance=2*kl,
                            success=bool(fit.success), status=int(fit.status),
                            message=fit.message, nfev=int(fit.nfev),
                            njev=int(fit.njev or 0), optimality=float(fit.optimality),
                            active_bounds=near, elapsed_seconds=time.perf_counter()-ts,
                            rms_probability_error=float(np.sqrt(np.mean((q-target_probs)**2))),
                            max_probability_error=float(np.max(np.abs(q-target_probs)))))
    best = min(results, key=lambda r: r["kl"])
    best = dict(best)
    best.update(all_starts=results, starts=n_starts,
                converged_starts=sum(r["success"] for r in results),
                near_best_starts=sum(r["kl"]-best["kl"] < 1e-4 for r in results),
                total_elapsed_seconds=time.perf_counter()-start_time,
                free_names=list(names), hypothesis=int(hypothesis))
    return best


def configuration():
    rows = fm.make_settings()
    return dict(
        status="Frozen design before deterministic cross-model profiling",
        seed=SEED, n_starts=N_STARTS, max_nfev=MAX_NFEV,
        parameter_order=list(fm.PARAM_NAMES), assumed_bounds=fm.BOUNDS,
        hypothetical_truths=TRUTHS, scenarios=SCENARIOS,
        sensing_shots=80000, calibration_shots=20000,
        settings=fm.settings_schema(rows),
        science_rows=sum(s.kind=="science" for s in rows),
        calibration_rows=sum(s.kind=="calibration" for s in rows),
        allocation="Uniform within science (80k) and calibration (20k), no control optimization",
        omitted_time="T=0.25 is omitted; all retained individual segments >=0.05",
        calibration_assumption=("Known deterministic detuning fields and binding-dependent tagged "
            "fields share the same gamma, offset, intrinsic decay and effective readout. "
            "These synthetic calibration experiments and bounds are assumed scenarios, "
            "not measured apparatus calibration."),
        scope=("Two predeclared truth points, both hypothesis directions. This is not a "
            "minimax or uniform guarantee over the nuisance set."),
        forward_model_sha256=hashlib.sha256((HERE/"forward_model.py").read_bytes()).hexdigest(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        validation=dict(random_parameters=12, tolerance=2e-10,
                        known_ramsey_original=[.1259196331736515,.31128449128344826],
                        known_ramsey_alias=[.11480594151524241,.2570104554652235]))


def validate_forward():
    rows = fm.make_settings()
    rng = np.random.default_rng(SEED+999)
    lo, hi = fm.internal_bounds()
    max_error = 0.
    for _ in range(12):
        params = fm.from_internal(rng.uniform(lo, hi))
        for h in (0, 1):
            a = fm.coherences(params, h, rows)
            b = fm.coherences(params, h, rows, method="expm")
            max_error = max(max_error, float(np.max(np.abs(a-b))))
    assert max_error < 2e-10, max_error
    single = [fm.Setting("known", "science", "Ramsey", 4., .5, 0.)]
    expected = [.1259196331736515+.31128449128344826j,
                .11480594151524241+.2570104554652235j]
    known_errors = [float(abs(fm.coherences(fm.TRUTH,h,single)[0]-expected[h]))
                    for h in (0,1)]
    assert max(known_errors) < 2e-10
    return dict(max_eig_vs_scipy_expm_error=max_error, known_ramsey_errors=known_errors,
                parameter_draws=12, hypotheses=2, settings_per_draw=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh directory for deterministic design and results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0=time.perf_counter()
    config=configuration()
    frozen=args.output_dir/"deterministic-design.json"
    encoded=json.dumps(config,indent=2,sort_keys=True)+"\n"
    if frozen.exists() and frozen.read_text()!=encoded:
        raise RuntimeError("Existing frozen design differs; do not overwrite after results")
    frozen.write_text(encoded)
    validation=validate_forward()
    result=dict(design_file=frozen.name,
                design_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                status="running", validation=validation, profiles=[])
    outfile=args.output_dir/"deterministic-results.json"
    outfile.write_text(json.dumps(result,indent=2)+"\n")
    for truth_idx,(truth_name,truth) in enumerate(TRUTHS.items()):
        for scenario_idx,(scenario,definition) in enumerate(SCENARIOS.items()):
            rows=fm.make_settings(include_calibration=definition["calibration"])
            weights=fm.shot_weights(rows)
            for htrue in (0,1):
                target=fm.predict(truth,htrue,rows)
                hfit=1-htrue
                fit=profile_fit(target,weights,hfit,rows,
                    free_names=definition["free_names"],fixed=truth,
                    seed=SEED+1000*truth_idx+100*scenario_idx+10*htrue)
                q=fm.predict(fit["params"],hfit,rows)
                sense=np.array([s.kind=="science" for s in rows])
                sense_kl=float(np.dot(weights[sense],bernoulli_kl(target[sense],q[sense])))
                cal_kl=float(np.dot(weights[~sense],bernoulli_kl(target[~sense],q[~sense])))
                record=dict(truth_name=truth_name,truth=truth,scenario=scenario,
                            truth_hypothesis=htrue,fit_hypothesis=hfit,
                            sensing_kl=sense_kl,calibration_kl=cal_kl,fit=fit)
                result["profiles"].append(record)
                result["elapsed_seconds"]=time.perf_counter()-t0
                outfile.write_text(json.dumps(result,indent=2)+"\n")
                print(json.dumps(dict(truth=truth_name,scenario=scenario,htrue=htrue,
                    kl=fit["kl"],rms=fit["rms_probability_error"],
                    converged=fit["converged_starts"],near_best=fit["near_best_starts"],
                    bounds=fit["active_bounds"],seconds=fit["total_elapsed_seconds"])),flush=True)
    result.update(status="complete",elapsed_seconds=time.perf_counter()-t0,
                  interpretation=("KL is the expected log likelihood ratio against the particular "
                    "best-found alternative at the stated total allocation. It is neither a "
                    "finite-shot power result nor a certified global minimum. Bounds and "
                    "calibrations are hypothetical; no physical inference is claimed."))
    outfile.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
