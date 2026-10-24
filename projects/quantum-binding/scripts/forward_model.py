"""Exact six-state controlled-coherence model and declared synthetic calibration.

Public API
----------
``settings = make_settings(include_calibration=True)`` returns ``Setting`` rows.
``predict(params, hypothesis, settings)`` returns their Bernoulli probabilities.
``params`` is a mapping, or a vector in PARAM_NAMES order. All values use physical
coordinates (not logs). Hypothesis 0/original uses E0; 1/alias uses E1.
``coherences(params, hypothesis, settings)`` returns ideal complex coherence.

Science rows use the proposed binding-dependent detuning f=(-1,-2,3,0,0,0).
Calibration rows have known deterministic f=0,+1,-1, no stochastic environment,
or use zero-duration opposite-phase contrast anchors. These are ASSUMED ideal
synthetic calibration experiments, not measurements from an existing apparatus.

Readout is p=b+(C/2)*exp(-Gamma*T)*Re[exp(i*alpha-i*delta*A)*W], where
A=int y(t)dt. C combines visibility and readout attenuation; separate SPAM
attenuation and visibility are not identifiable and are deliberately not fitted.
"""

from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
import numpy as np
from scipy.linalg import expm

PARAM_NAMES = ("a", "d", "lam", "gamma", "delta", "C", "Gamma", "b")
LOG_NAMES = frozenset(("a", "d", "lam", "gamma"))
TRUTH = dict(a=1., d=1., lam=1., gamma=1., delta=.02, C=.9, Gamma=.03, b=.5)
BOUNDS = dict(a=(.5, 2.), d=(.5, 2.), lam=(.5, 2.), gamma=(.8, 1.2),
              delta=(-.1, .1), C=(.6, .94), Gamma=(0., .1), b=(.48, .52))
SENSOR_NAMES = ("gamma", "delta", "C", "Gamma", "b")
KINETIC_NAMES = ("a", "d", "lam")
TIMES = (.5, 1., 2., 4., 8.)
GAINS = (.1, .5, 1.)
MIN_SEGMENT = .05
PULSES = dict(Ramsey=(), Hahn=(.5,), CPMG2=(.25, .75),
              CPMG4=(.125, .375, .625, .875),
              asym2_02_06=(.2, .6), asym2_03_08=(.3, .8))
E0 = np.array([[-1.05, 1., .05], [1., -2., 1.], [.05, 1., -1.05]])
_w = np.array([5., -4., -1.])
_R = np.eye(3) - np.outer(_w, _w) / 21.
E1 = _R @ E0 @ _R.T
F = np.array([-1., -2., 3., 0., 0., 0.])
D = np.diag(F)


@dataclass(frozen=True)
class Setting:
    """One independent readout condition; alpha is in radians.

    kind: science or calibration. known_f is used only for calibration.
    Pulse fractions specify sign changes beginning with y=+1.
    g is the prescribed gain before unknown common multiplicative gamma.
    """
    label: str
    kind: str
    sequence: str
    T: float
    g: float
    alpha: float
    pulse_fractions: tuple[float, ...] = ()
    known_f: float = 0.

    @property
    def segments(self):
        knots = (0.,) + tuple(self.pulse_fractions) + (1.,)
        return tuple((self.T * (knots[i+1] - knots[i]), 1 if i % 2 == 0 else -1)
                     for i in range(len(knots)-1))

    @property
    def area(self):
        return sum(dt * sign for dt, sign in self.segments)


def params_dict(params):
    if isinstance(params, Mapping):
        return {name: float(params[name]) for name in PARAM_NAMES}
    arr = np.asarray(params, dtype=float)
    if arr.shape != (len(PARAM_NAMES),):
        raise ValueError("Parameter vector must follow PARAM_NAMES and have length 8")
    return dict(zip(PARAM_NAMES, arr.tolist()))


def make_settings(include_calibration=True):
    rows = []
    for sequence, pulses in PULSES.items():
        for T in TIMES:
            for g in GAINS:
                for alpha, axis in ((0., "x"), (np.pi / 2, "y")):
                    row = Setting(f"science/{sequence}/T{T:g}/g{g:g}/{axis}",
                                  "science", sequence, T, g, alpha, pulses)
                    if min(dt for dt, _ in row.segments) + 1e-14 < MIN_SEGMENT:
                        continue
                    rows.append(row)
    if include_calibration:
        for alpha, axis in ((0., "plus"), (np.pi, "minus")):
            rows.append(Setting(f"calibration/anchor/{axis}", "calibration",
                                "Ramsey", 0., 0., alpha))
        # Thirty-two calibration rows plus two anchors: Ramsey and echo
        # controls at T=1,2,4,8 and deterministic f=0,+1,-1. The f=0 Hahn
        # duplicates across signs are omitted; known signed fields calibrate
        # gamma separately from offset. None are a binding-history label.
        # Assumption: known fields share the SAME gain/sensor transduction as
        # the stochastic tagged-binding field. No apparatus establishes this.
        for sequence in ("Ramsey", "Hahn"):
            for T in (1., 2., 4., 8.):
                known_values = (0., 1., -1.) if sequence == "Ramsey" else (0.,)
                for known_f in known_values:
                    for alpha, axis in ((0., "x"), (np.pi / 2, "y")):
                        rows.append(Setting(
                            f"calibration/{sequence}/T{T:g}/f{known_f:g}/{axis}",
                            "calibration", sequence, T, 1., alpha,
                            PULSES[sequence], known_f))
    return rows


def generator(params, hypothesis):
    p = params_dict(params)
    if hypothesis in (0, "original", "E0", "H0"):
        E = E0
    elif hypothesis in (1, "alias", "E1", "H1"):
        E = E1
    else:
        raise ValueError("hypothesis must be 0/original or 1/alias")
    a, d, lam = p["a"], p["d"], p["lam"]
    eye = np.eye(3)
    Q = np.block([[-d*eye, d*eye], [a*eye, lam*E-a*eye]])
    pi = np.r_[np.full(3, a/(3*(a+d))), np.full(3, d/(3*(a+d)))]
    return Q, pi


def coherences(params, hypothesis, settings, method="eig"):
    p = params_dict(params)
    Q, pi = generator(p, hypothesis)
    out = np.empty(len(settings), dtype=complex)
    # Shared diagonalizations and segment propagators across both quadratures.
    eig_cache, prop_cache, control_cache = {}, {}, {}
    for idx, s in enumerate(settings):
        if s.kind == "calibration":
            out[idx] = np.exp(-1j*p["gamma"]*s.g*s.known_f*s.area)
            continue
        key = (s.sequence, s.T, s.g, s.pulse_fractions)
        if key in control_cache:
            out[idx] = control_cache[key]
            continue
        row = pi.astype(complex)
        for dt, sign in s.segments:
            # Minus-sign matrices are conjugates; only diagonalize plus.
            pk = (s.g, round(dt, 14))
            if pk not in prop_cache:
                A = Q - 1j*p["gamma"]*s.g*D
                if method == "expm":
                    P = expm(A*dt)
                elif method == "eig":
                    if s.g not in eig_cache:
                        values, vecs = np.linalg.eig(A)
                        inv = np.linalg.inv(vecs)
                        eig_cache[s.g] = (values, vecs, inv, np.linalg.cond(vecs))
                    values, vecs, inv, cond = eig_cache[s.g]
                    P = ((vecs*np.exp(values*dt)) @ inv
                         if cond < 1e10 else expm(A*dt))
                else:
                    raise ValueError("method must be eig or expm")
                prop_cache[pk] = P
            P = prop_cache[pk]
            row = row @ (P if sign == 1 else P.conj())
        W = row.sum()
        control_cache[key] = W
        out[idx] = W
    return out


def predict(params, hypothesis, settings):
    p = params_dict(params)
    W = coherences(p, hypothesis, settings)
    times = np.array([s.T for s in settings])
    area = np.array([s.area for s in settings])
    alpha = np.array([s.alpha for s in settings])
    probs = p["b"] + .5*p["C"]*np.exp(-p["Gamma"]*times)*np.real(
        np.exp(1j*alpha-1j*p["delta"]*area)*W)
    if np.any(probs < -1e-10) or np.any(probs > 1+1e-10):
        raise ValueError("Nonphysical probabilities; check readout parameter bounds")
    return np.clip(probs, 1e-12, 1-1e-12)


def shot_weights(settings, sensing=80000, calibration=20000):
    ns = sum(s.kind == "science" for s in settings)
    nc = sum(s.kind == "calibration" for s in settings)
    return np.array([sensing/ns if s.kind == "science" else calibration/nc
                     for s in settings], dtype=float)


def to_internal(params, names=PARAM_NAMES):
    p = params_dict(params)
    return np.array([np.log(p[name]) if name in LOG_NAMES else p[name]
                     for name in names])


def from_internal(x, names=PARAM_NAMES, fixed=None):
    out = TRUTH.copy() if fixed is None else params_dict(fixed)
    for name, value in zip(names, x):
        out[name] = float(np.exp(value) if name in LOG_NAMES else value)
    return out


def internal_bounds(names=PARAM_NAMES):
    return tuple(np.array([np.log(BOUNDS[n][j]) if n in LOG_NAMES else BOUNDS[n][j]
                           for n in names]) for j in (0, 1))


def settings_schema(settings):
    return [asdict(s) for s in settings]
