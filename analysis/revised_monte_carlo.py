"""Re-run the Section 5 photon-budget Monte Carlo with an empirical organic prior.

The original Monte Carlo (`src_fixed.run_monte_carlo`) samples organic
concentration from a uniform 1-15 wt% prior — which Andrew Rivkin flagged as
unjustifiably broad given Bennu's known heterogeneity. This module replaces
*only* that prior with a `scipy.stats.rv_histogram` built from the empirical
OVIRS wt% C distribution (Recon-A, 1.6 m facets, ~48k samples). Every other
sampled parameter (pulse energy, aperture, integration time, surface
temperature, solar irradiance, environment factor) and the photon-budget
physics are unchanged from `src_fixed.calculate_snr_full`.

Outputs are written side-by-side (uniform vs empirical) to enable a
honest comparison of how the prior affects the headline detection statistics.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Make src_fixed importable from the project root regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src_fixed as photon_budget  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_DISTANCE_M = 680.0  # Orbital B reference distance (Section 5)
DEFAULT_N = 10_000
DEFAULT_SEED = 42

PRIOR_LABEL_UNIFORM = "uniform"
PRIOR_LABEL_EMPIRICAL = "empirical"


@dataclass
class MCSummary:
    label: str
    n: int
    distance_m: float
    snr_median: float
    snr_ci_lo: float
    snr_ci_hi: float
    p_snr_ge_5: float
    p_snr_ge_3: float
    range_median_m: float
    range_ci_lo: float
    range_ci_hi: float
    organic_pct_median: float
    organic_pct_p5: float
    organic_pct_p95: float
    spearman: dict[str, float] = field(default_factory=dict)


def build_empirical_prior(
    wt_c: np.ndarray, n_bins: int = 200
) -> stats.rv_histogram:
    """Build a `scipy.stats.rv_histogram` from the empirical wt% C array.

    The histogram edges span 0 to (max + small ε) so the rv_histogram is
    well-defined. The resulting distribution can be sampled via `.rvs()`.
    """
    wt_c = np.asarray(wt_c)
    edges = np.linspace(0.0, max(float(wt_c.max()) * 1.001, 1.0), n_bins + 1)
    hist, _ = np.histogram(wt_c, bins=edges)
    return stats.rv_histogram((hist, edges))


def _spearman_summary(samples: dict, snr: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, arr in samples.items():
        if np.std(arr) > 0 and np.std(snr) > 0:
            r, _ = stats.spearmanr(arr, snr)
            out[k] = float(r)
        else:
            out[k] = float("nan")
    return out


def _max_range_for_snr(
    organic_pct: float, snr_target: float = 5.0, kwargs: dict | None = None
) -> float:
    """Largest distance at which the photon budget still meets SNR ≥ target."""
    kwargs = kwargs or {}
    for d in np.linspace(5000, 50, 200):
        if photon_budget.calculate_snr(d, organic_pct, **kwargs) >= snr_target:
            return float(d)
    return 50.0


def run_mc(
    organic_sampler,
    label: str,
    distance_m: float = DEFAULT_DISTANCE_M,
    N: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> tuple[MCSummary, pd.DataFrame]:
    """Run the photon-budget MC with a custom organic-concentration sampler.

    Parameters
    ----------
    organic_sampler : callable(rng, n) → np.ndarray
        Returns an array of organic wt% values.
    label : str
        Tag for the prior (e.g. "uniform" or "empirical").
    distance_m, N, seed : numeric
        Standard MC controls.
    """
    rng = np.random.default_rng(seed)
    samples: dict[str, np.ndarray] = {k: np.empty(N) for k in photon_budget.MC_RANGES}
    snr = np.empty(N)
    n_sig = np.empty(N)
    n_bg = np.empty(N)
    rng_max = np.empty(N)

    org_arr = organic_sampler(rng, N)

    for i in range(N):
        params: dict[str, float] = {}
        for k, (lo, hi) in photon_budget.MC_RANGES.items():
            if k == "organic_pct":
                v = float(np.clip(org_arr[i], 0.0, 50.0))
            else:
                v = float(rng.uniform(lo, hi))
            params[k] = v
            samples[k][i] = v
        org = params.pop("organic_pct")
        ef = params.pop("env_factor")
        params["env_factor"] = ef
        result = photon_budget.calculate_snr_full(distance_m, org, **params)
        snr[i] = result["snr"]
        n_sig[i] = result["N_sig"]
        n_bg[i] = result["N_bg"]
        rng_max[i] = _max_range_for_snr(org, kwargs=params)

    df = pd.DataFrame(
        {
            "snr": snr,
            "N_sig": n_sig,
            "N_bg": n_bg,
            "max_range_m": rng_max,
            **samples,
        }
    )

    snr_ci = np.percentile(snr, [2.5, 97.5])
    rng_ci = np.percentile(rng_max, [2.5, 97.5])
    org_p = np.percentile(samples["organic_pct"], [5, 95])

    summary = MCSummary(
        label=label,
        n=N,
        distance_m=distance_m,
        snr_median=float(np.median(snr)),
        snr_ci_lo=float(snr_ci[0]),
        snr_ci_hi=float(snr_ci[1]),
        p_snr_ge_5=float(np.mean(snr >= 5.0)),
        p_snr_ge_3=float(np.mean(snr >= 3.0)),
        range_median_m=float(np.median(rng_max)),
        range_ci_lo=float(rng_ci[0]),
        range_ci_hi=float(rng_ci[1]),
        organic_pct_median=float(np.median(samples["organic_pct"])),
        organic_pct_p5=float(org_p[0]),
        organic_pct_p95=float(org_p[1]),
        spearman=_spearman_summary(samples, snr),
    )
    return summary, df


def uniform_sampler(lo: float = 1.0, hi: float = 15.0):
    def sample(rng, n):
        return rng.uniform(lo, hi, size=n)

    return sample


def empirical_sampler(empirical_dist: stats.rv_histogram):
    def sample(rng, n):
        return empirical_dist.rvs(size=n, random_state=rng)

    return sample


def run_both(
    wt_c_empirical: np.ndarray,
    distance_m: float = DEFAULT_DISTANCE_M,
    N: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> tuple[MCSummary, MCSummary, pd.DataFrame, pd.DataFrame, stats.rv_histogram]:
    """Run both uniform and empirical priors; return summaries and trial frames."""
    log.info("Running Monte Carlo with UNIFORM 1-15%% organic prior...")
    s_unif, df_unif = run_mc(uniform_sampler(), PRIOR_LABEL_UNIFORM, distance_m, N, seed)
    log.info(
        "  uniform: median SNR %.2f, 95%% CI [%.2f, %.2f], P(SNR≥5)=%.1f%%",
        s_unif.snr_median, s_unif.snr_ci_lo, s_unif.snr_ci_hi, 100 * s_unif.p_snr_ge_5,
    )

    log.info("Building empirical wt%%-C prior from %d OVIRS facets...", len(wt_c_empirical))
    emp_dist = build_empirical_prior(wt_c_empirical)

    log.info("Running Monte Carlo with EMPIRICAL OVIRS-derived organic prior...")
    s_emp, df_emp = run_mc(
        empirical_sampler(emp_dist), PRIOR_LABEL_EMPIRICAL, distance_m, N, seed + 1
    )
    log.info(
        "  empirical: median SNR %.2f, 95%% CI [%.2f, %.2f], P(SNR≥5)=%.1f%%",
        s_emp.snr_median, s_emp.snr_ci_lo, s_emp.snr_ci_hi, 100 * s_emp.p_snr_ge_5,
    )
    return s_unif, s_emp, df_unif, df_emp, emp_dist
