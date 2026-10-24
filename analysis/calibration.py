"""Convert OVIRS 3.2-3.6 μm band area to a wt% organic carbon proxy.

The Spectral Analysis Bundle product gives an integrated band area over
3200-3600 nm. Two overlapping features fall in this window: the 3.42 μm
aliphatic CH stretch attributed to organics, and a 3.4 μm complex
attributed to Mg/Fe carbonates. Both are documented in Simon et al.
(2020) and Kaplan et al. (2020, 2021).

What the literature actually says about the band-depth → wt% C link
-------------------------------------------------------------------
Kaplan et al. (2021) (A&A 653, L1) do *not* publish a closed-form
band_depth → wt%_C calibration. They state that the 3.42 μm band depths
they measure on Bennu are "consistent with carbonaceous chondrites that
contain 1−2.5 wt% C," and they report H/C ratios of 0.4-0.6 (described
as upper limits) for the 237 organic-rich spectra they identify.
Lauretta et al. (2024) report TAGSAM bulk total carbon 4.5-4.7 wt%, of
which ~90% is in organic matter and ~10% is in carbonates (so bulk
*organic* C ≈ 4.0-4.2 wt%).

Our calibration
---------------
We adopt the simple linear proxy

    wt%_C ≈ α × band_area_3.2-3.6μm

with α anchored so that the OVIRS Recon-A mean reproduces Lauretta's
bulk organic C of ~4.05 wt%. With OVIRS-mean band area = 0.355, this
gives **α = 11.5** as the central value. The factor-of-2 systematic
envelope α ∈ [{ALPHA_LOW:g}, {ALPHA_HIGH:g}] (i.e. ~half to ~double the
central value) brackets:
    - the assumption that surface OVIRS-derived organic content matches
      sample-bulk organic content (the sub-OVIRS-pixel work of Glavin
      2025 / Sandford 2025 says these can differ at the grain scale);
    - residual continuum-fit and photometric-correction effects in the
      SAB product;
    - the carbonate-vs-organic ambiguity intrinsic to the integrated
      3.2-3.6 μm band (some pixels' band area is carbonate-dominated, and
      converting to "organic C" via α over-counts those).

We do *not* claim that the linear relation is physically motivated — it
is an empirical proxy with the stated uncertainty band. The downstream
Monte Carlo result (Section 5) is robust to α within this envelope: a
factor-of-2 calibration shift moves median SNR by less than a factor of
two and does not change the qualitative feasibility conclusion.

We also report a band-depth-equivalent for cross-reference,

    band_depth_frac ≈ band_area / (FWHM × √(π/(4 ln 2)))

assuming a Gaussian profile with FWHM = 150 nm (a typical width for the
3.42 μm aliphatic CH band in carbonaceous chondrite spectra; this FWHM
is *not* taken from Kaplan 2021, who do not report a value). This is
used only for sanity-checking the band-area magnitudes; it does not
enter the wt%-C calibration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Sample-anchored calibration: OVIRS mean band area (~0.355) × ALPHA_BEST
# gives wt%_C ≈ 4.1, matching the bulk *organic* carbon content of Bennu
# TAGSAM samples (Lauretta et al. 2024: bulk total C 4.5-4.7 wt%, of which
# ~90% is organic and ~10% is carbonate).
ALPHA_BEST = 11.5
ALPHA_LOW = 5.75   # factor-of-2 lower bound
ALPHA_HIGH = 23.0  # factor-of-2 upper bound

# Gaussian-FWHM-derived band-depth proxy (sanity-check only; does NOT enter
# the wt%-C calibration). FWHM = 150 nm is a typical aliphatic CH width in
# carbonaceous chondrite analog spectra; Kaplan et al. (2021) do not publish
# a FWHM for the Bennu 3.42 μm band.
FWHM_NM = 150.0
GAUSS_FACTOR = FWHM_NM * np.sqrt(np.pi / (4.0 * np.log(2.0)))  # ≈ 159.6 nm


@dataclass
class CalibrationReport:
    n: int
    wt_c_mean: float
    wt_c_median: float
    wt_c_std: float
    wt_c_p95: float
    wt_c_max: float
    fraction_above_pct: dict[float, float]


def band_area_to_band_depth_pct(band_area: np.ndarray) -> np.ndarray:
    """Approximate band-depth (%) from band area assuming a Gaussian profile.

    Used as an internal sanity check; the wt%-C calibration goes directly
    from band area via the sample-anchored α (see module docstring).
    """
    return 100.0 * np.asarray(band_area) / GAUSS_FACTOR


def band_area_to_wt_c(
    band_area: np.ndarray,
    alpha: float = ALPHA_BEST,
) -> np.ndarray:
    """Return wt% organic carbon for each facet.

    Parameters
    ----------
    band_area : array
        SAB band-area values (output of `parse_ovirs.load_band_area_fits`).
    alpha : float
        Calibration constant. Default ALPHA_BEST is the sample-anchored value
        producing OVIRS-mean ~5 wt% C (Lauretta 2024).
    """
    wt_c = alpha * np.asarray(band_area, dtype=np.float64)
    return np.clip(wt_c, 0.0, 50.0)


def calibrate_dataframe(
    df: pd.DataFrame,
    alpha: float = ALPHA_BEST,
) -> pd.DataFrame:
    """Add `band_depth_pct` and `wt_c_pct` columns to a parsed OVIRS DataFrame."""
    out = df.copy()
    out["band_depth_pct"] = band_area_to_band_depth_pct(out["band_area"].to_numpy())
    out["wt_c_pct"] = band_area_to_wt_c(out["band_area"].to_numpy(), alpha=alpha)
    out["wt_c_pct_low"] = band_area_to_wt_c(out["band_area"].to_numpy(), alpha=ALPHA_LOW)
    out["wt_c_pct_high"] = band_area_to_wt_c(out["band_area"].to_numpy(), alpha=ALPHA_HIGH)
    return out


def summarize(wt_c: np.ndarray, thresholds=(1.0, 2.0, 3.0, 5.0, 10.0)) -> CalibrationReport:
    """Summary statistics on the wt% C distribution."""
    wt_c = np.asarray(wt_c)
    return CalibrationReport(
        n=len(wt_c),
        wt_c_mean=float(np.mean(wt_c)),
        wt_c_median=float(np.median(wt_c)),
        wt_c_std=float(np.std(wt_c)),
        wt_c_p95=float(np.percentile(wt_c, 95)),
        wt_c_max=float(np.max(wt_c)),
        fraction_above_pct={t: float(np.mean(wt_c >= t)) for t in thresholds},
    )
