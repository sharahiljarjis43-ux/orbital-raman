"""End-to-end and unit tests for the Bennu OVIRS heterogeneity analysis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import (
    calibration,
    classify,
    heterogeneity_stats,
    parse_ovirs,
    revised_monte_carlo,
)

DATA_FITS = Path(__file__).resolve().parent.parent / (
    "Data/l_1600mm_sp_ovirs_reca_bandarea3200to3600nm_allsites_wavc_0000n00000.fits"
)


@pytest.fixture(scope="module")
def ovirs_dataset():
    if not DATA_FITS.exists():
        pytest.skip(f"OVIRS FITS not found at {DATA_FITS}")
    return parse_ovirs.load_band_area_fits(DATA_FITS)


@pytest.fixture(scope="module")
def df_classified(ovirs_dataset):
    df_qc = parse_ovirs.apply_quality_filter(ovirs_dataset.df, min_snr=5.0)
    df_cal = calibration.calibrate_dataframe(df_qc)
    return classify.classify(df_cal)


def test_fits_basic_structure(ovirs_dataset):
    ds = ovirs_dataset
    assert ds.n_facets_total == 786432
    assert ds.n_facets_valid > 40_000
    assert ds.n_facets_valid < ds.n_facets_total
    assert "Recon A" in ds.mission_phase
    for col in ("latitude_deg", "longitude_deg", "band_area", "band_area_sigma"):
        assert col in ds.df.columns


def test_quality_filter_reduces_count(ovirs_dataset):
    df_qc = parse_ovirs.apply_quality_filter(ovirs_dataset.df, min_snr=5.0)
    assert 0 < len(df_qc) <= len(ovirs_dataset.df)
    assert (df_qc["snr_band_area"] >= 5.0).all()


def test_calibration_monotone():
    ba = np.array([0.0, 0.5, 1.0, 2.0])
    wt = calibration.band_area_to_wt_c(ba, alpha=calibration.ALPHA_BEST)
    assert np.all(np.diff(wt) > 0)
    # alpha is exact: wt%(band_area=1) = ALPHA_BEST
    assert wt[2] == pytest.approx(calibration.ALPHA_BEST, abs=1e-9)


def test_calibration_uncertainty_band():
    """Factor-of-2 envelope: ALPHA_HIGH = 2 × ALPHA_BEST = 4 × ALPHA_LOW."""
    ba = np.array([1.0])
    low = calibration.band_area_to_wt_c(ba, alpha=calibration.ALPHA_LOW)
    hi = calibration.band_area_to_wt_c(ba, alpha=calibration.ALPHA_HIGH)
    assert low[0] == pytest.approx(calibration.ALPHA_LOW, abs=1e-9)
    assert hi[0] == pytest.approx(calibration.ALPHA_HIGH, abs=1e-9)
    assert hi[0] / low[0] == pytest.approx(4.0, abs=1e-6)  # factor-of-2 each side
    # Sanity: ALPHA_BEST is the geometric mean of LOW and HIGH (within rounding).
    assert calibration.ALPHA_BEST == pytest.approx(
        np.sqrt(calibration.ALPHA_LOW * calibration.ALPHA_HIGH), rel=0.05
    )


def test_calibration_anchored_to_lauretta(ovirs_dataset):
    """OVIRS-mean wt%-C should reproduce Lauretta 2024 bulk organic C ≈ 4.1 wt%."""
    df_qc = parse_ovirs.apply_quality_filter(ovirs_dataset.df, min_snr=5.0)
    df_cal = calibration.calibrate_dataframe(df_qc)
    mean_wt_c = df_cal["wt_c_pct"].mean()
    # Lauretta 2024: bulk total C 4.5-4.7 wt%, ~90% organic → organic ≈ 4.0-4.2 wt%.
    assert 3.5 <= mean_wt_c <= 4.7, f"OVIRS-mean wt% C = {mean_wt_c:.2f} drifted from Lauretta anchor"


def test_validation_gate_passes(df_classified):
    """Plan-A validation gate: factor-of-2 of Ferrone (2021)."""
    rep = classify.report(df_classified)
    assert rep.gate_passes(), (
        f"Validation FAIL: organic ratio {rep.ratio_organic:.2f}, "
        f"carbonate ratio {rep.ratio_carbonate:.2f}"
    )
    # Tighter check: agreement should be much better than factor of 2.
    assert 0.7 < rep.ratio_organic < 1.5
    assert 0.7 < rep.ratio_carbonate < 1.5


def test_classification_partition(df_classified):
    rep = classify.report(df_classified)
    assert rep.n_organic + rep.n_carbonate + rep.n_weak == rep.n_total
    assert rep.n_weak > rep.n_carbonate > rep.n_organic > 0


def test_heterogeneity_report(df_classified):
    rep = classify.report(df_classified)
    het = heterogeneity_stats.build_report(
        df_classified, n_organic=rep.n_organic, n_total=rep.n_total
    )
    assert het.wt_c_median > 0
    assert het.wt_c_mean > 0
    assert het.wt_c_p95 > het.wt_c_median
    assert 0 < het.f_organic < 0.05
    assert het.n_required[0.95] >= het.n_required[0.50]


def test_morans_i_significant(df_classified):
    out = heterogeneity_stats.morans_i_5deg(df_classified)
    assert out["I"] is not None
    # Bennu organics are spatially clustered — expect positive autocorrelation.
    assert out["I"] > 0
    assert abs(out["z"]) > 2.0  # significant at p < 0.05


def test_empirical_prior_well_defined():
    rng = np.random.default_rng(0)
    wt = rng.gamma(shape=2.0, scale=2.0, size=10_000)
    dist = revised_monte_carlo.build_empirical_prior(wt)
    samples = dist.rvs(size=2_000, random_state=rng)
    # Samples should fall within the empirical support.
    assert (samples >= 0).all()
    assert (samples <= wt.max() * 1.001).all()
    # Mean should agree to a few percent.
    assert abs(samples.mean() - wt.mean()) < 0.2


def test_monte_carlo_short_run(df_classified):
    """Smoke test of the MC runner with a small N."""
    wt_c = df_classified["wt_c_pct"].to_numpy()
    s_unif, s_emp, df_u, df_e, _ = revised_monte_carlo.run_both(
        wt_c, distance_m=680.0, N=200, seed=42,
    )
    assert s_unif.n == 200
    assert s_emp.n == 200
    assert s_unif.snr_median > 0
    assert s_emp.snr_median > 0
    # Empirical prior should give a lower median organic concentration
    # because OVIRS distribution is right-skewed below the uniform-prior mid.
    assert s_emp.organic_pct_median < s_unif.organic_pct_median
    assert "organic_pct" in df_u.columns
