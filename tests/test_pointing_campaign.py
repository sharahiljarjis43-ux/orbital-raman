"""Tests for the spatial-sampling reframe: pointing campaign and K_R bounds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src_fixed as S
from analysis import kr_anchor, pointing_campaign

CLS = Path(__file__).resolve().parent.parent / "results/intermediate/classifications.parquet"


@pytest.fixture(scope="module")
def classified():
    if not CLS.exists():
        pytest.skip(f"classifications not found at {CLS}")
    return pd.read_parquet(CLS)


def test_positions_match_body_radius(classified):
    pos = pointing_campaign.facet_positions_m(classified)
    r = np.linalg.norm(pos, axis=1)
    expected = classified["radius_km"].to_numpy() * 1000.0
    assert np.allclose(r, expected, rtol=1e-9)


def test_correlogram_decays_to_baseline(classified):
    """Lift must be high at short lag and fall toward 1 (no memory) at long lag."""
    organic = (classified["class"].to_numpy() == "organic")
    pos = pointing_campaign.facet_positions_m(classified)
    counts, p, lift = pointing_campaign.correlogram(pos, organic)
    assert lift[0] > 20, "short-lag lift should be strongly positive"
    assert lift[0] > lift[3] > lift[-1], "lift must decrease monotonically across these bins"
    assert lift[-1] < 1.5, "long-lag lift should approach the no-memory baseline"
    dlen = pointing_campaign.decorrelation_length(pointing_campaign.LAG_EDGES_M, lift)
    assert 4.0 <= dlen <= 32.0, f"decorrelation length {dlen} m outside expected 4-32 m"


def test_loci_census_shows_pseudo_replication(classified):
    """The 822 organic facets must collapse to far fewer independent loci."""
    organic = (classified["class"].to_numpy() == "organic")
    pos = pointing_campaign.facet_positions_m(classified)
    n_loci, factor = pointing_campaign.loci_census(pos, organic)
    n_org = int(organic.sum())
    assert all(nl <= n_org for nl in n_loci)
    assert all(a >= b for a, b in zip(n_loci, n_loci[1:])), "loci count must fall as linking grows"
    i5 = pointing_campaign.LINK_DISTANCES_M.index(5.0)
    assert factor[i5] >= 3.0, "expect substantial pseudo-replication at 5 m linking"


def test_campaign_saturates_in_small_patch(classified):
    """Confined to one correlation cell, extra pointings barely help; dispersed, they do.

    With unique (without-replacement) pointings capped at patch size, a 5 m patch
    is fully exhausted, so its detection probability is P(patch holds any organic)
    and does not rise with n at all.
    """
    report = pointing_campaign.analyze(classified, seed=42)
    small = report.campaign_p_detect[5.0]
    assert small[-1] - small[0] < 0.05, "exhausted small patch should not gain with n"
    # A globally dispersed campaign of the same size does far better.
    assert report.campaign_global_p_detect[1] > 2 * small[1], "dispersed must beat confined"


def test_campaign_global_matches_analytic(classified):
    """Globally dispersed pointings behave as independent Bernoulli trials."""
    report = pointing_campaign.analyze(classified, seed=42)
    for sim, ana in zip(report.campaign_global_p_detect, report.campaign_analytic_p_detect):
        assert sim == pytest.approx(ana, abs=0.03)


def test_campaign_is_deterministic(classified):
    a = pointing_campaign.analyze(classified, seed=42)
    b = pointing_campaign.analyze(classified, seed=42)
    assert a.campaign_p_detect[5.0] == b.campaign_p_detect[5.0]


def test_block_bootstrap_ci_is_wider_than_naive_wilson(classified):
    """Clustering must inflate the prevalence CI relative to the independent Wilson."""
    prev = pointing_campaign.prevalence_block_bootstrap(classified, seed=42)
    wlo, whi = prev["wilson_ci"]
    blo, bhi = prev["block_ci"]
    # The block interval must bracket the point estimate and exceed the Wilson width.
    assert blo < prev["f_organic"] < bhi
    assert (bhi - blo) > (whi - wlo)
    assert prev["design_effect"] > 2.0, "pseudo-replication should widen the CI several-fold"
    assert prev["n_blocks"] >= 20, "need enough blocks to bootstrap"


def test_block_bootstrap_is_deterministic(classified):
    a = pointing_campaign.prevalence_block_bootstrap(classified, seed=42)
    b = pointing_campaign.prevalence_block_bootstrap(classified, seed=42)
    assert a["block_ci"] == b["block_ci"]


def test_prevalence_is_robust_to_site_removal(classified):
    """f_org must not be an artifact of any single OSIRIS-REx candidate site."""
    loso = pointing_campaign.leave_one_site_out(classified)
    vals = list(loso.values())
    # Dropping any one site should move f_org by less than half its own value.
    assert max(vals) - min(vals) < 0.5 * loso["all_sites"]


def test_kr_depth_bound_flags_asserted_value_as_optimistic():
    """1e-11 must require a sampling depth near/beyond the 532 nm ceiling."""
    depth = kr_anchor.depth_bound_kr()
    # Range [hi-xs, lo-xs] depths needed to reach 1e-11; even the shallow end is
    # comparable to the tens-of-um sampling-depth ceiling in dark carbon.
    d_hi, d_lo = depth["depth_required_for_baseline_um"]
    assert d_hi > 10.0, "even the optimistic cross-section needs a deep column for 1e-11"
    assert d_lo > d_hi, "smaller cross-section requires a deeper column"
    # A physical 1 um column lands below the asserted value for both cross-sections.
    lo, hi = depth["kr_by_depth_um"][1.0]
    assert hi < kr_anchor.KR_BASELINE_532


def test_standoff_inversion_is_a_real_inversion():
    """The two experiments must give DIFFERENT organic-K_R bounds (not a no-op)."""
    s = kr_anchor.standoff_inversion()
    misra = s["Misra_2012_120m"]["kr_organic_range_532"]
    acosta = s["AcostaMaeda_2016_430m"]["kr_organic_range_532"]
    # A no-op would return the asserted baseline identically for both.
    assert misra != acosta, "different geometries/targets must yield different bounds"
    assert all(v != kr_anchor.KR_BASELINE_532 for v in misra + acosta)
    # The raw envelope is deliberately loose (unpublished SNRs) and may exceed the
    # conservative ceiling; the cross-section literature, not this, sets the top bound.
    # The consistency envelope must bracket both experiments' ranges.
    clo, chi = s["combined_organic_range_532"]
    assert clo <= min(misra + acosta) and chi >= max(misra + acosta)
    # Achieved SNR is carried as an estimate range, not false precision.
    assert s["Misra_2012_120m"]["snr_is_estimate"] is True
    assert len(s["Misra_2012_120m"]["achieved_snr_range"]) == 2
    # Both experiments' inversions are consistent with the central estimate to within
    # an order of magnitude (a consistency check, not a tight bound).
    for rng in (misra, acosta):
        assert rng[0] <= kr_anchor.KR_CENTRAL_532 * 10 and rng[1] >= kr_anchor.KR_CENTRAL_532 * 0.1
    # The literature-tightened range is ~one decade and contains the central estimate.
    tlo, thi = s["literature_tightened_range_532"]
    assert tlo < kr_anchor.KR_CENTRAL_532 < thi
    assert (thi / tlo) < 20, "tightened range should be ~one decade, not two"


def test_kr_detection_boundary_brackets_threshold():
    """Across the plausible K_R range, 375 m detection should span feasible to marginal."""
    kr = kr_anchor.analyze(S.calculate_snr_full)
    b = kr.detection_boundary
    hi = b[f"{kr_anchor.KR_HIGH_532:.0e}"][375]
    lo = b[f"{kr_anchor.KR_LOW_532:.0e}"][375]
    central = b[f"{kr_anchor.KR_CENTRAL_532:.0e}"][375]
    assert hi > 5.0, "at the optimistic K_R, Reconnaissance detects comfortably"
    assert lo < 5.0, "at the pessimistic K_R, Reconnaissance is below threshold"
    assert central > 5.0, "at the central K_R, Reconnaissance still detects"
