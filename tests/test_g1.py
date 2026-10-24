"""Tests for the g⁽¹⁾ Section-7 analysis.

Includes the two off-ramp #2 sanity checks specified in the project plan:
- Coherent-source limit: V̄ → 1, SNR_g¹ → standard intensity SNR
- Maximally-incoherent limit: V̄ → 0, SNR_g¹ → 0
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from analysis import g1_analysis


# --- Sanity checks (off-ramp #2) ----------------------------------------

def test_coherent_source_limit_visibility():
    """τ_c → ∞ (Δν → 0): V̄ → 1."""
    tau_c = g1_analysis.coherence_time_s(1e-30)
    V = g1_analysis.gate_averaged_visibility(100e-12, tau_c)
    assert V == pytest.approx(1.0, abs=1e-9)


def test_coherent_source_limit_snr_equals_intensity():
    """SNR_g¹ → standard intensity SNR in coherent limit."""
    N_signal, N_bg = 1000.0, 1e6
    tau_c = g1_analysis.coherence_time_s(1e-30)
    V = g1_analysis.gate_averaged_visibility(100e-12, tau_c)
    snr_g1 = g1_analysis.snr_g1(N_signal, N_bg, V)
    snr_int = g1_analysis.snr_intensity(N_signal, N_bg)
    assert snr_g1 == pytest.approx(snr_int, rel=1e-9)


def test_incoherent_source_limit_visibility():
    """τ_c → 0 (Δν → ∞): V̄ → 0."""
    tau_c = g1_analysis.coherence_time_s(1e30)
    V = g1_analysis.gate_averaged_visibility(100e-12, tau_c)
    assert V == pytest.approx(0.0, abs=1e-9)


def test_incoherent_source_limit_snr_zero():
    """SNR_g¹ → 0 in incoherent limit."""
    tau_c = g1_analysis.coherence_time_s(1e30)
    V = g1_analysis.gate_averaged_visibility(100e-12, tau_c)
    snr_g1 = g1_analysis.snr_g1(1000.0, 1e6, V)
    assert snr_g1 == pytest.approx(0.0, abs=1e-9)


# --- Math identities ----------------------------------------------------

def test_carbonate_coherence_time_pi_c_dnu():
    """τ_c = 1/(πcΔν); Δν = 3 cm⁻¹ ⇒ τ_c ≈ 3.54 ps."""
    tau_c_ps = g1_analysis.coherence_time_s(3.0) * 1e12
    assert tau_c_ps == pytest.approx(3.54, abs=0.05)


def test_visibility_pointwise_lorentzian():
    """V(τ) = exp(−|τ|/τ_c) for a Lorentzian line."""
    tau_c_s = 3.5e-12
    for tau_ps in [0.5, 1.0, 3.5, 10.0]:
        v = g1_analysis.visibility_pointwise(tau_ps * 1e-12, tau_c_s)
        expected = math.exp(-tau_ps / 3.5)
        assert v == pytest.approx(expected, rel=1e-6)


def test_gate_averaged_formula():
    """V̄ = (τ_c/Δt)(1 − exp(−Δt/τ_c)); spot-check at known operating point."""
    V = g1_analysis.gate_averaged_visibility(100e-12, 3.54e-12)
    expected = (3.54 / 100.0) * (1 - math.exp(-100.0 / 3.54))
    assert V == pytest.approx(expected, rel=1e-4)


# --- Operational regime checks ------------------------------------------

def test_n_per_mode_is_photon_starved():
    """At Recon altitude with 10 s integration, n_per_mode ≪ 1."""
    # N_signal at 375 m from src_fixed photon budget ~ 20,000.
    n = g1_analysis.n_per_mode(20_000, 10.0, 3.5e-12)
    assert n < 1e-6, f"Expected n ≪ 1; got {n:.2e}"


def test_g1_beats_g2_in_photon_starved_regime():
    """In the photon-starved regime, g¹ SNR ≫ g² SNR."""
    N_signal, N_bg, T_int = 20_000.0, 3.1e6, 10.0
    tau_c = g1_analysis.coherence_time_s(3.0)
    V = g1_analysis.gate_averaged_visibility(100e-12, tau_c)
    snr_g1 = g1_analysis.snr_g1(N_signal, N_bg, V)
    snr_g2 = g1_analysis.snr_g2_coincidence(N_signal, N_bg, T_int)
    assert snr_g1 > 100 * snr_g2, (
        f"Expected g¹ ≫ g²; got g¹ = {snr_g1:.3g}, g² = {snr_g2:.3g}"
    )


def test_snr_g1_monotonic_in_altitude():
    """SNR_g¹ should decrease with altitude."""
    res = g1_analysis.compute_mode_result(
        delta_nu_cm=g1_analysis.DELTA_NU_CARBONATE_CM,
        delta_nu_lo_cm=g1_analysis.DELTA_NU_CARBONATE_LO,
        delta_nu_hi_cm=g1_analysis.DELTA_NU_CARBONATE_HI,
        label="Carbonate",
    )
    snrs = [r.snr_g1 for r in res.altitudes]
    assert snrs == sorted(snrs, reverse=True), f"Non-monotonic: {snrs}"


def test_t_req_monotonic_in_altitude():
    """T_req should increase with altitude."""
    res = g1_analysis.compute_mode_result(
        delta_nu_cm=g1_analysis.DELTA_NU_CARBONATE_CM,
        delta_nu_lo_cm=g1_analysis.DELTA_NU_CARBONATE_LO,
        delta_nu_hi_cm=g1_analysis.DELTA_NU_CARBONATE_HI,
        label="Carbonate",
    )
    treqs = [r.t_req_g1_s for r in res.altitudes]
    assert treqs == sorted(treqs), f"Non-monotonic: {treqs}"


def test_gate_sensitivity_visibility_increases_as_gate_narrows():
    """Narrower gate ⇒ higher V̄."""
    sweep = g1_analysis.gate_sensitivity_sweep()
    v_by_gate = {int(k.rstrip("ps")): m["V_bar_central"] for k, m in sweep.items()}
    assert v_by_gate[10] > v_by_gate[30] > v_by_gate[100]


# --- Required deliverables exist ----------------------------------------

RESULTS = Path(__file__).resolve().parent.parent / "results"


def test_g1_deliverables_exist():
    expected = [
        "fig_g1_analysis.png",
        "fig_g1_analysis.pdf",
        "section_7_revised.md",
        "g1_findings.md",
        "numerical_results_g1.json",
    ]
    if not any((RESULTS / name).exists() for name in expected):
        pytest.skip("Generated reports are not bundled; run g1_main.py to check artifacts")
    for name in expected:
        assert (RESULTS / name).exists(), f"Missing deliverable: {name}"
