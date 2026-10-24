"""First-order coherence (g⁽¹⁾) SNR analysis for orbital DUV Raman on Bennu.

This module implements the textbook-standard photon-shot-noise-limited g⁽¹⁾
SNR formula for fringe-visibility detection of spontaneous Raman from a
Lorentzian-broadened mode, and compares it side-by-side with the second-order
g⁽²⁾ photon-coincidence SNR derivation already in Section 7 of the paper.

Theoretical framework
---------------------
For a Lorentzian-broadened Raman line of FWHM Δν (cm⁻¹), the field
autocorrelation is γ⁽¹⁾(τ) = exp(−|τ|/τ_c) where τ_c = 1/(πcΔν) is the
coherence time (Wiener–Khintchine theorem; see Loudon 2000 §3.5–3.6,
Goodman 2015 §3.5).

The fringe visibility V(τ) = |γ⁽¹⁾(τ)|. In the photon-shot-noise-limited
regime (Goodman 2015 §9.5, Mandel & Wolf 1995 §14.6, Loudon 2000 §6.5),

    SNR_g¹ = V · N_signal / √(N_signal + N_bg)

with N_signal and N_bg the total detected signal and background counts
across the integration time (background is incoherent and contributes only
shot noise, not fringe modulation).

When the temporal gate Δt_gate is not much smaller than τ_c, the visibility
is effectively gate-averaged (the gate selects photons spanning a range of
self-coherence delays):

    V̄(Δt_gate, τ_c) = (1/Δt_gate) ∫₀^{Δt_gate} exp(−τ/τ_c) dτ
                    = (τ_c/Δt_gate) (1 − exp(−Δt_gate/τ_c))

Both sanity-check limits are recovered:
  - τ_c → ∞ (coherent source): V̄ → 1 ⇒ SNR_g¹ → standard intensity SNR
  - τ_c → 0 (maximally incoherent): V̄ → 0 ⇒ SNR_g¹ → 0

g⁽¹⁾ vs g⁽²⁾ crossover (Foellmi 2009 A&A 507, 1719; Bojer et al. 2022 NJP 24)
----------------------------------------------------------------------------
g⁽²⁾ SNR ∝ n × √(Δf · T) × |γ|² where n = photons per coherence-time-mode.
g⁽¹⁾ SNR ∝ √n × V. Therefore SNR_g¹/SNR_g² ∝ 1/√n × (1/V), and g⁽¹⁾ wins
whenever n_per_mode = N_signal · τ_c / T_int ≪ 1 — the photon-starved
regime characteristic of orbital Raman returns at hundreds of meters.

References
----------
Goodman, J. W. (2015). Statistical Optics, 2nd ed., Wiley §9.5.
Mandel, L. & Wolf, E. (1995). Optical Coherence & Quantum Optics, CUP §14.6.
Loudon, R. (2000). The Quantum Theory of Light, 3rd ed., OUP §3.5–3.6, §6.5.
Foellmi, C. (2009). A&A 507, 1719, doi:10.1051/0004-6361/200911739.
Bojer, M., et al. (2022). NJP 24, 043026, doi:10.1088/1367-2630/ac5f30.
Dravins, D. (2008). In High Time Resolution Astrophysics (Springer), Ch. 6.
Dravins, D., et al. (2013). Astropart. Phys. 43, 331.

Linewidth calibration sources
-----------------------------
Carbonate ν₁ at 1085 cm⁻¹: Δν = 3 ± 2 cm⁻¹ (Lorentzian, intrinsic).
  - Razzell Hollis et al. 2021, Planet. Space Sci. 209, 105356.
  - Jakubek et al. 2024, Appl. Spectrosc. 78, 993 (SHERLOC slit deconvolution).
  - Akpovo et al. 2022, Opt. Comm. 524, 128480 (Γ ≈ 1.5–2 cm⁻¹ at 532 nm).
  - Calandra et al. 2023, Sci. Rep. 13, 13133.
G band at 1580 cm⁻¹: Δν = 90 ± 15 cm⁻¹ (Bonal 2024, Yabuta 2023; 514–532 nm).
CH-stretch at 2920 cm⁻¹: Δν = 40 ± 15 cm⁻¹ (Quirico 2008; 532 nm).

Hardware sources
----------------
SNSPD timing jitter: Korzh et al. 2020, Nat. Photon. 14, 250 (<3 ps).
Active SNSPD gating: Kong et al. 2023, Opt. Express 31, 1244 (~5 ns).
UV PTR-glass VBG pre-filters: Glebov et al. 2012, Proc. SPIE 8428, 84280C.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# Make src_fixed importable regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src_fixed as photon_budget  # noqa: E402

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Physical constants and pre-resolved literature values (Phase 1).
# --------------------------------------------------------------------------

SPEED_OF_LIGHT_CM_S = 2.998e10  # cm/s (CODATA)

# Carbonate ν₁ at 1085 cm⁻¹ — primary g⁽¹⁾ target.
# Razzell Hollis 2021 + Jakubek 2024 (DUV deconvolution); Akpovo 2022,
# Calandra 2023 (intrinsic Γ at 532 nm); wavelength-independent for
# non-resonant CO₃²⁻ (Loudon §10.3).
DELTA_NU_CARBONATE_CM = 3.0     # cm⁻¹ FWHM (central value)
DELTA_NU_CARBONATE_LO = 1.0     # narrow tail
DELTA_NU_CARBONATE_HI = 5.0     # wide tail
DELTA_NU_CARBONATE_LABEL = "Carbonate ν₁ (1085 cm⁻¹)"

# Aromatic G band at 1580 cm⁻¹ — secondary, qualitative only.
DELTA_NU_G_BAND_CM = 90.0
DELTA_NU_G_BAND_LO = 75.0
DELTA_NU_G_BAND_HI = 105.0
DELTA_NU_G_BAND_LABEL = "Aromatic G band (1580 cm⁻¹)"

# Aliphatic CH-stretch at 2920 cm⁻¹ — tertiary, qualitative only.
DELTA_NU_CH_STRETCH_CM = 40.0
DELTA_NU_CH_STRETCH_LO = 25.0
DELTA_NU_CH_STRETCH_HI = 55.0
DELTA_NU_CH_STRETCH_LABEL = "Aliphatic CH-stretch (2920 cm⁻¹)"

# Hardware (Phase 3).
ETA_TOTAL = 0.045                # combined optical+detector (Section 6 convention)
DELTA_T_GATE_S_DEFAULT = 100e-12 # Δt_gate = 100 ps via SNSPD time-tagging post-selection
DELTA_T_GATE_VARIANTS_S = (10e-12, 30e-12, 100e-12)
SPECTRAL_FILTER_FWHM_CM = 5.0    # cm⁻¹ FWHM volume Bragg grating at 268 nm (Glebov 2012)
G2_DETECTOR_RES_S = 100e-12      # Δτ for g⁽²⁾ coincidence (Section 7 baseline)

# Mission-phase altitudes (m).
ALTITUDES_M = (50.0, 200.0, 375.0, 680.0)
ALTITUDE_LABELS = {
    50.0: "Sample Collection",
    200.0: "Close approach",
    375.0: "Reconnaissance",
    680.0: "Orbital B",
}

DEFAULT_ORGANIC_PCT = 5.0  # carbonate-dominant baseline (cf. §4.4.5)
DEFAULT_INTEGRATION_S = 10.0
TARGET_SNR = 5.0


# --------------------------------------------------------------------------
# Core derivation.
# --------------------------------------------------------------------------

def coherence_time_s(delta_nu_cm: float) -> float:
    """τ_c = 1/(π c Δν) for a Lorentzian-broadened mode.

    Wiener–Khintchine: the field autocorrelation of a Lorentzian power
    spectrum of FWHM Δν is exp(−|τ|/τ_c) with τ_c = 1/(πcΔν).
    See Goodman 2015 §3.5; Loudon 2000 §3.5; Mandel & Wolf 1995 §4.3.
    """
    if delta_nu_cm <= 0:
        return float("inf")
    return 1.0 / (np.pi * SPEED_OF_LIGHT_CM_S * delta_nu_cm)


def visibility_pointwise(tau_s: float, tau_c_s: float) -> float:
    """V(τ) = |γ⁽¹⁾(τ)| = exp(−|τ|/τ_c) for a Lorentzian line."""
    if tau_c_s <= 0:
        return 0.0
    if not np.isfinite(tau_c_s):
        return 1.0
    return float(np.exp(-abs(tau_s) / tau_c_s))


def gate_averaged_visibility(delta_t_gate_s: float, tau_c_s: float) -> float:
    """V̄(Δt_gate, τ_c) = (τ_c/Δt) (1 − exp(−Δt/τ_c)).

    This is the temporal-gate-averaged fringe visibility when a detection
    window of width Δt_gate selects photons spanning a range of self-
    coherence delays. Two limits:
      Δt_gate ≪ τ_c (or τ_c → ∞):  V̄ → 1
      Δt_gate ≫ τ_c (or τ_c → 0):  V̄ → τ_c/Δt_gate → 0
    """
    if not np.isfinite(tau_c_s):  # coherent-source limit
        return 1.0
    if tau_c_s <= 0:               # maximally-incoherent limit
        return 0.0
    if delta_t_gate_s <= 0:
        return 1.0
    x = delta_t_gate_s / tau_c_s
    if x < 1e-9:
        return 1.0 - 0.5 * x
    return float((1.0 / x) * (1.0 - np.exp(-x)))


def snr_g1(N_signal: float, N_bg: float, V_bar: float) -> float:
    """Photon shot-noise-limited g⁽¹⁾ SNR (Goodman 2015 §9.5).

    Signal in the visibility measurement is V·N_signal (background is
    incoherent); shot noise on the difference of fringe-port counts is
    √(N_signal + N_bg).
    """
    denom = np.sqrt(max(N_signal + N_bg, 1e-30))
    return float(V_bar * N_signal / denom)


def snr_intensity(N_signal: float, N_bg: float) -> float:
    """Standard intensity-based SNR for cross-comparison (V = 1 limit)."""
    denom = np.sqrt(max(N_signal + N_bg, 1e-30))
    return float(N_signal / denom)


def snr_g2_coincidence(
    N_signal: float,
    N_bg: float,
    integration_time_s: float,
    detector_res_s: float = G2_DETECTOR_RES_S,
) -> float:
    """g⁽²⁾(0) shot-noise SNR for spontaneous Raman against background.

    Ports the existing Section 7 derivation:
        T_min = 9 / (ρ⁴ · R_total² · Δτ)
    so SNR(T) = √(T/T_min) × 3 (since SNR=3 ⇒ T=T_min for the standard
    3σ threshold; we report SNR = 3 × √(T/T_min) instead, i.e. SNR scales
    as √T as expected for a coincidence count).

    Returns SNR_g² achieved at integration_time_s.
    """
    rho = N_signal / max(N_signal + N_bg, 1e-30)
    R_total = (N_signal + N_bg) / max(integration_time_s, 1e-9)
    if rho <= 0 or R_total <= 0:
        return 0.0
    # T_min from the existing paper's Equation (8): SNR=3 in T_min.
    T_min = 9.0 / (rho**4 * R_total**2 * detector_res_s)
    if T_min <= 0 or not np.isfinite(T_min):
        return 0.0
    return 3.0 * np.sqrt(integration_time_s / T_min)


def integration_time_for_snr(
    snr_at_T0: float, T0: float, target_snr: float = TARGET_SNR
) -> float:
    """T_req scales as (target_snr/snr_at_T0)² × T0 for shot-noise-limited
    counting, since SNR ∝ √T."""
    if snr_at_T0 <= 0:
        return float("inf")
    return T0 * (target_snr / snr_at_T0) ** 2


def n_per_mode(N_signal: float, integration_time_s: float, tau_c_s: float) -> float:
    """Modal photon occupation: photons-per-coherence-time-mode.
    Foellmi 2009 §3 — controls g⁽²⁾ vs g⁽¹⁾ crossover."""
    if not np.isfinite(tau_c_s) or integration_time_s <= 0:
        return float("inf")
    n_modes = integration_time_s / max(tau_c_s, 1e-30)
    return N_signal / n_modes


# --------------------------------------------------------------------------
# Structured outputs.
# --------------------------------------------------------------------------

@dataclass
class AltitudeResult:
    altitude_m: float
    label: str
    organic_pct: float
    integration_s: float
    N_signal: float
    N_bg: float
    snr_intensity: float
    tau_c_ps: float
    V_bar: float
    snr_g1: float
    t_req_g1_s: float
    snr_g2: float
    t_req_g2_s: float
    g1_over_g2_snr: float
    n_per_mode: float


@dataclass
class ModeResult:
    mode_label: str
    delta_nu_cm: float
    delta_nu_lo_cm: float
    delta_nu_hi_cm: float
    tau_c_ps: float
    tau_c_lo_ps: float
    tau_c_hi_ps: float
    delta_t_gate_ps: float
    V_bar_central: float
    V_bar_lo: float
    V_bar_hi: float
    altitudes: list[AltitudeResult] = field(default_factory=list)


def compute_mode_result(
    delta_nu_cm: float,
    delta_nu_lo_cm: float,
    delta_nu_hi_cm: float,
    label: str,
    delta_t_gate_s: float = DELTA_T_GATE_S_DEFAULT,
    altitudes: Sequence[float] = ALTITUDES_M,
    organic_pct: float = DEFAULT_ORGANIC_PCT,
    integration_s: float = DEFAULT_INTEGRATION_S,
) -> ModeResult:
    tau_c = coherence_time_s(delta_nu_cm)
    tau_c_lo = coherence_time_s(delta_nu_hi_cm)  # narrow τ_c ↔ wide Δν
    tau_c_hi = coherence_time_s(delta_nu_lo_cm)
    V_bar = gate_averaged_visibility(delta_t_gate_s, tau_c)
    V_bar_lo = gate_averaged_visibility(delta_t_gate_s, tau_c_lo)
    V_bar_hi = gate_averaged_visibility(delta_t_gate_s, tau_c_hi)

    rows: list[AltitudeResult] = []
    for alt in altitudes:
        budget = photon_budget.calculate_snr_full(
            alt, organic_pct, integration_time_s=integration_s
        )
        N_sig = float(budget["N_sig"])
        N_bg = float(budget["N_bg"])
        snr_int = snr_intensity(N_sig, N_bg)
        snr_g1_val = snr_g1(N_sig, N_bg, V_bar)
        t_req_g1 = integration_time_for_snr(snr_g1_val, integration_s)
        snr_g2_val = snr_g2_coincidence(N_sig, N_bg, integration_s)
        t_req_g2 = integration_time_for_snr(snr_g2_val, integration_s)
        n_pm = n_per_mode(N_sig, integration_s, tau_c)
        ratio = (snr_g1_val / snr_g2_val) if snr_g2_val > 0 else float("inf")
        rows.append(
            AltitudeResult(
                altitude_m=alt,
                label=ALTITUDE_LABELS.get(alt, f"{alt:.0f} m"),
                organic_pct=organic_pct,
                integration_s=integration_s,
                N_signal=N_sig,
                N_bg=N_bg,
                snr_intensity=snr_int,
                tau_c_ps=tau_c * 1e12,
                V_bar=V_bar,
                snr_g1=snr_g1_val,
                t_req_g1_s=t_req_g1,
                snr_g2=snr_g2_val,
                t_req_g2_s=t_req_g2,
                g1_over_g2_snr=ratio,
                n_per_mode=n_pm,
            )
        )

    return ModeResult(
        mode_label=label,
        delta_nu_cm=delta_nu_cm,
        delta_nu_lo_cm=delta_nu_lo_cm,
        delta_nu_hi_cm=delta_nu_hi_cm,
        tau_c_ps=tau_c * 1e12,
        tau_c_lo_ps=tau_c_lo * 1e12,
        tau_c_hi_ps=tau_c_hi * 1e12,
        delta_t_gate_ps=delta_t_gate_s * 1e12,
        V_bar_central=V_bar,
        V_bar_lo=V_bar_lo,
        V_bar_hi=V_bar_hi,
        altitudes=rows,
    )


def gate_sensitivity_sweep(
    delta_nu_cm: float = DELTA_NU_CARBONATE_CM,
    altitudes: Sequence[float] = ALTITUDES_M,
    organic_pct: float = DEFAULT_ORGANIC_PCT,
    integration_s: float = DEFAULT_INTEGRATION_S,
    gates_s: Sequence[float] = DELTA_T_GATE_VARIANTS_S,
) -> dict[str, dict]:
    """Sweep Δt_gate ∈ {10, 30, 100} ps and report SNR_g¹ at each altitude."""
    out: dict[str, dict] = {}
    for g in gates_s:
        key = f"{int(g * 1e12)}ps"
        m = compute_mode_result(
            delta_nu_cm=delta_nu_cm,
            delta_nu_lo_cm=DELTA_NU_CARBONATE_LO,
            delta_nu_hi_cm=DELTA_NU_CARBONATE_HI,
            label=DELTA_NU_CARBONATE_LABEL,
            delta_t_gate_s=g,
            altitudes=altitudes,
            organic_pct=organic_pct,
            integration_s=integration_s,
        )
        out[key] = asdict(m)
    return out
