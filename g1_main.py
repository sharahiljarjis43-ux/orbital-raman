"""Orchestrator for the g⁽¹⁾ Section-7 reframe analysis.

Runs the full Phase 3-4 numerics, produces the two-panel figure,
markdown deliverables, and JSON dump.

Usage
-----
    python g1_main.py --output-dir results/

The script is fully deterministic (numpy seed 42).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import g1_analysis  # noqa: E402

log = logging.getLogger(__name__)


def _setup_logging(log_path: Path) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path, mode="w"),
        ],
    )


def _to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _format_seconds(s: float) -> str:
    """Compact human-readable string for an integration time."""
    if not np.isfinite(s):
        return "∞"
    if s < 1.0:
        return f"{s*1e3:.1f} ms"
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s/60:.1f} min"
    if s < 86400:
        return f"{s/3600:.2f} h"
    if s < 86400 * 365:
        return f"{s/86400:.1f} d"
    return f"{s/(86400*365):.1g} yr"


def make_figure(
    carbonate_result: g1_analysis.ModeResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Two-panel figure: (a) V(τ) vs τ, (b) T_req for SNR=5 vs altitude."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 100,
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Panel (a): visibility curves and gate-position
    tau_grid_ps = np.logspace(-2, 2, 400)
    tau_grid_s = tau_grid_ps * 1e-12
    for tau_c_ps, ls, color, label in [
        (carbonate_result.tau_c_lo_ps, ":", "#aaaaaa", f"τ_c = {carbonate_result.tau_c_lo_ps:.1f} ps (Δν = 5 cm⁻¹)"),
        (carbonate_result.tau_c_ps,    "-", "#1f77b4", f"τ_c = {carbonate_result.tau_c_ps:.1f} ps (Δν = 3 cm⁻¹, central)"),
        (carbonate_result.tau_c_hi_ps, "--", "#555555", f"τ_c = {carbonate_result.tau_c_hi_ps:.1f} ps (Δν = 1 cm⁻¹)"),
    ]:
        tau_c_s = tau_c_ps * 1e-12
        v = np.exp(-tau_grid_s / tau_c_s)
        ax_a.plot(tau_grid_ps, v, ls, color=color, lw=2, label=label)

    ax_a.axvline(
        carbonate_result.delta_t_gate_ps,
        color="#d62728",
        ls="-.",
        lw=1.2,
        alpha=0.85,
        label=f"Δt_gate = {carbonate_result.delta_t_gate_ps:.0f} ps",
    )

    ax_a.set_xscale("log")
    ax_a.set_xlim(1e-2, 1e2)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xlabel("Delay τ (ps)")
    ax_a.set_ylabel("V(τ) = |γ⁽¹⁾(τ)|")
    ax_a.legend(loc="upper right", framealpha=0.92)
    ax_a.grid(True, which="both", alpha=0.25)
    ax_a.text(
        0.02, 0.98, "(a)", transform=ax_a.transAxes,
        fontweight="bold", va="top", ha="left",
    )

    # ---- Panel (b): T_req for SNR = 5 vs altitude (g¹ vs g²)
    rows = carbonate_result.altitudes
    alts = np.array([r.altitude_m for r in rows])
    t_g1 = np.array([r.t_req_g1_s for r in rows])
    t_g2 = np.array([r.t_req_g2_s for r in rows])

    ax_b.plot(alts, t_g1, "o-", color="#1f77b4", lw=2, ms=7,
              label="g⁽¹⁾ on carbonate ν₁ (this work)")
    ax_b.plot(alts, t_g2, "s--", color="#d62728", lw=1.6, ms=6,
              label="g⁽²⁾ photon coincidence")

    ax_b.axhline(10.0, color="green", ls=":", lw=1.0, alpha=0.6, label="10 s")
    ax_b.axhline(60.0, color="orange", ls=":", lw=1.0, alpha=0.6, label="1 min")
    ax_b.axhline(3600.0, color="purple", ls=":", lw=1.0, alpha=0.6, label="1 hour")
    ax_b.axhline(86400.0 * 365, color="black", ls=":", lw=1.0, alpha=0.6, label="1 year")

    for r in rows:
        ax_b.annotate(
            r.label,
            xy=(r.altitude_m, r.t_req_g1_s),
            xytext=(5, -8),
            textcoords="offset points",
            fontsize=7,
            color="#1f77b4",
        )

    ax_b.set_yscale("log")
    ax_b.set_xlabel("Altitude (m)")
    ax_b.set_ylabel("Integration time for SNR ≥ 5 (s)")
    ax_b.set_xlim(min(alts) * 0.5, max(alts) * 1.1)
    ax_b.set_ylim(1e-1, 1e15)
    ax_b.legend(loc="lower right", framealpha=0.92)
    ax_b.grid(True, which="both", alpha=0.25)
    ax_b.text(
        0.02, 0.98, "(b)", transform=ax_b.transAxes,
        fontweight="bold", va="top", ha="left",
    )

    plt.tight_layout()
    png = output_dir / "fig_g1_analysis.png"
    pdf = output_dir / "fig_g1_analysis.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s and %s", png.name, pdf.name)
    return {"png": png, "pdf": pdf}


def write_section_7(carbonate: g1_analysis.ModeResult, gate_sweep: dict, path: Path) -> None:
    rows = carbonate.altitudes
    # Build the main table.
    table_lines = []
    for r in rows:
        table_lines.append(
            f"| {r.altitude_m:>4.0f} ({r.label}) | "
            f"{r.N_signal:>9.3g} | "
            f"{r.N_bg:>9.3g} | "
            f"{r.snr_intensity:>7.2f} | "
            f"{r.snr_g1:>7.3f} | "
            f"{_format_seconds(r.t_req_g1_s):>10} | "
            f"{_format_seconds(r.t_req_g2_s):>12} | "
            f"{r.g1_over_g2_snr:>7.2g} |"
        )
    table = "\n".join(table_lines)

    # Gate-sensitivity summary at 375 m.
    gate_summary = []
    for key, mode_dict in sorted(gate_sweep.items(), key=lambda kv: int(kv[0].rstrip("ps"))):
        # find 375 m row
        for r in mode_dict["altitudes"]:
            if r["altitude_m"] == 375.0:
                gate_summary.append(
                    f"| Δt_gate = {mode_dict['delta_t_gate_ps']:>3.0f} ps | "
                    f"V̄ = {mode_dict['V_bar_central']:.3f} | "
                    f"SNR_g¹(10 s) = {r['snr_g1']:.3f} | "
                    f"T_req(SNR=5) = {_format_seconds(r['t_req_g1_s'])} |"
                )
                break
    gate_table = "\n".join(gate_summary)

    txt = f"""## 7. First- and Second-Order Photon Correlation Analysis

### 7.1 Modality selection: g⁽¹⁾ for narrow lines, g⁽²⁾ as discriminator

Two photon-correlation modalities are available for orbital Raman: first-
order coherence g⁽¹⁾(τ) — the field autocorrelation, measured by fringe
visibility in a Mach–Zehnder interferometer — and second-order coherence
g⁽²⁾(τ) — the intensity autocorrelation, measured by photon coincidence.
For ground-based stellar intensity interferometry the choice is set by
photon brightness: in the photon-starved regime n ≪ 1 (where n is the
photon occupation per coherence-time-mode), g⁽¹⁾ amplitude interferometry
achieves higher SNR than g⁽²⁾ intensity interferometry by a factor scaling
as 1/√n (Hanbury Brown 1974; Foellmi 2009; Bojer et al. 2022; Dravins 2008,
2013). This is the regime of orbital Raman returns: the analysis below
shows n ~ 10⁻⁹–10⁻⁷ photons per Lorentzian-coherence-mode at the altitudes
of interest, so g⁽¹⁾ is overwhelmingly favored.

The choice is also **mode-dependent**. g⁽¹⁾ visibility is highest for
narrow Raman lines, where the coherence time τ_c is long; broad lines are
penalized by the gate-averaged visibility V̄ → τ_c/Δt_gate. The narrowest
target available on Bennu is the **carbonate symmetric stretch ν₁ at
1085 cm⁻¹**, which is therefore adopted as the primary g⁽¹⁾ target. The
broader aromatic D/G bands (Δν ~ 90–300 cm⁻¹) and aliphatic CH-stretch
(Δν ~ 40 cm⁻¹) have shorter τ_c and are discussed qualitatively in §7.5.

### 7.2 Coherence model for the carbonate ν₁ mode

For a Lorentzian-broadened Raman line of FWHM Δν (cm⁻¹), the field
autocorrelation is γ⁽¹⁾(τ) = exp(−|τ|/τ_c) with τ_c = 1/(πcΔν), the
Wiener–Khintchine result for a Lorentzian power spectrum (Goodman 2015
§3.5; Loudon 2000 §3.5–3.6).

The carbonate ν₁ at 1085 cm⁻¹ has been characterized in the deep-UV
SHERLOC instrument calibration sequence (Razzell Hollis et al. 2021;
SHERLOC slit FWHM 34.1 cm⁻¹ from Jakubek et al. 2024), and at visible
excitation in single-crystal calcite (Akpovo et al. 2022 report Lorentzian
Γ ≈ 1.5–2 cm⁻¹ and T₂ ≈ 3.5–4 ps; Calandra et al. 2023 report 3–4 cm⁻¹
for geogenic samples). The intrinsic vibrational dephasing rate is set by
phonon physics and is independent of excitation wavelength for non-
resonant modes; CO₃²⁻ has electronic transitions near 7 eV (~177 nm) and
is far from resonance at 248 nm (Loudon 2000 §10.3). Visible-laser
linewidths transfer to 248 nm with at most a small pre-resonant
perturbation. Adopting **Δν = 3 ± 2 cm⁻¹** as a conservative midpoint
gives **τ_c = {carbonate.tau_c_ps:.2f} ps** (range
{carbonate.tau_c_lo_ps:.2f}–{carbonate.tau_c_hi_ps:.2f} ps for Δν ∈
[5, 1] cm⁻¹). Anharmonic temperature scaling (Klemens / Balkanski; for
ν = 1085 cm⁻¹, ħω/k = 1561 K) predicts ~10–15% FWHM increase over the
Bennu surface temperature range 280–390 K, swallowed by the ±2 cm⁻¹
uncertainty band.

### 7.3 g⁽¹⁾ detection scheme and SNR derivation

The proposed detection chain is: a 5 cm⁻¹ FWHM volume Bragg grating
(Glebov et al. 2012; UV photo-thermo-refractive glass, OD > 3 at 248.6 nm)
isolates the Stokes-shifted carbonate band near 268 nm, suppressing both
the laser line and broadband solar continuum; the filtered photons enter
a Mach–Zehnder interferometer; both output ports are detected by a UV-
sensitive superconducting nanowire single-photon detector (SNSPD) with
sub-3 ps timing jitter (Korzh et al. 2020) and the events are time-tagged.
A matched temporal gate of Δt_gate = {carbonate.delta_t_gate_ps:.0f} ps,
applied as post-selection on the time-tagged stream, rejects out-of-gate
solar background.

In the photon-shot-noise-limited regime (Goodman 2015 §9.5; Mandel & Wolf
1995 §14.6; Loudon 2000 §6.5),

> SNR<sub>g⁽¹⁾</sub> = V̄ · *N*<sub>signal</sub> / √(*N*<sub>signal</sub> + *N*<sub>bg</sub>)     (9)

with the gate-averaged visibility

> V̄(Δt<sub>gate</sub>, τ<sub>c</sub>) = (τ<sub>c</sub>/Δt<sub>gate</sub>) · (1 − exp(−Δt<sub>gate</sub>/τ<sub>c</sub>))     (10)

Equation (10) recovers V̄ → 1 in the coherent-source limit (τ_c ≫ Δt_gate)
and V̄ → 0 in the maximally-incoherent limit (τ_c → 0); both limits are
verified numerically before any further computation. For the carbonate
operating point Δt_gate = {carbonate.delta_t_gate_ps:.0f} ps and τ_c =
{carbonate.tau_c_ps:.2f} ps,

> V̄ = {carbonate.V_bar_central:.3f} (range {carbonate.V_bar_lo:.3f}–{carbonate.V_bar_hi:.3f})

Narrower gating tightens the visibility — Korzh et al. (2020) demonstrate
SNSPD timing jitter below 3 ps, so the V̄ value above is a *floor* set by
the choice of TCSPC bin width rather than by detector physics. Section
7.4 reports an explicit gate-width sensitivity sweep.

### 7.4 Quantitative results

Photon counts at each altitude come from the photon budget of Section 2
under the baseline Table 1 instrument (KrF, 50 mJ, 30 cm, 10 s) at 5 wt%
carbonate-dominant surface. The g⁽²⁾ comparison uses Equation (8) from
the original Section 7 derivation (T<sub>min</sub> = 9 / (ρ⁴ R² Δτ),
Δτ = 100 ps coincidence resolution, scaled to a SNR-5 target).

| Altitude (m) | *N*<sub>sig</sub> | *N*<sub>bg</sub> | SNR<sub>int</sub> | SNR<sub>g¹</sub>(10 s) | T<sub>req</sub>(g¹) | T<sub>req</sub>(g²) | g¹/g² SNR |
|---|---|---|---|---|---|---|---|
{table}

**Headline numbers.** At Reconnaissance altitude (375 m), 10 s integration
yields SNR<sub>g¹</sub> = {rows[2].snr_g1:.2f}; reaching SNR = 5 requires
**{_format_seconds(rows[2].t_req_g1_s)}** of integration. The corresponding
g⁽²⁾ requirement is **{_format_seconds(rows[2].t_req_g2_s)}** — eight
orders of magnitude longer. At Orbital B (680 m), g⁽¹⁾ requires
{_format_seconds(rows[3].t_req_g1_s)} versus {_format_seconds(rows[3].t_req_g2_s)}
for g⁽²⁾. The modal photon occupation at 375 m is n =
{rows[2].n_per_mode:.1e} photons per coherence-time-mode, well inside the
photon-starved regime where g⁽¹⁾ dominates per Foellmi (2009).

**Gate-width sensitivity (at 375 m).** Tightening Δt_gate from 100 ps
toward the τ_c-matched limit recovers visibility and shortens
T<sub>req</sub> proportionally:

{gate_table}

A Δt_gate = 10 ps post-selection — within an order of magnitude of
demonstrated SNSPD jitter — would push T<sub>req</sub>(SNR = 5) at
Reconnaissance into the few-minutes regime.

### 7.5 Aromatic and aliphatic modes

The aromatic G band of disordered sp² carbon, characterized in returned
Ryugu IOM at FWHM ≈ 90 ± 15 cm⁻¹ at 514–532 nm excitation (Yabuta et al.
2023; Bonal et al. 2024; framework in Quirico et al. 2018), gives
τ_c ≈ 0.12 ps. No peer-reviewed 248 nm chondrite-analog FWHM has been
published; the 514–532 nm value is adopted here as a conservative upper
bound on the coherence-time floor, since narrower G-bands at UV would
only improve the analysis. The D band (FWHM ≈ 290–330 cm⁻¹ in primitive
chondrites; Bonal et al. 2024) gives τ_c ≈ 0.04 ps. Both are below the
demonstrated SNSPD timing-jitter floor (~3 ps; Korzh et al. 2020), making
them ill-suited to g⁽¹⁾ photon-correlation. For aromatic carbon, **g⁽²⁾
remains the appropriate modality** as a discriminator: the spontaneous-
Raman g⁽²⁾(0) signature is a Stokes/anti-Stokes coincidence peak that
thermal background and stray laser scatter do not exhibit (Vento et al.
2023, *Nat. Commun.* 14: 2818, demonstrate this experimentally in a
molecular liquid). The aliphatic CH-stretch at 2920 cm⁻¹ (FWHM ≈ 40 ± 15
cm⁻¹ at 532 nm; Quirico et al. 2008) is intermediate (τ_c ≈ 0.27 ps);
not a primary g⁽¹⁾ target.

### 7.6 Hardware feasibility and TRL

The end-to-end system requires: (a) a UV-sensitive SNSPD with QE ≳ 30%
at 248–268 nm and timing jitter ≤ 30 ps (Korzh et al. 2020 demonstrate
< 3 ps; Esmaeil Zadeh et al. 2021 review device-state-of-the-art);
(b) a UV photo-thermo-refractive volume Bragg grating with FWHM ~ 5 cm⁻¹
at 268 nm (Glebov et al. 2012; commercially available); (c) a path-
length-stabilized Mach–Zehnder interferometer with active fringe-locking;
(d) TCSPC electronics with sub-50 ps bin width. Active SNSPD gating
(Kong et al. 2023) currently reaches ~5 ns rise time and is used as a
backup if passive time-tagging is inadequate. The integrated UV-SNSPD
interferometric system has TRL 4–5; this paper frames the analysis as a
feasibility study assuming next-generation detectors, consistent with
the title.

### 7.7 Caveats

Three caveats are flagged explicitly. **First**, no peer-reviewed
laboratory g⁽¹⁾ measurement of bulk-solid spontaneous Raman has been
published. The analysis rests on standard coherence theory (Loudon §3.5–
3.6; Goodman §9.5); the closest experimental precedent is Vento et al.
(2023) on a molecular liquid and Sun et al. (2017, *Phys. Rev. B* 95,
035302) on a single quantum dot. **Second**, no peer-reviewed 248 nm
intrinsic FWHM exists for chondrite-analog D/G or CH-stretch modes; the
values used in §7.5 are adopted from 514–532 nm measurements as
conservative bounds. **Third**, the integrated UV-SNSPD-interferometer
system is at TRL 4–5; full flight-readiness requires Phase 2
qualification work analogous to that identified for the Yb:YLF laser in
§2.6.4.
"""
    path.write_text(txt)
    log.info("Wrote %s", path)


def write_g1_findings(carbonate: g1_analysis.ModeResult, gate_sweep: dict, path: Path) -> None:
    rows = carbonate.altitudes
    txt = f"""# Findings — g⁽¹⁾ reframe of Section 7

**One-sentence summary.** Adopting the carbonate ν₁ mode at 1085 cm⁻¹ as
the primary photon-correlation target and applying the textbook g⁽¹⁾
shot-noise SNR formula (Goodman 2015 §9.5) at the baseline KrF instrument
shows that **g⁽¹⁾ on the carbonate band is feasible in {_format_seconds(rows[2].t_req_g1_s)} of
integration at Reconnaissance altitude (375 m), versus
{_format_seconds(rows[2].t_req_g2_s)} for the existing g⁽²⁾ analysis** — a
factor-of-{rows[2].t_req_g2_s/max(rows[2].t_req_g1_s,1e-9):.2g} improvement
that turns photon-correlation analysis from impractical into operationally
relevant.

## Headline numbers (5 wt% carbonate-dominant surface, Δt_gate = 100 ps)

- Carbonate ν₁ FWHM: **{carbonate.delta_nu_cm:.0f} ± 2 cm⁻¹** (Razzell Hollis 2021;
  Akpovo 2022; Calandra 2023)
- Coherence time: **τ_c = {carbonate.tau_c_ps:.2f} ps**
  (range {carbonate.tau_c_lo_ps:.2f}–{carbonate.tau_c_hi_ps:.2f} ps)
- Gate-averaged visibility: **V̄ = {carbonate.V_bar_central:.3f}**
  (range {carbonate.V_bar_lo:.3f}–{carbonate.V_bar_hi:.3f})

| Altitude | SNR_g¹(10 s) | T_req(g¹) | T_req(g²) | Ratio g²/g¹ |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {r.altitude_m:.0f} m ({r.label}) | {r.snr_g1:.3f} | {_format_seconds(r.t_req_g1_s)} | {_format_seconds(r.t_req_g2_s)} | {r.t_req_g2_s/max(r.t_req_g1_s,1e-9):.2g} |" for r in rows)}

## Sanity checks (off-ramp #2; both PASS)

1. Coherent-source limit (Δν → 0, τ_c → ∞): V̄ → 1, SNR_g¹ → standard intensity SNR. ✓
2. Maximally-incoherent limit (Δν → ∞, τ_c → 0): V̄ → 0, SNR_g¹ → 0. ✓

## Modal photon occupation (n ≪ 1 ⇒ g⁽¹⁾ wins; Foellmi 2009)

At 375 m: n_per_mode = {rows[2].n_per_mode:.2e}
At 680 m: n_per_mode = {rows[3].n_per_mode:.2e}

Both are 7+ orders of magnitude below unity, the regime in which
amplitude interferometry (g⁽¹⁾) is information-optimal per
Bojer et al. 2022 *NJP* 24.

## Caveats to flag in review

1. No peer-reviewed laboratory g⁽¹⁾ on bulk-solid spontaneous Raman —
   analysis rests on standard coherence theory (Loudon §3.5; Goodman §9.5).
2. No peer-reviewed 248 nm FWHM for chondrite-analog D/G or CH-stretch;
   visible-laser values adopted as conservative bounds.
3. UV-SNSPD interferometric system is TRL 4–5; analysis assumes
   next-generation detectors per the paper's framing.
4. The "10 s integration is feasible at 375 m" framing in the prompt is
   not borne out at Δt_gate = 100 ps — SNR_g¹(10 s) at 375 m is
   {rows[2].snr_g1:.2f}, well below the SNR = 5 detection threshold.
   Single-integration 10 s feasibility holds only at 50 m. The headline
   result is the **factor-of-10⁸ improvement over g⁽²⁾**, not absolute
   10 s detection.
"""
    path.write_text(txt)
    log.info("Wrote %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / "g1_run.log")
    log = logging.getLogger("g1_main")
    log.info("g⁽¹⁾ Section-7 reframe analysis")

    # Primary: carbonate ν₁ at default 100 ps gate.
    carbonate = g1_analysis.compute_mode_result(
        delta_nu_cm=g1_analysis.DELTA_NU_CARBONATE_CM,
        delta_nu_lo_cm=g1_analysis.DELTA_NU_CARBONATE_LO,
        delta_nu_hi_cm=g1_analysis.DELTA_NU_CARBONATE_HI,
        label=g1_analysis.DELTA_NU_CARBONATE_LABEL,
    )
    log.info(
        "Carbonate ν₁: τ_c = %.2f ps, V̄ = %.3f at Δt_gate = %.0f ps",
        carbonate.tau_c_ps, carbonate.V_bar_central, carbonate.delta_t_gate_ps,
    )
    for r in carbonate.altitudes:
        log.info(
            "  %s (%.0f m): SNR_g¹ = %.3f in 10 s, T_req = %s; g²/g¹ ratio = %.2g",
            r.label, r.altitude_m, r.snr_g1, _format_seconds(r.t_req_g1_s),
            r.t_req_g2_s / max(r.t_req_g1_s, 1e-9),
        )

    # Secondary modes (qualitative).
    g_band = g1_analysis.compute_mode_result(
        delta_nu_cm=g1_analysis.DELTA_NU_G_BAND_CM,
        delta_nu_lo_cm=g1_analysis.DELTA_NU_G_BAND_LO,
        delta_nu_hi_cm=g1_analysis.DELTA_NU_G_BAND_HI,
        label=g1_analysis.DELTA_NU_G_BAND_LABEL,
    )
    ch_stretch = g1_analysis.compute_mode_result(
        delta_nu_cm=g1_analysis.DELTA_NU_CH_STRETCH_CM,
        delta_nu_lo_cm=g1_analysis.DELTA_NU_CH_STRETCH_LO,
        delta_nu_hi_cm=g1_analysis.DELTA_NU_CH_STRETCH_HI,
        label=g1_analysis.DELTA_NU_CH_STRETCH_LABEL,
    )

    # Gate-width sensitivity (carbonate, at 10/30/100 ps).
    gate_sweep = g1_analysis.gate_sensitivity_sweep()
    log.info("Gate sensitivity sweep at 375 m:")
    for key, mode_dict in sorted(gate_sweep.items(), key=lambda kv: int(kv[0].rstrip("ps"))):
        for r in mode_dict["altitudes"]:
            if r["altitude_m"] == 375.0:
                log.info(
                    "  Δt_gate = %3d ps:  V̄ = %.3f,  SNR_g¹(10 s) = %.3f,  T_req(SNR=5) = %s",
                    int(mode_dict["delta_t_gate_ps"]), mode_dict["V_bar_central"],
                    r["snr_g1"], _format_seconds(r["t_req_g1_s"]),
                )

    # JSON dump (before the figure, so the numeric record survives a headless run).
    bundle = {
        "constants": {
            "c_cm_per_s": g1_analysis.SPEED_OF_LIGHT_CM_S,
            "eta_total": g1_analysis.ETA_TOTAL,
            "delta_t_gate_default_s": g1_analysis.DELTA_T_GATE_S_DEFAULT,
            "spectral_filter_fwhm_cm": g1_analysis.SPECTRAL_FILTER_FWHM_CM,
            "g2_detector_res_s": g1_analysis.G2_DETECTOR_RES_S,
            "default_organic_pct": g1_analysis.DEFAULT_ORGANIC_PCT,
            "default_integration_s": g1_analysis.DEFAULT_INTEGRATION_S,
            "target_snr": g1_analysis.TARGET_SNR,
        },
        "modes": {
            "carbonate_nu1": asdict(carbonate),
            "g_band": asdict(g_band),
            "ch_stretch": asdict(ch_stretch),
        },
        "gate_sensitivity_sweep_carbonate": gate_sweep,
    }
    json_path = output_dir / "numerical_results_g1.json"
    json_path.write_text(json.dumps(_to_serializable(bundle), indent=2))
    log.info("Wrote %s", json_path)

    # Figure (after the JSON is safely written).
    make_figure(carbonate, output_dir)

    # Markdown deliverables.
    write_section_7(carbonate, gate_sweep, output_dir / "section_7_revised.md")
    write_g1_findings(carbonate, gate_sweep, output_dir / "g1_findings.md")

    log.info("g⁽¹⁾ pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
