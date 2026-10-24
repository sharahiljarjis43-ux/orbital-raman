"""Photon-budget tests, principally the solar background of Equation (3).

These exist because a 3.6e5 overstatement of N_solar survived the whole existing
suite: nothing asserted a magnitude for the background, only that the pipeline
ran. The binding check is `test_solar_background_respects_geometric_limit`, which
is independent of any convention or fudge factor -- a telescope cannot collect
more light than its own solid angle subtends.
"""
from __future__ import annotations

import numpy as np
import pytest

import src_fixed as S

H, C = 6.626e-34, 3e8
ALBEDO = 0.044
DLAM_NM = 0.5


def _reflected_power_W(distance_m, solar_irr=0.015):
    """Total power the illuminated spot reflects into the hemisphere."""
    a_spot = np.pi * (distance_m * 0.5e-3) ** 2
    return solar_irr * DLAM_NM * a_spot * ALBEDO


def _collected_power_W(n_solar_photons, distance_m, integration_time_s=10.0,
                       wavelength_nm=248.0, qe=0.30, rep_rate=20):
    """Invert a detected photon count back to power at the aperture."""
    e_photon = H * C / (wavelength_nm * 1e-9)
    gate = rep_rate * integration_time_s * S.GATE_PER_PULSE_S
    return n_solar_photons * e_photon / (gate * qe)


@pytest.mark.parametrize("distance_m", [50.0, 200.0, 375.0, 680.0, 1000.0])
def test_solar_background_respects_geometric_limit(distance_m):
    """The telescope cannot collect more than Omega_tel/pi of reflected light.

    A Lambertian surface radiates into pi sr. This is pure geometry: no modelling
    convention can lift it. The pre-fix model exceeded it by ~3.6e5 at 375 m.
    """
    n_solar = S.solar_background_photons(distance_m, 0.30, 10.0, 0.015)
    collected = _collected_power_W(n_solar, distance_m)
    reflected = _reflected_power_W(distance_m)

    a_tel = np.pi * (0.30 / 2) ** 2
    omega_tel = a_tel / distance_m ** 2
    geometric_ceiling = omega_tel / np.pi

    assert collected / reflected <= geometric_ceiling * (1 + 1e-9), (
        f"at {distance_m} m the model collects {collected / reflected:.3e} of the "
        f"reflected light; geometry allows at most {geometric_ceiling:.3e}"
    )


def test_solar_background_matches_first_principles():
    """Equation (3) against an independent target-side radiometric derivation."""
    d, ap, t, irr = 375.0, 0.30, 10.0, 0.015
    a_tel = np.pi * (ap / 2) ** 2
    expected = (
        _reflected_power_W(d, irr)          # W reflected by the spot
        * (a_tel / d ** 2) / np.pi          # fraction into the telescope
        * (20 * t * S.GATE_PER_PULSE_S)     # total gate-open time
        * 0.30                              # QE
        / (H * C / (248 * 1e-9))            # -> photons
    )
    got = S.solar_background_photons(d, ap, t, irr)
    assert got == pytest.approx(expected, rel=1e-6)
    assert got == pytest.approx(8.7, abs=0.5)


def test_solar_background_is_range_invariant():
    """A_spot ~ R^2 and Omega_tel ~ 1/R^2, so the etendue does not depend on range."""
    vals = [S.solar_background_photons(d, 0.30, 10.0, 0.015)
            for d in (50.0, 375.0, 680.0)]
    assert all(v == pytest.approx(vals[0], rel=1e-9) for v in vals)


def test_solar_background_scales_linearly_with_irradiance_and_gate():
    base = S.solar_background_photons(375.0, 0.30, 10.0, 0.015)
    assert S.solar_background_photons(375.0, 0.30, 10.0, 0.030) == pytest.approx(2 * base, rel=1e-9)
    assert S.solar_background_photons(375.0, 0.30, 20.0, 0.015) == pytest.approx(2 * base, rel=1e-9)


def test_budget_is_read_noise_dominated_not_solar():
    """Post-fix the 4e-7 duty cycle leaves read noise dominant, as time-gated
    standoff Raman practice implies (Misra 2012; SuperCam's 100 ns daylight gate)."""
    r = S.calculate_snr_full(375.0, 5.0)
    assert r["N_solar"] < 0.01 * r["N_bg"]
    assert r["N_read"] > 0.5 * r["N_bg"]


def test_snr_entry_points_agree():
    """calculate_snr and calculate_snr_full must not drift apart: the N_solar bug
    was duplicated in both, so a shared-helper regression would be silent."""
    for d in (50.0, 375.0, 680.0):
        assert S.calculate_snr(d, 5.0) == pytest.approx(
            S.calculate_snr_full(d, 5.0)["snr"], rel=1e-9
        )


def test_tsis1_loader_reads_archive_or_raises():
    """The loader must never silently substitute a constant for the archive."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "Data/tsis_ssi_L3_c24h_latest.nc"
    if not path.exists():
        pytest.skip("TSIS-1 archive not present")
    pytest.importorskip("netCDF4")

    irr = S.load_tsis1(path)
    # TSIS-1 gives ~0.0496 W/m^2/nm at 248 nm at 1 AU; Bennu sits at 1.126 AU.
    assert irr == pytest.approx(0.0391, abs=0.005)
    assert irr != 0.015, "loader fell back to the hardcoded default"

    with pytest.raises(Exception):
        S.load_tsis1(path.parent / "does_not_exist.nc")
