"""Laser trade study for Section 2.6 / Table 3.

Max SNR>=5 detection range at the central K_R for the space-qualifiable laser
options: the KrF excimer baseline (50 mJ, 20 Hz) and the Yb:YLF 4-omega design
points (1.0 mJ at 20% quadrupling and 2.0 mJ at 40%, both at 500 Hz). The 500 Hz
options fire 25x more pulses in a 10 s integration than the 20 Hz KrF, so they
accrue proportionally more per-pulse read noise; this trade is what the function
captures, and it is why the higher-rep options do not simply win on average power.

Writes Outputs/results/laser_trade.csv so Table 3 is reproducible from the repo.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

import src_fixed

log = logging.getLogger(__name__)

# (label, UV pulse energy J, rep rate Hz)
LASER_CASES = (
    ("KrF excimer (baseline)", 0.050, 20),
    ("Yb:YLF 4omega, 20% quad", 0.0010, 500),
    ("Yb:YLF 4omega, 40% quad", 0.0020, 500),
)


def max_detection_range_m(pulse_energy_J, rep_rate, organic_pct=5.0,
                          aperture_m=0.30, integration_time_s=10.0,
                          snr_target=5.0, lo=50.0, hi=5000.0) -> float:
    """Largest range at which SNR >= snr_target, by bisection."""
    if src_fixed.calculate_snr(lo, organic_pct, pulse_energy_J=pulse_energy_J,
                               aperture_m=aperture_m,
                               integration_time_s=integration_time_s,
                               rep_rate=rep_rate) < snr_target:
        return float("nan")
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        snr = src_fixed.calculate_snr(mid, organic_pct,
                                      pulse_energy_J=pulse_energy_J,
                                      aperture_m=aperture_m,
                                      integration_time_s=integration_time_s,
                                      rep_rate=rep_rate)
        if snr >= snr_target:
            lo = mid
        else:
            hi = mid
    return lo


def run(output_dir="Outputs/results") -> list[dict]:
    rows = []
    for label, energy, rep in LASER_CASES:
        rng = max_detection_range_m(energy, rep)
        rows.append({
            "config": label,
            "uv_pulse_energy_mJ": energy * 1e3,
            "rep_rate_hz": rep,
            "max_detection_range_m": round(rng, 1),
        })
        log.info("  %-26s %.1f mJ / %d Hz -> %.0f m", label, energy * 1e3, rep, rng)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "laser_trade.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %s", path)
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
