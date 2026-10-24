"""Produce the two-panel heterogeneity figure (Section 4.4.5 / Figure 11).

Panel (a): Empirical wt% C distribution from OVIRS Recon-A vs the uniform
1-15% prior used in the original Monte Carlo.

Panel (b): P(at least one organic intercept | n pointings) with
factor-of-2 calibration sensitivity band.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


def make_figure(
    wt_c_empirical: np.ndarray,
    f_organic: float,
    f_organic_lo: float,
    f_organic_hi: float,
    output_dir: str | Path,
    n_pointings_grid: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000),
) -> dict[str, Path]:
    output_dir = Path(output_dir)
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
            "legend.fontsize": 9,
            "figure.dpi": 100,
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Panel (a): empirical wt%-C distribution vs uniform prior ----
    bins = np.linspace(0, 20, 81)
    ax_a.hist(
        wt_c_empirical,
        bins=bins,
        density=True,
        color="#1f77b4",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
        label=f"OVIRS α-anchored proxy (n={len(wt_c_empirical):,})",
    )

    uniform_density = 1.0 / (15.0 - 1.0)
    ax_a.plot(
        [1.0, 1.0, 15.0, 15.0],
        [0.0, uniform_density, uniform_density, 0.0],
        color="#d62728",
        linewidth=2.0,
        label="Uniform 1-15% prior (original)",
    )

    median = float(np.median(wt_c_empirical))
    p95 = float(np.percentile(wt_c_empirical, 95))
    ax_a.axvline(median, color="black", linestyle="--", linewidth=1.0,
                 label=f"Proxy median = {median:.2f}%")
    ax_a.axvline(p95, color="gray", linestyle=":", linewidth=1.0,
                 label=f"Proxy 95th pct = {p95:.2f}%")

    ax_a.set_xlabel("α-anchored proxy organic carbon (wt%)")
    ax_a.set_ylabel("Probability density")
    ax_a.set_xlim(0, 20)
    ax_a.legend(loc="upper right", framealpha=0.92)
    ax_a.text(
        0.02, 0.98,
        "(a)",
        transform=ax_a.transAxes,
        fontweight="bold",
        va="top", ha="left",
    )

    # ---- Panel (b): multi-pointing detection probability ----
    # No facet-level Wilson CI band: the 48,686 facets derive from ~15,585
    # spatially clustered OVIRS spectra, so a facet-level CI would understate
    # the true sampling uncertainty (Section 4.4.5). Only the point-estimate
    # curve and the factor-of-2 calibration envelope are shown.
    n_arr = np.asarray(n_pointings_grid)
    p = 1.0 - (1.0 - f_organic) ** n_arr

    ax_b.plot(n_arr, p, "o-", color="#2ca02c", lw=2.0, ms=5,
              label="Calibrated f_org prevalence")

    # Factor-of-2 calibration sensitivity envelope on f_org itself.
    f_lo_cal = max(f_organic / 2.0, 1e-6)
    f_hi_cal = min(f_organic * 2.0, 0.999)
    p_cal_lo = 1.0 - (1.0 - f_lo_cal) ** n_arr
    p_cal_hi = 1.0 - (1.0 - f_hi_cal) ** n_arr
    ax_b.plot(n_arr, p_cal_lo, "--", color="gray", lw=1.0,
              label="½× / 2× calibration band")
    ax_b.plot(n_arr, p_cal_hi, "--", color="gray", lw=1.0)

    for p_target, color in [(0.90, "#ff7f0e"), (0.95, "#9467bd")]:
        if f_organic > 0:
            n_star = int(np.ceil(np.log(1.0 - p_target) / np.log(1.0 - f_organic)))
            ax_b.axvline(n_star, color=color, linestyle=":", lw=1.0,
                         label=f"n* for P={p_target:.0%}: {n_star}")

    ax_b.set_xscale("log")
    ax_b.set_xlabel("Number of pointings n")
    ax_b.set_ylabel("P(≥1 organic intercept)")
    ax_b.set_xlim(1, max(n_arr))
    ax_b.set_ylim(0, 1.02)
    ax_b.legend(loc="lower right", framealpha=0.92)
    ax_b.text(
        0.02, 0.98,
        "(b)",
        transform=ax_b.transAxes,
        fontweight="bold",
        va="top", ha="left",
    )

    plt.tight_layout()

    png_path = output_dir / "fig_heterogeneity.png"
    pdf_path = output_dir / "fig_heterogeneity.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s and %s", png_path.name, pdf_path.name)
    return {"png": png_path, "pdf": pdf_path}
