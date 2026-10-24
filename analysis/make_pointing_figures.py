"""Figures for the spatial-sampling reframe (pointing campaign + K_R bounds)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 100,
}


def make_pointing_figure(report, output_dir) -> dict:
    """Three panels: correlogram, loci census, campaign-extent saturation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(_RC)
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) correlogram: lift vs lag
    edges = report.lag_edges_m
    centers = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)]
    lift = report.correlogram_lift
    axa.loglog(centers, lift, "o-", color="#1f77b4", lw=2, ms=6)
    axa.axhline(1.0, color="gray", ls="--", lw=1, label="no spatial memory (lift = 1)")
    axa.axvline(report.decorrelation_length_m, color="#d62728", ls=":", lw=1.5,
                label=f"decorrelation ~{report.decorrelation_length_m:.0f} m")
    axa.set_xlabel("3D lag between facets (m)")
    axa.set_ylabel("organic-neighbor lift  P(org|org)/f$_{org}$")
    axa.set_title("(a) Organic-terrain correlogram")
    axa.legend(loc="upper right")
    axa.grid(True, which="both", alpha=0.25)

    # (b) loci census: distinct loci vs linking distance
    axb.plot(report.link_distances_m, report.n_loci, "s-", color="#2ca02c", lw=2, ms=6)
    axb.axhline(report.n_organic, color="gray", ls="--", lw=1,
                label=f"{report.n_organic} organic facets")
    i5 = report.link_distances_m.index(5.0) if 5.0 in report.link_distances_m else 0
    axb.annotate(
        f"{report.n_loci[i5]} loci\n({report.pseudo_replication_factor[i5]:.0f}x pseudo-rep)",
        xy=(5.0, report.n_loci[i5]), xytext=(8, report.n_organic * 0.7),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8), fontsize=8.5,
    )
    axb.set_xlabel("linking distance (m)")
    axb.set_ylabel("distinct organic loci")
    axb.set_title("(b) Independent-loci census")
    axb.legend(loc="upper right")
    axb.grid(True, alpha=0.25)

    # (c) campaign-extent saturation: P(detect) vs n, one curve per patch radius
    n_arr = report.campaign_n_pointings
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(report.patch_radii_m)))
    for D, col in zip(report.patch_radii_m, cmap):
        axc.plot(n_arr, report.campaign_p_detect[D], "o-", color=col, lw=1.8, ms=4,
                 label=f"patch D = {D:.0f} m")
    axc.plot(n_arr, report.campaign_global_p_detect, "k^--", lw=2, ms=6,
             label="globally dispersed")
    axc.axhline(0.95, color="gray", ls=":", lw=1)
    axc.set_xscale("log")
    axc.set_xlabel("number of pointings n")
    axc.set_ylabel("P($\\geq$1 organic intercept)")
    axc.set_ylim(0, 1.02)
    axc.set_title("(c) Campaign-extent saturation")
    axc.legend(loc="center right", fontsize=7.5)
    axc.grid(True, alpha=0.25)

    plt.tight_layout()
    png = output_dir / "fig_pointing_campaign.png"
    pdf = output_dir / "fig_pointing_campaign.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s and %s", png.name, pdf.name)
    return {"png": png, "pdf": pdf}


def make_kr_figure(kr_report, output_dir) -> dict:
    """Detection boundary: SNR vs distance as a band across the plausible K_R range."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    boundary = kr_report.detection_boundary
    keys = sorted(boundary.keys(), key=lambda k: float(k))
    dists = sorted(next(iter(boundary.values())).keys())
    lo = [boundary[keys[0]][d] for d in dists]
    hi = [boundary[keys[-1]][d] for d in dists]
    ax.fill_between(dists, lo, hi, alpha=0.2, color="#1f77b4",
                    label=f"K$_R$ range {keys[0]}–{keys[-1]} sr$^{{-1}}$")
    for k in keys:
        row = [boundary[k][d] for d in dists]
        style = "-" if k == f"{kr_report.kr_central:.0e}" else "--"
        lw = 2.4 if k == f"{kr_report.kr_central:.0e}" else 1.2
        lbl = f"K$_R$ = {k}" + (" (central)" if k == f"{kr_report.kr_central:.0e}" else "")
        ax.plot(dists, row, style, lw=lw, label=lbl)

    ax.axhline(5.0, color="#d62728", ls=":", lw=1.5, label="detection threshold SNR = 5")
    ax.set_yscale("log")
    ax.set_xlabel("range (m)")
    ax.set_ylabel("SNR (5 wt% organics, 10 s)")
    ax.set_title("Detection boundary vs Raman backscatter coefficient K$_R$")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    png = output_dir / "fig_kr_boundary.png"
    pdf = output_dir / "fig_kr_boundary.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s and %s", png.name, pdf.name)
    return {"png": png, "pdf": pdf}
