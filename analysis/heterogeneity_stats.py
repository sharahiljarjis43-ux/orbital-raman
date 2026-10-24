"""Empirical heterogeneity statistics for OVIRS Recon-A facets.

Produces:
- Empirical wt%-C distribution and CDF table
- Multi-pointing detection probability P(≥1 organic intercept | n)
- 5°×5° spatial autocorrelation (Moran's I) on band area / wt% C

The detection-probability calculation uses the empirical fraction of facets
classified as organic-dominant as f_org. P(at least one hit | n pointings)
= 1 − (1 − f_org)^n, with binomial 95% Wilson-style CIs derived from the
Ferrone (2021) Recon-A sample size (15,585 spectra).

Moran's I is computed on a coarse 5°×5° lat-lon grid: bin facets, average
band_area within each bin, weight neighbors by inverse great-circle distance,
report I and its z-score. This is sufficient for "is heterogeneity spatially
clustered or random" — a publication-grade Moran's I requires a Bennu shape-
model adjacency graph that is out of scope here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

# Multi-pointing analysis grid.
N_POINTINGS_GRID = (1, 5, 10, 25, 50, 100, 250, 500, 1000)


@dataclass
class HeterogeneityReport:
    n_facets: int
    wt_c_mean: float
    wt_c_median: float
    wt_c_std: float
    wt_c_p95: float
    wt_c_max: float
    cdf_thresholds: list[float]
    f_above_threshold: list[float]
    f_organic: float
    f_organic_ci_lo: float
    f_organic_ci_hi: float
    detection_prob: dict[int, dict[str, float]] = field(default_factory=dict)
    n_required: dict[float, int] = field(default_factory=dict)
    morans_i: float | None = None
    morans_i_z: float | None = None
    morans_i_p: float | None = None
    spatial_coverage_5deg: float | None = None


def cdf_table(
    wt_c: np.ndarray, thresholds=(1.0, 2.0, 3.0, 5.0, 10.0)
) -> tuple[list[float], list[float]]:
    wt_c = np.asarray(wt_c)
    return list(thresholds), [float(np.mean(wt_c >= t)) for t in thresholds]


def organic_fraction_with_ci(
    n_organic: int, n_total: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Wilson-score 95% CI on the organic fraction."""
    f = n_organic / n_total
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n_total
    centre = (f + z**2 / (2 * n_total)) / denom
    half = (z / denom) * np.sqrt(f * (1 - f) / n_total + z**2 / (4 * n_total**2))
    return f, max(0.0, centre - half), min(1.0, centre + half)


def detection_probability_table(
    f_org: float,
    f_lo: float,
    f_hi: float,
    n_grid: tuple[int, ...] = N_POINTINGS_GRID,
) -> dict[int, dict[str, float]]:
    """P(at least 1 organic intercept | n pointings) with 95% CI band."""
    table: dict[int, dict[str, float]] = {}
    for n in n_grid:
        table[int(n)] = {
            "p": 1.0 - (1.0 - f_org) ** n,
            "p_lo": 1.0 - (1.0 - f_lo) ** n,
            "p_hi": 1.0 - (1.0 - f_hi) ** n,
        }
    return table


def n_required(f_org: float, p_targets=(0.50, 0.90, 0.95, 0.99)) -> dict[float, int]:
    """Smallest n such that 1 − (1 − f_org)^n ≥ p_target."""
    out: dict[float, int] = {}
    for p in p_targets:
        if f_org <= 0:
            out[float(p)] = -1
            continue
        n = int(np.ceil(np.log(1.0 - p) / np.log(1.0 - f_org)))
        out[float(p)] = max(1, n)
    return out


def morans_i_5deg(df: pd.DataFrame, value_col: str = "band_area") -> dict:
    """Compute Moran's I on a 5°×5° lat-lon grid of mean band-area per cell.

    Uses inverse great-circle-distance neighbor weights, row-standardized.
    Returns I, expected-under-null E[I] = -1/(N-1), z-score, and two-sided p.
    """
    lat_edges = np.arange(-90.0, 90.001, 5.0)
    lon_edges = np.arange(0.0, 360.001, 5.0)
    lat_idx = np.digitize(df["latitude_deg"].to_numpy(), lat_edges) - 1
    lon_idx = np.digitize(df["longitude_deg"].to_numpy() % 360.0, lon_edges) - 1

    nlat = len(lat_edges) - 1
    nlon = len(lon_edges) - 1
    sums = np.zeros((nlat, nlon))
    counts = np.zeros((nlat, nlon))
    vals = df[value_col].to_numpy()

    for k in range(len(vals)):
        i, j = lat_idx[k], lon_idx[k]
        if 0 <= i < nlat and 0 <= j < nlon:
            sums[i, j] += vals[k]
            counts[i, j] += 1

    mask = counts > 0
    means = np.full_like(sums, np.nan)
    means[mask] = sums[mask] / counts[mask]
    coverage = float(mask.mean())

    cells = np.argwhere(mask)
    n = len(cells)
    if n < 5:
        return {"I": None, "z": None, "p": None, "coverage": coverage, "n_cells": n}

    centers_lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])[cells[:, 0]]
    centers_lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])[cells[:, 1]]
    cell_vals = means[mask]
    mean_v = cell_vals.mean()
    dev = cell_vals - mean_v

    lat_rad = np.radians(centers_lat)
    lon_rad = np.radians(centers_lon)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    cos_dl = np.cos(lon_rad[None, :] - lon_rad[:, None])
    cos_d = (
        sin_lat[:, None] * sin_lat[None, :]
        + cos_lat[:, None] * cos_lat[None, :] * cos_dl
    )
    cos_d = np.clip(cos_d, -1.0, 1.0)
    d = np.arccos(cos_d)  # great-circle angular distance, radians

    with np.errstate(divide="ignore"):
        W = np.where(d > 0, 1.0 / (d + 1e-3), 0.0)
    np.fill_diagonal(W, 0.0)
    row_sum = W.sum(axis=1, keepdims=True)
    W = np.where(row_sum > 0, W / row_sum, 0.0)

    S0 = W.sum()
    num = (W * np.outer(dev, dev)).sum()
    denom = (dev**2).sum()
    if denom <= 0 or S0 <= 0:
        return {"I": None, "z": None, "p": None, "coverage": coverage, "n_cells": n}
    I = (n / S0) * (num / denom)

    EI = -1.0 / (n - 1)
    Wt = W + W.T
    S1 = 0.5 * (Wt**2).sum()
    S2 = ((W.sum(axis=1) + W.sum(axis=0)) ** 2).sum()
    A = n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * S0**2)
    B = (((dev**4).sum() / n) / ((denom / n) ** 2)) * (
        (n**2 - n) * S1 - 2 * n * S2 + 6 * S0**2
    )
    C = (n - 1) * (n - 2) * (n - 3) * S0**2
    if C <= 0:
        return {
            "I": float(I),
            "z": None,
            "p": None,
            "coverage": coverage,
            "n_cells": n,
        }
    EI2 = (A - B) / C
    var_I = EI2 - EI**2
    if var_I <= 0:
        return {
            "I": float(I),
            "z": None,
            "p": None,
            "coverage": coverage,
            "n_cells": n,
        }
    z = (I - EI) / np.sqrt(var_I)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "I": float(I),
        "z": float(z),
        "p": float(p),
        "coverage": coverage,
        "n_cells": int(n),
    }


def build_report(
    df: pd.DataFrame,
    n_organic: int,
    n_total: int,
    cdf_thresholds=(1.0, 2.0, 3.0, 5.0, 10.0),
) -> HeterogeneityReport:
    wt_c = df["wt_c_pct"].to_numpy()
    f_org, f_lo, f_hi = organic_fraction_with_ci(n_organic, n_total)
    thr, frac = cdf_table(wt_c, cdf_thresholds)
    morans = morans_i_5deg(df)
    return HeterogeneityReport(
        n_facets=len(df),
        wt_c_mean=float(np.mean(wt_c)),
        wt_c_median=float(np.median(wt_c)),
        wt_c_std=float(np.std(wt_c)),
        wt_c_p95=float(np.percentile(wt_c, 95)),
        wt_c_max=float(np.max(wt_c)),
        cdf_thresholds=thr,
        f_above_threshold=frac,
        f_organic=f_org,
        f_organic_ci_lo=f_lo,
        f_organic_ci_hi=f_hi,
        detection_prob=detection_probability_table(f_org, f_lo, f_hi),
        n_required=n_required(f_org),
        morans_i=morans.get("I"),
        morans_i_z=morans.get("z"),
        morans_i_p=morans.get("p"),
        spatial_coverage_5deg=morans.get("coverage"),
    )
