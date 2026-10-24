"""Compute monostatic self-occlusion on Bennu's shape from the OVIRS FITS geometry.

This is the COARSE cross-check (a global-shape self-facing diagnostic). The
DEFINITIVE occlusion computation is the line-of-sight mesh ray-trace in
`analysis/occlusion_raytrace.py`.

FINDING (this module): self-facing occlusion from the 1.6 m GLOBAL shape is ~0%
(p95 slope 33 deg, max 83 deg; nothing is back-facing near-nadir, and the 1.6 m
GSD is coarser than the ~0.375 m laser footprint) -- so occlusion is
SUB-RESOLUTION. `occlusion_raytrace.py` then ray-traces the boulder-resolving
5 cm Nightingale DTM and finds a few percent, but weighting-dependent
(face-count ~1.6%, raw area-weighted ~8.9% inflated by Poisson-fill artifacts,
artifact-corrected ~3.9%) -- consistent with the retained 3% nominal and 1-8%
band, not a precise point value; the coarse 75 cm/1.6 m global mesh gives 0%
there too, confirming the sub-resolution conclusion. The slope distribution here
(mean 16.6 deg) validates the estimated normals against the published Bennu
global mean ~17 deg (DellaGiustina et al. 2019; Barnouin et al. 2019).

Method (data-in-hand; see design consult with GPT/Codex)
-------------------------------------------------------
We have all 786,432 shape-model facet CENTERS as (lat, lon, radius) -- a
single-valued heightfield -- for the 1.6 m global model. From these:

1. Cartesian facet positions r_i.
2. Per-facet outward normals n_i via k-NN local-plane fit (PCA / smallest
   eigenvector of the neighbor covariance), oriented outward (n . r > 0).
3. Local slope = angle(n_i, radial_i).
4. Monostatic self-occlusion under PER-FACET near-nadir viewing: place the
   spacecraft at altitude h along facet i's local radial (near-nadir), look
   vector s_i = (sc - r_i)/|.|, emission angle e_i = arccos(n_i . s_i). Facet
   is self-occluded (returns no signal to the co-aligned receiver) when
   e_i > e_max (default 90 deg; a grazing variant at 85 deg is also reported).

Honest limits (stated in the paper):
- A radius(lat,lon) heightfield CANNOT represent true overhangs / concavities;
  those are an unmodeled residual on top of this number.
- Under per-facet near-nadir the line of sight is ~radial, so ridge-blocking by
  OTHER facets is negligible in a heightfield; self-facing dominates. Genuine
  ridge/overhang blocking needs the full OBJ mesh (future work).
- The number is therefore a LOWER BOUND on true occlusion; the 1-8% sensitivity
  band in SpaceEnvironment is retained as the robustness bracket.

Reported separately (bias check): occluded fraction over the full 786k-facet
shape AND over the ~48.7k OVIRS-data-carrying facets (Recon-A is not a random
sample of Bennu).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

FILL = -9990.0
ALT_KM_DEFAULT = 0.375  # Reconnaissance altitude


def _latlonrad_to_xyz(lat_deg, lon_deg, radius_km):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    cl = np.cos(lat)
    x = radius_km * cl * np.cos(lon)
    y = radius_km * cl * np.sin(lon)
    z = radius_km * np.sin(lat)
    return np.column_stack([x, y, z])


def estimate_normals(pos: np.ndarray, k: int = 12) -> np.ndarray:
    """Outward unit normals via k-NN local-plane PCA (vectorized)."""
    tree = cKDTree(pos)
    # k+1 because the nearest neighbor is the point itself.
    _, idx = tree.query(pos, k=k + 1)
    neigh = pos[idx[:, 1:]]                      # (N, k, 3)
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k   # (N, 3, 3)
    # smallest-eigenvalue eigenvector = surface normal
    _, vecs = np.linalg.eigh(cov)               # ascending eigenvalues
    normals = vecs[:, :, 0]                      # (N, 3)
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    flip = np.sum(normals * radial, axis=1) < 0
    normals[flip] *= -1.0
    return normals


@dataclass
class OcclusionResult:
    k: int
    e_max_deg: float
    alt_km: float
    n_full: int
    n_ovirs: int
    occ_frac_full: float
    occ_frac_ovirs: float
    slope_mean_deg: float
    slope_median_deg: float
    slope_p95_deg: float
    extra: dict = field(default_factory=dict)


def compute(
    fits_path: str | Path,
    k: int = 12,
    e_max_deg: float = 90.0,
    alt_km: float = ALT_KM_DEFAULT,
) -> OcclusionResult:
    hdu = fits.open(fits_path)[1].data
    lat = np.asarray(hdu["LATITUDE"], float)
    lon = np.asarray(hdu["LONGITUDE"], float)
    rad = np.asarray(hdu["RADIUS"], float)
    val = np.asarray(hdu["VALUE"], float)
    has_data = val > FILL

    pos = _latlonrad_to_xyz(lat, lon, rad)       # (N, 3), km
    normals = estimate_normals(pos, k=k)
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)

    # local slope (facet normal vs local radial)
    cos_slope = np.clip(np.sum(normals * radial, axis=1), -1, 1)
    slope_deg = np.degrees(np.arccos(cos_slope))

    # per-facet near-nadir spacecraft along local radial, at altitude alt_km
    sc = pos + alt_km * radial
    look = sc - pos
    look /= np.linalg.norm(look, axis=1, keepdims=True)
    cos_e = np.clip(np.sum(normals * look, axis=1), -1, 1)
    emission_deg = np.degrees(np.arccos(cos_e))

    occluded = emission_deg > e_max_deg

    occ_full = float(occluded.mean())
    occ_ovirs = float(occluded[has_data].mean())

    return OcclusionResult(
        k=k,
        e_max_deg=e_max_deg,
        alt_km=alt_km,
        n_full=int(len(pos)),
        n_ovirs=int(has_data.sum()),
        occ_frac_full=occ_full,
        occ_frac_ovirs=occ_ovirs,
        slope_mean_deg=float(slope_deg.mean()),
        slope_median_deg=float(np.median(slope_deg)),
        slope_p95_deg=float(np.percentile(slope_deg, 95)),
    )


def sweep(fits_path: str | Path) -> list[OcclusionResult]:
    """Sensitivity across neighborhood size k and emission threshold."""
    out = []
    for k in (8, 12, 20):
        for e_max in (90.0, 85.0):
            out.append(compute(fits_path, k=k, e_max_deg=e_max))
    return out


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "Data/l_1600mm_sp_ovirs_reca_bandarea3200to3600nm_allsites_wavc_0000n00000.fits"
    )
    base = compute(path)
    print(f"Facets: full={base.n_full:,}  OVIRS-data={base.n_ovirs:,}")
    print(f"Local slope: mean={base.slope_mean_deg:.1f} deg  "
          f"median={base.slope_median_deg:.1f}  p95={base.slope_p95_deg:.1f}")
    print()
    print("Self-facing occluded fraction (per-facet near-nadir, 375 m):")
    print(f"{'k':>3} {'e_max':>6} | {'full-shape':>11} | {'OVIRS-subset':>12}")
    for r in sweep(path):
        print(f"{r.k:>3} {r.e_max_deg:>5.0f}° | {100*r.occ_frac_full:>10.2f}% | "
              f"{100*r.occ_frac_ovirs:>11.2f}%")
