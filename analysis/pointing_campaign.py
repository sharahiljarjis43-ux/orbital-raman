"""Spatial-sampling analysis of organic-dominant terrain on Bennu.

Once the photon budget is comfortable (Section 4/5), the binding constraint on
an orbital Raman campaign is no longer signal, it is where to point. This module
measures the spatial structure of organic-dominant facets directly from the
OVIRS Recon-A classification and converts it into a pointing-campaign design rule.

Three results, all derived from the facet map alone and therefore independent of
the photon budget, K_R, and the instrument design:

1. Correlogram: P(neighbor organic | center organic) as a function of 3D lag,
   normalized by the global prevalence f_org. Gives the decorrelation length of
   organic-rich terrain. This supersedes the coarse 5 deg x 5 deg Moran's I of
   heterogeneity_stats (a ~250 m grid, far coarser than the structure that
   governs pointing).

2. Loci census: connected components of the organic facets at a range of linking
   distances. The number of distinct loci is the effective sample size, which
   quantifies the pseudo-replication the classification otherwise hides.

3. Campaign-extent simulation: P(>=1 organic intercept | n pointings) when the
   n pointings are confined to a patch of radius D. Confining a campaign to a
   single correlation cell saturates far below unity no matter how many pointings
   are taken; dispersing over many cells recovers the independent-Bernoulli count.

Determinism: every random draw uses an explicit numpy Generator seeded from the
`seed` argument (default 42), matching the rest of the pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

# Lag-bin edges (metres) for the correlogram. Chosen to straddle the expected
# decorrelation length; the first bin is the facet-adjacency scale.
LAG_EDGES_M = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)

# Linking distances (metres) for the connected-component loci census.
LINK_DISTANCES_M = (1.5, 2.0, 3.0, 5.0, 10.0, 20.0)

# Patch radii (metres) and pointing counts for the campaign-extent simulation.
PATCH_RADII_M = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
CAMPAIGN_N_POINTINGS = (41, 176, 271, 500)
CAMPAIGN_TRIALS = 2000

# Spatial block-bootstrap block size (degrees). Must exceed the ~10-20 m (~3 deg
# at Bennu's radius) organic decorrelation length so within-block correlation is
# captured and blocks are approximately independent; 15 deg leaves ~70 blocks.
PREVALENCE_BLOCK_DEG = 15.0
PREVALENCE_N_BOOT = 2000

# OSIRIS-REx candidate sample sites (Lauretta et al. 2021), (lat_deg, lon_deg),
# for the leave-one-site-out robustness check.
SAMPLE_SITES = {
    "Nightingale": (56.0, 42.0),
    "Kingfisher": (11.0, 56.0),
    "Osprey": (11.0, 88.0),
    "Sandpiper": (-47.0, 322.0),
}


@dataclass
class PointingReport:
    n_facets: int
    n_organic: int
    f_organic: float
    # correlogram
    lag_edges_m: list[float]
    correlogram_pairs: list[int]
    correlogram_p_organic: list[float]
    correlogram_lift: list[float]
    decorrelation_length_m: float
    # loci census
    link_distances_m: list[float]
    n_loci: list[int]
    pseudo_replication_factor: list[float]
    # campaign-extent simulation
    patch_radii_m: list[float]
    campaign_n_pointings: list[int]
    campaign_p_detect: dict[float, list[float]] = field(default_factory=dict)
    campaign_global_p_detect: list[float] = field(default_factory=list)
    campaign_analytic_p_detect: list[float] = field(default_factory=list)
    # clustering-aware prevalence CI
    f_organic_wilson_ci: list[float] = field(default_factory=list)
    f_organic_block_ci: list[float] = field(default_factory=list)
    prevalence_block_deg: float = PREVALENCE_BLOCK_DEG
    prevalence_n_blocks: int = 0
    prevalence_design_effect: float = 1.0
    leave_one_site_out: dict[str, float] = field(default_factory=dict)


def facet_positions_m(df: pd.DataFrame) -> np.ndarray:
    """3D Cartesian positions of facet centres, in metres, on the real shape.

    Uses the archived per-facet latitude, longitude, and body-centred radius, so
    distances are measured on Bennu's actual figure rather than a sphere.
    """
    lat = np.radians(df["latitude_deg"].to_numpy())
    lon = np.radians(df["longitude_deg"].to_numpy())
    r = df["radius_km"].to_numpy() * 1000.0
    return np.column_stack(
        [r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)]
    )


def correlogram(
    pos: np.ndarray, organic: np.ndarray, edges=LAG_EDGES_M
) -> tuple[list[int], list[float], list[float]]:
    """P(neighbor organic | center organic) and its lift, per lag bin.

    For every organic facet, all facets within the largest lag are gathered once
    (a single ball query per centre) and bucketed by exact 3D distance.
    """
    f0 = float(organic.mean())
    tree = cKDTree(pos)
    org_idx = np.flatnonzero(organic)
    max_lag = float(edges[-1])
    neighbors = tree.query_ball_point(pos[org_idx], max_lag)

    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    hits = np.zeros(len(edges) - 1, dtype=np.int64)
    edges_arr = np.asarray(edges)
    for ci, nb in zip(org_idx, neighbors):
        if not nb:
            continue
        nb = np.asarray(nb)
        nb = nb[nb != ci]
        if nb.size == 0:
            continue
        d = np.linalg.norm(pos[nb] - pos[ci], axis=1)
        b = np.searchsorted(edges_arr, d, side="right") - 1
        valid = (b >= 0) & (b < len(counts))
        np.add.at(counts, b[valid], 1)
        np.add.at(hits, b[valid], organic[nb][valid].astype(np.int64))

    p = [float(h / c) if c else float("nan") for h, c in zip(hits, counts)]
    lift = [pi / f0 if np.isfinite(pi) else float("nan") for pi in p]
    return counts.tolist(), p, lift


def decorrelation_length(edges, lift, threshold: float = 2.0) -> float:
    """First lag-bin upper edge at which the lift falls below `threshold`.

    The lift starts high at short lag and decays toward 1 (no spatial memory).
    The decorrelation length is where organic-rich structure stops predicting
    its neighbourhood; we report the bin edge at which lift first drops below 2x.
    """
    for (lo, hi), lf in zip(zip(edges[:-1], edges[1:]), lift):
        if np.isfinite(lf) and lf < threshold:
            return float(hi)
    return float(edges[-1])


def loci_census(
    pos: np.ndarray, organic: np.ndarray, link_distances=LINK_DISTANCES_M
) -> tuple[list[int], list[float]]:
    """Number of connected components of organic facets vs linking distance.

    Two organic facets are linked if within `link` metres; a connected component
    is one physical organic locus. n_organic / n_loci is the pseudo-replication
    factor, i.e. how many facets the classification spends per independent locus.
    """
    org_pos = pos[organic]
    n = len(org_pos)
    tree = cKDTree(org_pos)
    n_loci, factor = [], []
    for link in link_distances:
        pairs = tree.query_pairs(link, output_type="ndarray")
        if len(pairs):
            m = csr_matrix(
                (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
            )
            ncomp, _ = connected_components(m + m.T, directed=False)
        else:
            ncomp = n
        n_loci.append(int(ncomp))
        factor.append(float(n / ncomp))
    return n_loci, factor


def campaign_extent_simulation(
    pos: np.ndarray,
    organic: np.ndarray,
    radii=PATCH_RADII_M,
    n_pointings=CAMPAIGN_N_POINTINGS,
    trials: int = CAMPAIGN_TRIALS,
    seed: int = 42,
) -> tuple[dict[float, list[float]], list[float], list[float]]:
    """P(>=1 organic | n distinct pointings confined to a patch of radius D).

    Sampling frame: the classified facets ARE the resolvable targets (there is no
    facet-area column, and Recon-A coverage is targeted rather than global), so this
    is "n distinct classified facets drawn within radius D", not random surface area.

    Pointings are UNIQUE planned targets: drawn WITHOUT replacement, capped at the
    number of facets in the patch (you cannot usefully point at the same resolvable
    spot twice). Hence for a small patch the detection probability approaches
    P(patch contains any organic facet) and cannot be inflated by taking more
    pointings than the patch holds -- which is the saturation the figure reports.

    Each scenario (patch radius x pointing count, and the global control) uses an
    independent Generator seeded by SeedSequence(seed).spawn, so adding or reordering
    a scenario does not perturb the others. Also returns the globally-dispersed
    control and the analytic independent-Bernoulli value 1-(1-f_org)^n.
    """
    ss = np.random.SeedSequence(seed)
    center_rng = np.random.default_rng(ss.spawn(1)[0])
    tree = cKDTree(pos)
    n_facets = len(pos)
    f0 = float(organic.mean())

    centers = center_rng.choice(n_facets, size=min(trials, n_facets), replace=False)
    patches = tree.query_ball_point(pos[centers], max(radii))
    patch_by_center = []
    for c, nb in zip(centers, patches):
        nb = np.asarray(nb)
        d = np.linalg.norm(pos[nb] - pos[c], axis=1)
        order = np.argsort(d)
        patch_by_center.append((nb[order], d[order]))

    # One independent child seed per (radius, count) scenario plus the global control.
    n_scen = len(radii) * len(n_pointings) + len(n_pointings)
    children = iter(ss.spawn(n_scen))

    p_detect: dict[float, list[float]] = {}
    for D in radii:
        row = []
        for n in n_pointings:
            rng = np.random.default_rng(next(children))
            hits = valid = 0
            for nb, d in patch_by_center:
                idx = nb[d <= D]
                if idx.size == 0:
                    continue
                take = min(n, idx.size)  # distinct pointings, capped at patch size
                pick = rng.choice(idx, size=take, replace=False)
                valid += 1
                if organic[pick].any():
                    hits += 1
            row.append(hits / valid if valid else float("nan"))
        p_detect[float(D)] = row

    global_row = []
    for n in n_pointings:
        rng = np.random.default_rng(next(children))
        hits = sum(
            bool(organic[rng.choice(n_facets, size=n, replace=False)].any())
            for _ in range(trials)
        )
        global_row.append(hits / trials)

    analytic = [1.0 - (1.0 - f0) ** n for n in n_pointings]
    return p_detect, global_row, analytic


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Naive binomial Wilson interval, treating every facet as independent."""
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def prevalence_block_bootstrap(
    df: pd.DataFrame, block_deg: float = PREVALENCE_BLOCK_DEG,
    n_boot: int = PREVALENCE_N_BOOT, seed: int = 42,
) -> dict:
    """Clustering-aware CI on the organic-dominant prevalence f_org.

    This is a facet-weighted cluster (block) bootstrap: it resamples whole lat-lon
    blocks (each larger than the organic decorrelation length, hence approximately
    independent) with replacement, pools their facets, and recomputes the ratio
    f_org = organic / total. It therefore targets the same facet-weighted prevalence
    the paper reports, but over the OVIRS-covered facets only. It is NOT a sampling
    interval for all of Bennu: Recon-A covers ~13% of the surface and is targeted,
    so this quantifies clustering-driven uncertainty within the mapped region, not
    global surface prevalence. The naive Wilson interval treats all facets as
    independent and is spuriously tight; the design effect is the ratio of
    block-bootstrap to Wilson interval widths.
    """
    organic = (df["class"].to_numpy() == "organic")
    lat = df["latitude_deg"].to_numpy()
    lon = df["longitude_deg"].to_numpy()
    n = len(df)
    k = int(organic.sum())

    wlo, whi = _wilson_ci(k, n)

    block = ((lat + 90) // block_deg).astype(int) * 100000 + (lon // block_deg).astype(int)
    ublocks = np.unique(block)
    idx_by_block = {b: np.flatnonzero(block == b) for b in ublocks}

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(ublocks, size=len(ublocks), replace=True)
        facets = np.concatenate([idx_by_block[b] for b in pick])
        boots[i] = organic[facets].mean()
    blo, bhi = np.percentile(boots, [2.5, 97.5])

    wilson_w = whi - wlo
    return dict(
        f_organic=float(organic.mean()),
        wilson_ci=[float(wlo), float(whi)],
        block_ci=[float(blo), float(bhi)],
        n_blocks=int(len(ublocks)),
        block_deg=float(block_deg),
        design_effect=float((bhi - blo) / wilson_w) if wilson_w > 0 else float("nan"),
    )


def leave_one_site_out(df: pd.DataFrame, sites=SAMPLE_SITES, exclude_radius_deg=20.0) -> dict:
    """f_org recomputed with each candidate sample site's neighbourhood removed.

    Recon-A coverage concentrates on four sites chosen for scientific interest, so a
    reviewer will ask whether the prevalence is a selection artifact. If f_org is
    stable as each site is dropped, it is not driven by any single site.
    """
    lat = df["latitude_deg"].to_numpy()
    lon = df["longitude_deg"].to_numpy()
    organic = (df["class"].to_numpy() == "organic")

    def angsep(la, lo, la2, lo2):
        p1, p2 = np.radians(la), np.radians(la2)
        dl = np.radians(lo - lo2)
        return np.degrees(np.arccos(np.clip(
            np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1)))

    out = {"all_sites": float(organic.mean())}
    for name, (la, lo) in sites.items():
        keep = angsep(lat, lon, la, lo) >= exclude_radius_deg
        out[f"drop_{name}"] = float(organic[keep].mean())
    return out


def analyze(df: pd.DataFrame, seed: int = 42) -> PointingReport:
    """Full spatial-sampling analysis on a classified facet dataframe."""
    organic = (df["class"].to_numpy() == "organic")
    pos = facet_positions_m(df)
    f0 = float(organic.mean())

    counts, p_org, lift = correlogram(pos, organic)
    dlen = decorrelation_length(LAG_EDGES_M, lift)
    n_loci, factor = loci_census(pos, organic)
    p_detect, global_row, analytic = campaign_extent_simulation(
        pos, organic, seed=seed
    )
    prev = prevalence_block_bootstrap(df, seed=seed)
    loso = leave_one_site_out(df)

    log.info(
        "Correlogram: lift %.1fx at %s m -> decorrelation ~%.0f m",
        lift[0],
        f"{LAG_EDGES_M[0]:.0f}-{LAG_EDGES_M[1]:.0f}",
        dlen,
    )
    i5 = LINK_DISTANCES_M.index(5.0)
    log.info(
        "Loci census: %d organic facets -> %d loci at 5 m (%.1fx pseudo-replication)",
        int(organic.sum()),
        n_loci[i5],
        factor[i5],
    )
    log.info(
        "Campaign: patch D=5 m saturates at P=%.2f (n=%d..%d); global-dispersed reaches %.2f",
        p_detect[5.0][-1],
        CAMPAIGN_N_POINTINGS[0],
        CAMPAIGN_N_POINTINGS[-1],
        global_row[1],
    )
    log.info(
        "Prevalence f_org=%.2f%%: Wilson [%.2f, %.2f] vs clustering-aware block "
        "[%.2f, %.2f] (%.1fx wider, %d blocks)",
        100 * prev["f_organic"], 100 * prev["wilson_ci"][0], 100 * prev["wilson_ci"][1],
        100 * prev["block_ci"][0], 100 * prev["block_ci"][1],
        prev["design_effect"], prev["n_blocks"],
    )
    loso_spread = max(loso.values()) - min(loso.values())
    log.info(
        "Leave-one-site-out: f_org spans %.2f-%.2f%% (%.2f pt), stable under site removal",
        100 * min(loso.values()), 100 * max(loso.values()), 100 * loso_spread,
    )

    return PointingReport(
        n_facets=len(df),
        n_organic=int(organic.sum()),
        f_organic=f0,
        lag_edges_m=list(LAG_EDGES_M),
        correlogram_pairs=counts,
        correlogram_p_organic=p_org,
        correlogram_lift=lift,
        decorrelation_length_m=dlen,
        link_distances_m=list(LINK_DISTANCES_M),
        n_loci=n_loci,
        pseudo_replication_factor=factor,
        patch_radii_m=list(PATCH_RADII_M),
        campaign_n_pointings=list(CAMPAIGN_N_POINTINGS),
        campaign_p_detect=p_detect,
        campaign_global_p_detect=global_row,
        campaign_analytic_p_detect=analytic,
        f_organic_wilson_ci=prev["wilson_ci"],
        f_organic_block_ci=prev["block_ci"],
        prevalence_block_deg=prev["block_deg"],
        prevalence_n_blocks=prev["n_blocks"],
        prevalence_design_effect=prev["design_effect"],
        leave_one_site_out=loso,
    )
