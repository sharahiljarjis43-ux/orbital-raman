"""Orchestrator for the spatial-sampling reframe (pointing campaign + K_R bounds).

Produces:
  results/numerical_results_pointing.json  -- correlogram, loci census, campaign sim, K_R bounds
  results/fig_pointing_campaign.{png,pdf}   -- 3-panel spatial-sampling figure
  results/fig_kr_boundary.{png,pdf}         -- detection boundary vs K_R
  results/section_pointing.md               -- paste-ready prose for the reframed sections

Usage:
    python pointing_main.py --output-dir results/

Deterministic (numpy seed 42). Reads the classified facet product written by
main.py; run main.py first if results/intermediate/classifications.parquet is absent.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src_fixed  # noqa: E402
from analysis import kr_anchor, make_pointing_figures, pointing_campaign  # noqa: E402

log = logging.getLogger(__name__)

CLASSIFICATIONS = "results/intermediate/classifications.parquet"


def _setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(log_path, mode="w")],
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


def write_section_markdown(pr, kr, path: Path) -> None:
    """Paste-ready prose fragments for the reframed sections, numbers filled in."""
    i5 = pr.link_distances_m.index(5.0)
    b = kr.detection_boundary
    central = f"{kr.kr_central:.0e}"
    lift0 = pr.correlogram_lift[0]
    dsmall = pr.campaign_p_detect[5.0]
    snr_lo = b[f"{kr.kr_low:.0e}"][375]
    snr_hi = b[f"{kr.kr_high:.0e}"][375]
    snr_c = b[central][375]
    tight_lo, tight_hi = kr.standoff["literature_tightened_range_532"]
    txt = f"""### Spatial-sampling results (pointing campaign)

Organic-dominant facets are strongly clustered. The organic-neighbor lift,
P(neighbor organic | center organic) normalized by the global prevalence
f_org = {100*pr.f_organic:.2f}%, is {lift0:.0f}x within {pr.lag_edges_m[1]:.0f} m, falls
below 2x by {pr.decorrelation_length_m:.0f} m, and approaches unity by roughly 32 to 64 m,
giving a decorrelation length of order {pr.decorrelation_length_m:.0f} m. This structure is
invisible to the 5-degree Moran's I of Section 4.4.5, whose grid is coarser than the
correlation length by more than an order of magnitude.

The {pr.n_organic} organic-dominant facets form only {pr.n_loci[i5]} connected loci at a
5 m linking distance, a pseudo-replication factor of {pr.pseudo_replication_factor[i5]:.0f}.
The effective number of independent organic detections in the map is therefore of order
{pr.n_loci[i5]}, not {pr.n_organic}.

Because the facets are clustered, a binomial confidence interval that treats them as
independent is spuriously tight. A facet-weighted cluster bootstrap (resampling
{pr.prevalence_n_blocks} lat-lon blocks of {pr.prevalence_block_deg:.0f} degrees, each
larger than the decorrelation length, then pooling their facets) gives f_org =
{100*pr.f_organic:.2f}% with a 95% interval of [{100*pr.f_organic_block_ci[0]:.2f}%,
{100*pr.f_organic_block_ci[1]:.2f}%], about {pr.prevalence_design_effect:.0f} times wider
than the naive Wilson interval [{100*pr.f_organic_wilson_ci[0]:.2f}%,
{100*pr.f_organic_wilson_ci[1]:.2f}%], with the design effect consistent with the
per-locus pseudo-replication above. This interval quantifies clustering-driven uncertainty
over the OVIRS-covered facets; it is not a sampling interval for the whole surface, since
Recon-A covers about 13% of Bennu and is targeted rather than uniform. The estimate is at
least not dominated by any single OSIRIS-REx candidate sample site: removing any one site's
neighbourhood leaves f_org between {100*min(pr.leave_one_site_out.values()):.2f}% and
{100*max(pr.leave_one_site_out.values()):.2f}%.

Because the loci are about {pr.decorrelation_length_m:.0f} m across, a pointing campaign
confined to a single correlation cell saturates far below certainty. In a simulation where
each pointing is a distinct classified facet drawn without replacement from within a patch
(the classified facets are the resolvable targets; Recon-A coverage is targeted rather than
global, so this is a random-facet rather than random-area frame), restricting the pointings
to a 5 m-radius patch holds detection probability near {dsmall[0]:.2f}-{dsmall[-1]:.2f}
whether {pr.campaign_n_pointings[0]} or {pr.campaign_n_pointings[-1]} pointings are used:
extra pointings inside one correlation cell add nothing. Dispersing the same pointings across
the surface recovers independent-Bernoulli behavior, reaching {pr.campaign_global_p_detect[1]:.2f}
at {pr.campaign_n_pointings[1]} pointings. The operational rule is to space pointings by at
least the decorrelation length and to cover many loci rather than to dwell.

### Detection feasibility as a bounded requirement (K_R)

The Raman signal scales linearly with the backscatter coefficient K_R; the SNR, being
N_sig / sqrt(N_sig + N_bg), does not. Rather than assert a single K_R, we bound it. Measured
532 nm Raman cross-sections of disordered and aromatic carbon (Le, Lefumeux and Pino 2017)
are large, so organic carbon is not orders of magnitude weaker than the mineral targets of
the cited standoff demonstrations; combined with abundance dilution and a dark-matrix escape
penalty this places K_R for 1 wt% organic carbon at 532 nm in a plausible range of
{tight_lo:.0e} to {tight_hi:.0e} sr^-1, central near {central} sr^-1, with {kr.kr_high:.0e} a
loose ceiling. An independent inversion of the standoff demonstrations of Misra et al. (2012)
and Acosta-Maeda et al. (2016) is consistent with this central value, though it does not
tighten it because neither paper reports a numeric SNR. Across the conservative outer range
{kr.kr_low:.0e} to {kr.kr_high:.0e} sr^-1, detection at 375 m (Reconnaissance) spans SNR {snr_lo:.1f}
at the pessimistic bound to {snr_hi:.0f} at the optimistic bound, and is {snr_c:.0f} at the
central estimate. Reconnaissance detection is therefore met at the central and optimistic
values and falls below the SNR = 5 threshold only at the pessimistic bound; Orbital B (680 m)
reaches SNR {b[central][680]:.1f} at the central estimate. The feasibility conclusion is a
bounded requirement, not a single-point claim.
"""
    path.write_text(txt)
    log.info("Wrote %s", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _setup_logging(out / "pointing_run.log")

    cls_path = PROJECT_ROOT / CLASSIFICATIONS
    if not cls_path.exists():
        log.error("Missing %s -- run main.py first.", cls_path)
        sys.exit(1)
    df = pd.read_parquet(cls_path)
    log.info("Loaded %d classified facets", len(df))

    pr = pointing_campaign.analyze(df, seed=args.seed)
    kr = kr_anchor.analyze(src_fixed.calculate_snr_full)

    # JSON before figures, so the numeric record survives a headless run.
    payload = {"pointing": asdict(pr), "kr_bounds": asdict(kr)}
    jpath = out / "numerical_results_pointing.json"
    jpath.write_text(json.dumps(_to_serializable(payload), indent=2))
    log.info("Wrote %s", jpath)

    make_pointing_figures.make_pointing_figure(pr, out)
    make_pointing_figures.make_kr_figure(kr, out)

    write_section_markdown(pr, kr, out / "section_pointing.md")
    log.info("Pointing / K_R pipeline complete.")


if __name__ == "__main__":
    main()
