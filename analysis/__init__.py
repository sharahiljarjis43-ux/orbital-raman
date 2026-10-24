"""Bennu OVIRS heterogeneity analysis package (Plan A).

Modules build the per-facet 3.42 μm band-area distribution, calibrate it to
wt% organic carbon, validate against Ferrone et al. (2021), and produce a
revised Monte Carlo prior for the Section 5 photon budget.

References
----------
Ferrone, S. M., et al. (2021), Icarus 368, 114579,
    doi:10.1016/j.icarus.2021.114579 — Visible-near-infrared observations of
    organics and carbonates on (101955) Bennu: Classification method and
    search for surface context. Source of the 245/544/15585 KS-classifier
    headline statistics.
Simon, A. A., et al. (2020), Science 370, abc3522,
    doi:10.1126/science.abc3522 — Widespread carbon-bearing materials on
    near-Earth asteroid (101955) Bennu. Documents the ubiquitous 3.4 μm
    feature and its variable mix of organics + carbonates.
Kaplan, H. H., et al. (2020), Science 370, abc3557,
    doi:10.1126/science.abc3557 — Bright carbonate veins on asteroid
    (101955) Bennu. Carbonate doublet identification.
Kaplan, H. H., et al. (2021), A&A 653, L1,
    doi:10.1051/0004-6361/202141167 — Composition of organics on asteroid
    (101955) Bennu. Reports H/C = 0.4-0.6 (upper limits) and notes Bennu
    band depths are consistent with 1-2.5 wt% C carbonaceous chondrites.
    Nominal OVIRS spectral SNR ≈ 50 in 3.1-3.6 μm.
Praet, A., et al. (2021), Icarus 363, 114427,
    doi:10.1016/j.icarus.2021.114427 — Hydrogen abundance estimation and
    distribution on (101955) Bennu (note: this paper covers the OH/H2O
    feature, NOT the 3.4 μm organic band).
Reuter, D. C., et al. (2018), Space Sci. Rev. 214, 54,
    doi:10.1007/s11214-018-0482-9 — The OSIRIS-REx Visible and InfraRed
    Spectrometer (OVIRS) instrument paper.
Lauretta, D. S., et al. (2024), Meteoritics & Planetary Science 60, 1545,
    doi:10.1111/maps.14227 — Asteroid (101955) Bennu in the laboratory.
    Reports bulk total carbon 4.5-4.7 wt%, of which ~90% is organic and
    ~10% is in carbonates (i.e. organic C ≈ 4.0-4.2 wt%).
Glavin, D. P., et al. (2025), Nature Astronomy 9, 199,
    doi:10.1038/s41550-024-02472-9 — Abundant ammonia and nitrogen-rich
    soluble organic matter in Bennu samples.
Sandford, S. A., et al. (2025), Nature Astronomy 9, 1803,
    doi:10.1038/s41550-025-02694-5 — Nitrogen- and oxygen-rich organic
    material indicative of polymerization in pre-aqueous cryochemistry on
    Bennu's parent body.
"""
__all__ = [
    "parse_ovirs",
    "calibration",
    "classify",
    "heterogeneity_stats",
    "revised_monte_carlo",
    "make_figures",
]
