# Orbital Raman: instrument and observing feasibility

**Project owner and maintainer: Sharahil Jarjis.**

A research software project exploring the photon budget and spatial sampling requirements for Raman spectroscopy of carbonaceous asteroid surfaces, with Bennu as the motivating case.

**Other current projects:** [research portfolio](RESEARCH_PORTFOLIO.md).

The repository combines a time-gated Raman count model, uncertainty studies, and simulations of how pointing placement changes target interception. It is a **simulation and methods study**. Its outputs are conditional on the optical, material and surface assumptions supplied to the model.

## Start here

- [Quick start and reproduction](docs/REPRODUCING.md): installation, a data-free example, tests, and optional mission-data workflows.
- [Methods and limitations](docs/METHODS_AND_LIMITATIONS.md): what the calculations mean and the assumptions that need validation.
- [Development and data notes](docs/PROVENANCE.md): source revision, data credits and tool use.
- [Project attribution](AUTHORSHIP.md): ownership attribution and development-note references.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_check.py
python -m pytest -q
```

The smoke check runs without mission or laboratory data. Tests that need external data or previously generated artifacts skip when those inputs are absent. The dependency versions record the environment used for this release's checks; see [reproduction notes](docs/REPRODUCING.md) for the verification scope.

## What is included

| Component | Entry point | Purpose |
| --- | --- | --- |
| Photon budget | `src_fixed.py` | Signal, reflected-solar background, detector noise, environmental assumptions and sensitivity calculations |
| Minimal example | `scripts/smoke_check.py` | Print a count budget and check basic numerical behavior using explicit scenario inputs |
| OVIRS scenario analysis | `main.py` | Parse a specified Bennu band-area product, apply the historical proxy model, and run Monte Carlo scenarios |
| Pointing campaign | `pointing_main.py` | Spatial correlation, clustering-aware prevalence uncertainty and campaign simulations from generated facet classifications |
| Coherence calculations | `g1_main.py` | Compare simplified first- and second-order photon-correlation scenarios |
| Optional shape analysis | `analysis/occlusion_raytrace.py` | Investigate line-of-sight occlusion with externally supplied Bennu meshes |

The current source incorporates the later local work on active-illumination geometry, reflected-solar collection geometry, pointing campaigns, and explicit Raman-coefficient sensitivity. The original scientific behavior is retained so the calculations can be inspected and compared with their history.

## How to interpret the results

- **Detection is a model output.** An SNR above a selected threshold does not demonstrate flight readiness or measured detection at Bennu.
- **A band-area score is not an organic-abundance measurement.** The historical `wt_c_pct`, `organic` and `carbonate` names in the OVIRS pipeline are retained for compatibility. They denote a scenario proxy and threshold labels; the integrated infrared band does not uniquely identify those chemical classes.
- **Spatial sampling and photon collection are separate questions.** A detectable spectrum is useful only if the beam samples the relevant material. Campaign simulations currently use proxy-defined targets and require comparison with independently supported labels and cost-matched targeting policies.
- **Laboratory and deep-UV validation remain separate.** Existing visible Raman work motivates follow-up, but it does not measure the assumed deep-UV gain, photodamage limits or orbital performance. Unpublished laboratory inputs and third-party laboratory pipelines are not bundled.

See [methods and limitations](docs/METHODS_AND_LIMITATIONS.md) before using generated tables or prose in a report. Several historical writers emit manuscript-style text with stronger chemical interpretations than the evidence supports; those outputs are reproducibility artifacts and require scientific revision.

## Project status

The instrument and mission model is available for inspection and self-contained numerical checks. Full OVIRS reproduction requires the identified external product. The next research step is to test target-selection stability and the incremental value of Raman against published full-spectrum infrared baselines, followed by matched laboratory measurements with complete acquisition metadata.

Earlier figures, manuscript drafts and interactive demos are retained in `archive/2026-04/` for history. They are not current validated results. Some previously published mission files remain under `Data/`; see its provenance notes before use.

Source credits and third-party data and software terms are recorded in [provenance](docs/PROVENANCE.md).
