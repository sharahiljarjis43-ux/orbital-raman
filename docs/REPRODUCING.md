# Reproducing the calculations

Run commands from the repository root. Python 3.10.5 and the package versions in `requirements.txt` were used for release verification. Installation into a newly created environment was not independently tested; the checks used the available environment with those versions.

## Data-free example

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
MPLBACKEND=Agg python scripts/smoke_check.py
MPLBACKEND=Agg python -m pytest -q
```

The example reports the signal and background count budget for explicitly assumed parameters at 375 m and 680 m. It checks that counts are finite, nonnegative, consistent across the two SNR entry points, and that the longer-range scenario gives lower SNR. This establishes executable numerical behavior for those inputs, not physical validation of the instrument.

The inherited tests include independent collection-geometry checks for the solar background, analytical coherence limits and Raman-coefficient sensitivity checks. Input-dependent tests skip if the required mission products, classifications or generated report files are missing. A passing source-only suite does not reproduce the spacecraft-data analysis.

## Coherence scenario

```bash
MPLBACKEND=Agg python g1_main.py --output-dir results/
```

This data-free workflow writes JSON, figures and historical report prose. Its coherence comparison is an analytical scenario, without a demonstrated interferometric instrument or measured operational advantage. Interpret it using [the limitations](METHODS_AND_LIMITATIONS.md).

## OVIRS and pointing scenarios

First obtain and identify the external FITS product described in [Data/README.md](../Data/README.md). Then run:

```bash
MPLBACKEND=Agg python main.py --fits Data/l_1600mm_sp_ovirs_reca_bandarea3200to3600nm_allsites_wavc_0000n00000.fits --output-dir results/ --seed 42 --n-trials 10000
MPLBACKEND=Agg python pointing_main.py --output-dir results/ --seed 42
```

`pointing_main.py` reads `results/intermediate/classifications.parquet` at its fixed repository-relative location, even if another output directory is selected. Use `results/` for the first command unless you explicitly relocate that intermediate file.

These workflows generate numerical summaries, intermediate tables, plots and manuscript-style sections. The chemical names and abundance scale in the historical pipeline are assumptions. Agreement with an aggregate literature fraction is not validation of individual target labels. Generated prose must be checked against [the current limitations](METHODS_AND_LIMITATIONS.md) before reuse.

## Optional external-mesh calculation

`analysis/occlusion_raytrace.py` takes two mesh paths: a Bennu global mesh and a local high-resolution mesh. Install its optional `trimesh` and `embreex` dependencies, obtain the meshes with their original provenance and terms, then run:

```bash
python analysis/occlusion_raytrace.py /path/to/global.obj /path/to/local.obj
```

The mesh workflow was not rerun for this packaging update. Its result depends on geometry, coordinate conventions, mesh coverage and the particular pointing assumptions.

## Reproducibility record

For a report, retain the source commit, Python and dependency versions, input identifiers and hashes, command line, seed and numerical JSON together. Use fresh output directories when comparing parameter settings. Report excluded and failed observations, and distinguish numerical reproduction from independent scientific validation.
