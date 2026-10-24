# Verification record

Checked 24 October 2026 using Python 3.10.5 and the versions recorded in `requirements.txt`.

| Check | Result | Scope |
| --- | --- | --- |
| Data-free count-budget example | Passed | Finite nonnegative counts, SNR entry-point agreement and range ordering for two explicit scenarios |
| Inherited source-only test suite | 28 passed, 19 skipped | Geometry, analytical limits and sensitivity behavior; external-data and unbundled-artifact cases skipped |
| Python compilation | Passed | All included source, example and test files parse |
| `g1_main.py` headless execution | Passed | Produced numerical JSON, PNG/PDF figures and historical report text in a separate verification folder |
| Scientific-source comparison | Passed | All 18 calculation and entry-point files match source commit `cf215304f26567e5449b3bb67f542c82bf8719be` byte for byte |
| Credential-pattern scan of new package | No matches | Checked for common GitHub token and private-key signatures; this is a limited scan |

The exact checks were:

```bash
MPLBACKEND=Agg python scripts/smoke_check.py
MPLBACKEND=Agg python -m pytest -q
python -m compileall -q src_fixed.py main.py g1_main.py pointing_main.py analysis scripts tests
MPLBACKEND=Agg python g1_main.py --output-dir /separate/verification/folder
```

The source-only suite completed in approximately 24 seconds. The smoke example's assumed 5% scenario returned SNR approximately 22.27 at 375 m and 7.45 at 680 m. These numbers are a numerical replay reference under the code's assumptions, not measured target abundance or validated flight performance.

No fresh full OVIRS pipeline replay, external-mesh ray trace, unpublished laboratory-data refit, clean-environment installation or hardware validation was performed for this update. The generated coherence report was checked for execution, not adopted as submission-ready prose. See [methods and limitations](METHODS_AND_LIMITATIONS.md).
