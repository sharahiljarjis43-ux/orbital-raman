# Methods and limitations

## Photon budget

`src_fixed.py` calculates a detected Raman signal and combines shot, reflected-solar, dark and read-noise contributions. The current source separates active laser illumination from solar incidence: solar shadow alone does not suppress an actively illuminated Raman signal. It also applies telescope collection geometry to the reflected-solar background. The solar-budget tests check the collection fraction against an independently derived geometric limit.

Raman coefficient, concentration-like scaling, wavelength dependence, resonance enhancement, collection efficiency, detector settings, gate width, roughness, dust and unresolved relief remain model assumptions. `analysis/kr_anchor.py` exposes coefficient scenarios and consistency arguments. A literature-based bound or dimensional check is not an in-situ calibration, and a central value must not be presented as measured Bennu performance.

The model includes optional squeezing and photon-correlation explorations. Their simplified analytical comparisons do not establish a practical instrument advantage. Timing, coherence, throughput, loss, calibration and implementation would need separate assessment.

## Infrared proxy and chemical interpretation

The OVIRS pipeline uses an integrated 3.2–3.6 µm band-area product. It preserves a historical linear scale named `wt_c_pct` and threshold classes named `organic`, `carbonate` and `weak`. Those fields must be read as **scenario variables**, not independently validated chemical abundance or mineral labels. The original thresholds were selected to resemble aggregate literature fractions; matching those fractions does not identify the same observations.

A single integrated band can include multiple chemical and measurement contributions. Full-spectrum infrared methods already discriminate some organic-like and carbonate spectra. Future comparisons should include those methods, ambiguous and rejected observations, coordinate/footprint uncertainty, and independently supported chemical labels. Covariate regression can expose score sensitivity; it cannot establish a chemically purified organic map.

The product is a facet representation of Recon A coverage, with neighboring facets and repeated observation contributions. It is not a set of independent chemical samples or a complete global census. The pointing module adds block-bootstrap and site-removal sensitivity analyses, but those do not validate the target labels or remove all sampling and footprint uncertainty.

## Spatial campaign interpretation

The campaign module studies how interception varies with clustering and pointing placement. Results are conditional on the proxy map, selected correlation scale, surface geometry and pointing policy. A hit is an intersection with a proxy-defined target, not necessarily successful chemical identification. A useful policy comparison must hold observing cost fixed and compare against policies that use the infrared information actually available.

## Laboratory follow-up

Local visible Raman and infrared analyses are distinct evidence streams and are not released as a validated end-to-end mission calibration. The analyzed Raman collection uses 514.6 nm excitation. Planned 405 nm measurements are not deep-UV measurements; comparisons across different meteorites do not isolate wavelength. Deep-UV yield, fluorescence separation, optical losses and photodamage constraints need appropriate measurements or explicit uncertainty.

D/G band parameters depend on baseline and line-shape choices. A published thermometry relation cannot be transferred to a different fit protocol without checking its validity. D/G structure does not measure total organic abundance. Existing infrared sample-form contrasts also confound preparation, session, material and geometry; they do not alone prove a causal surface-scattering mechanism or a universal contamination source.

## Historical generated reports

The code preserves older report writers for traceability. Some generated sections use phrases such as “organic-dominant”, “validation” and “wt% C” more strongly than the present evidence supports. They also include historical numerical examples and literature interpretations. Treat these as legacy calculation outputs requiring revision, not submission-ready scientific conclusions. This packaging update revises the public explanation while preserving calculation behavior.

## Selected scientific context

- [Ferrone et al. (2021)](https://doi.org/10.1016/j.icarus.2021.114579), infrared spectral-shape classification.
- [Kaplan et al. (2020)](https://doi.org/10.1126/science.abc3557), Bennu carbonate observations.
- [Kaplan et al. (2021)](https://doi.org/10.1051/0004-6361/202141167), interpretation of Bennu organic-like infrared features.
- [Busemann et al. (2007)](https://doi.org/10.1111/j.1945-5100.2007.tb00581.x), Raman characterization of meteoritic insoluble organic matter.

These sources provide context and comparators. Their inclusion does not imply endorsement or validation of this repository's scenarios.
