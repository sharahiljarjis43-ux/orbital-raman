# Research status

Updated October 24, 2026.

## Verified within the stated model

- The rational three-partner construction gives two valid reversible generators with equal scalar covariance at every lag. The exact algebra shows a labeled return change `P′₂₂−P₂₂ = 152/1029` and the nonzero sixth-order Ramsey witness.
- A nonlinear classical two-time moment separates the same pair: the difference in the third lag derivative of `E[f(X₀)² f(X_t)]` is `−76/21`. Higher-order information is not unique to quantum readout.
- Occupancy-only readout is blind to encounter-network changes in this homogeneous-rate model. Distinct tags are an essential modeling assumption.
- Current deterministic checks reproduce the saved expected-data KL contrasts and validate the numerical propagation engine against direct matrix exponentiation.

## Exploratory mathematical extension

The [moment-order note](moment-order-proof.md) develops the worst-case `2n−2` threshold for the full hierarchy of exact stationary scalar moments with calibrated distinct tags. The upper argument reconstructs the generator; the lower argument constructs hidden-mode perturbations whose lower-degree moments coincide at all times. This derivation and its checks came from AI-assisted sessions. It has not been independently refereed, formally verified, or established as novel.

"Moment order" is total scalar polynomial degree, including powers at a repeated observation time. It is not the number of distinct time points or the number of shots. Exact access to this hierarchy does not establish that the finite qubit library recovers it.

The broader notes also explored a known-trace improvement, rich-control classification, and resource bounds. Those extensions are not asserted by this compact package. The included general moment-order proof should be reviewed before expanding the public claims.

## Open questions

1. Which return quantities are globally identifiable from a precisely fixed finite library with unknown kinetics and readout parameters?
2. Which nuisance parameters are actually calibrated, and with what observation model and uncertainty?
3. How stable is inversion when tags nearly coincide or a mode becomes nearly invisible?
4. What finite-data guarantees hold over a declared compact parameter class?
5. Can a real system supply distinct partner tags and the assumed shared calibration response?
6. Which model-specific statements are already consequences of earlier hidden-process or realization results?

No claim is made here of a laboratory demonstration, tau sensing, arbitrary-network reconstruction, optimal shot budget, or quantum advantage.
