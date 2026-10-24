# Model, observations, and numerical checks

## Binding model

There are `n` bound states `B_i` and `n` encounter states `U_i`. Row probability vectors evolve under

```text
Q = [ −d I       d I  ]
    [  a I     E−a I  ],    a,d > 0.
```

The encounter generator `E` is symmetric and irreducible, has zero row sums, and has nonnegative off-diagonal rates. At stationarity each bound state has weight `a/[n(a+d)]`; each encounter state has weight `d/[n(a+d)]`. The next-binding partner matrix is `P = a(aI−E)⁻¹`. The labeled entry `P_ij` means the next binding partner is `j`, given the current partner `i` when unbinding begins.

The signal is `f=(c,0)`, with distinct, prescribed partner tags `c`. The benchmark uses `c=(−1,−2,3)` and the two rational three-by-three encounter generators in the [exact witness](exact-ramsey-witness.md). Both have strictly positive off-diagonal rates.

## Controlled coherence and readout

For sign control `y(t)`, the ideal coherence is the stationary row vector propagated through matrices `exp[(Q−i γ g y D) Δt]`, followed by summation over final states. `D=diag(f)`, `g` is prescribed coupling, and `γ` is unknown common gain. The implementation uses row-vector time ordering.

The synthetic readout is Bernoulli with

```text
p = b + (C/2) exp(−Γ T) Re[exp(i α − i δ A) W],
A = ∫ y(t) dt.
```

The fitted parameters are association `a`, dissociation `d`, encounter scale `λ`, common gain `γ`, detuning offset `δ`, effective contrast `C`, extra decay `Γ`, and baseline `b`. `C` combines visibility and readout attenuation; those two factors are not separately identified by this model.

The control grid contains Ramsey, Hahn echo, CPMG with two or four pulses, and two asymmetric two-pulse patterns. Times are `0.5,1,2,4,8`; prescribed gains are `0.1,0.5,1`; both quadratures are included. Individual intervals shorter than `0.05` are excluded. Definitions and full pulse timings are in `scripts/forward_model.py`.

Synthetic calibration uses deterministic zero or signed fields and zero-duration contrast anchors. It assumes those fields share the **same** gain, offset, decay, and readout response as tagged binding. No apparatus calibration has established this assumption.

## Expected-data nuisance profiles

The historical October 19 profiles fit the opposite fixed network shape to noiseless probabilities using eight deterministic starts and bounded binomial-deviance least squares. The design assigns expected weights totaling 80,000 science shots and, when calibration is present, 20,000 calibration shots. Fractional per-setting expected weights are permitted; these profiles are not binomial count experiments.

| Truth | Sensor parameters | True base → fit alias KL | True alias → fit base KL |
|---|---|---:|---:|
| Central rates | Known sensor | 1.7088 | 1.0904 |
| Central rates | Free sensor, science only | 1.5983 | 1.0170 |
| Central rates | Free sensor, shared synthetic calibration | 1.6569 | 1.0648 |
| Off-center rates | Known sensor | 0.4963 | 0.3756 |
| Off-center rates | Free sensor, science only | 0.4433 | 0.3498 |
| Off-center rates | Free sensor, shared synthetic calibration | 0.4813 | 0.3658 |

KL is the expected log-likelihood contrast against the **best alternative found**, at the stated allocation. These values illustrate close fitted responses and dependence on nuisance assumptions. They are neither a certified minimum nor a finite-shot error rate. These twelve profiles concern two declared truth points and restricted network shapes; they do not characterize every allowed model.

Full parameters and all starts are retained in [the historical results](../results/expected_data_profiles_20261019.json); the [frozen historical design](../results/deterministic-design.json) preserves the settings, ranges, allocation, runtime, and forward-model hash. The current forward check recomputes every displayed contrast from those saved parameters. A fresh optimization writes its own design, parameters, starts, and results.

## Verification

The exact audit uses `fractions.Fraction` and checks all derivative-polynomial coefficients through order six, generator positivity, stationarity, reflection identities, and the return-matrix difference. Numerical forward checks compare eigendecomposition against SciPy matrix exponentials at 12 seeded random parameter points for both hypotheses and every control setting. They also check covariance equality on a lag grid, occupancy-only blindness, the nonlinear classical comparator, and all saved expected-data contrasts.

The moment-order script checks lower-order agreement, a distinguishing threshold moment, explicit return change, reconstruction from exact mixed moments, and the return-integral formula for `n=2,…,6`. These finite numerical examples supplement the algebraic proof; they cannot establish its all-times statement by themselves. The conditioning of polynomial inversion worsens with `n`.
