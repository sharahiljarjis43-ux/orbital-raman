# Exact Ramsey witness for the three partner binding model

For the fixed three partner pair in the revised quantum binding proposal, the scalar covariance is identical at every lag, but the complex Ramsey coherence differs. The separation can be proved with exact arithmetic: for fixed real coupling `g`, the alias minus base difference is

```text
ΔW(T) = −(19/1890) i g³ T⁶ + O(T⁷),  as T → 0.
```

Thus every fixed `g ≠ 0` separates this pair for sufficiently small positive `T` in the ideal, exactly calibrated model. This proves fixed pair separation. It does not establish global identification of a return network, finite measurement performance, physical tau sensing or scientific novelty.

The construction is retained from the October 2026 model notes. The exact derivative calculation extends their numerical sanity check. These extensions were generated during AI-assisted sessions; see [provenance](provenance.md). Independent mathematical review remains appropriate before a publication claim.

## Model and conventions

Use row generators, state order `(B₁,B₂,B₃,U₁,U₂,U₃)`, association and dissociation rates `a = d = 1`, and

```text
E = [ −21/20      1       1/20 ]
    [    1      −2         1  ]
    [  1/20      1      −21/20]

c = (−1, −2, 3)ᵀ,       w = (5, −4, −1)ᵀ,
R = I − wwᵀ/21,
E′ = R E Rᵀ,

Q  = [ −I       I   ],   Q′ = [ −I       I    ],
     [  I     E − I ]        [  I     E′ − I  ]

f = (−1, −2, 3, 0, 0, 0)ᵀ,
D = diag(f),
π = (1, 1, 1, 1, 1, 1)/6.
```

All off-diagonal rates of `E` and `E′` are strictly positive. Both `Q` and `Q′` have zero row sums and are symmetric, so the displayed `π` is stationary and satisfies detailed balance. Both bound and unbound aggregate dwell laws are exponential with rate one; internal unbound transitions do not alter the constant unit binding hazard.

For Ramsey control `y(t) = 1`, set `z = −i g`. The coherences are

```text
W(T)  = π exp[(Q  + zD)T] 1,
W′(T) = π exp[(Q′ + zD)T] 1.
```

All differences below mean alias minus base: `ΔW = W′ − W`.

## Exact covariance equality

Since `wᵀ1 = wᵀc = 0` and `wᵀw = 42`, the matrix `R` is orthogonal and fixes both `1` and `c`. Set `S = diag(R,R)`. Then

```text
Q′ = S Q Sᵀ,    Sf = f,    (πD)S = πD.
```

The mean detuning is zero. Consequently the full scalar covariance agrees for every lag `t ≥ 0`:

```text
C′(t) = πD exp(Q′t)f
      = πD S exp(Qt) Sᵀf
      = πD exp(Qt)f
      = C(t).
```

The one-time detuning distributions also agree because both models use the same stationary probabilities and coupling values. Equality of covariance does not imply equality of all two-time joint distributions or higher moments.

The time-integrated partner return matrix is `P = (I−E)⁻¹`, and `P′ = R P Rᵀ`. The exact audit confirms `P′₂₂ − P₂₂ = 152/1029` (approximately `0.147716`). The mean same-partner return `Tr(P)/3 = 145/252` is invariant. The changing labeled entries, rather than this invariant average, are the return targets.

## First unequal Ramsey derivative

Write

```text
Aₘ(z) = π(Q + zD)ᵐ1.
W(T) = Σₘ Aₘ(z) Tᵐ/m!.
```

Stationarity gives `πQ = 0`, and probability conservation gives `Q1 = 0`. Hence any nonzero word contributing to `Aₘ` must start and end with `D`, unless it is the order zero term.

Words containing two factors of `D` are derivatives of the covariance and therefore agree for the pair. Words with no `Q` depend only on the common one-time distribution. A segment connecting two bound states first depends on `E` at three powers of `Q`:

```text
(Q)BB   = −I,
(Q²)BB  = 2I,
(Q³)BB  = E − 4I.
```

Therefore words containing at least three factors of `D` cannot distinguish the models before total order six. At order six, the only new terms are the two words

```text
z³ π D² Q³ D 1   and   z³ π D Q³ D² 1.
```

The other distributions of three `Q` factors among the `D` factors involve only the first two bound blocks above. Contributions containing two `D` factors still agree by covariance equality. Contributions with four or more `D` factors have at most two `Q` factors and cannot involve `E`.

Let `h = c ⊙ c = (1,4,9)ᵀ`. Since the stationary weight of each bound state is `1/6` and `E` is symmetric, the unequal coefficient is

```text
ΔA₆(z) = (z³/6) Δ[hᵀEc + cᵀEh]
        = (z³/3) Δ[hᵀEc].
```

Using `Rc = c` and `Rᵀ = R`,

```text
Δ[hᵀEc] = hᵀ(E′−E)c
         = −(hᵀw)(wᵀEc)/21.

hᵀw = −20,        wᵀEc = −114/5,
Δ[hᵀEc] = −152/7.
```

It follows exactly that

```text
ΔAₘ(z) = 0                 for m = 0,1,2,3,4,5,
ΔA₆(z) = −(152/21) z³.
```

Divide by `6! = 720` and substitute `z³ = (−ig)³ = i g³`:

```text
ΔW(T) = −(152/(21×720)) z³ T⁶ + O(T⁷)
       = −(19/1890) i g³ T⁶ + O(T⁷).
```

At `g = 1/2`, the sixth derivative is `−(19/21)i` and the coefficient of `T⁶` is `−(19/15120)i`. These are derivatives at zero, not estimates at the proposal's finite setting `T = 4`. With the proposal convention `Im W = −⟨σy⟩`, the leading difference in the measured y quadrature has the opposite sign.

Finite matrix exponentials are analytic in `T`. The nonzero leading coefficient therefore proves a nonzero coherence difference for sufficiently small positive `T` at every fixed nonzero real `g`. The term is cubic in `g`; since the first and second integrated phase cumulants agree, it supplies an exact third-order witness of this pair's difference.

## Reproduce the exact audit

From the project root, create `rerun/`, then run the standard-library script:

```text
python3 scripts/verify_exact_witness.py --output rerun/exact_witness_checks.json
```

The script checks the actual rational `E`, `R`, `c`, stationary distribution, positivity, covariance-conjugacy ingredients, return matrices and every polynomial coefficient through the sixth derivative. It performs no stochastic or finite measurement simulation and uses no floating-point exponentiation. The saved JSON records the exact fractions and checks.

## Conditions and interpretation

This result assumes the stated generator family, stationary preparation, distinct known partner couplings, ideal pure dephasing and ideal Ramsey evolution. Unknown kinetic and sensor parameters can create further observational ambiguities; they were not profiled in this calculation. If the coupling observes only bound versus unbound occupancy, that aggregate process is independent of the encounter generator in this model, and this witness is unavailable.

The model's immediate physical interpretation is engineered replay with imposed partner tags. Distinct natural couplings in an actual tau system require a separate transduction and calibration argument. The result makes no quantum advantage claim. Its significance relative to earlier hidden-process and non-Gaussian spectroscopy results requires a separate prior-art assessment.
