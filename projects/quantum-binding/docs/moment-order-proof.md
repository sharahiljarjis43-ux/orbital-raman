# Stationary moment order in a reversible binding model

Exploratory AI-assisted derivation, October 22, 2026; numerical checks refreshed October 24. This note has been internally checked, but is not independently refereed or formally verified. Priority is unresolved. See [provenance](provenance.md) and [literature context](literature.md).

## Statement and information definition

Take `n≥2` known partners, distinct calibrated real labels `c_i`, positive association and dissociation rates `a,d`, and a symmetric irreducible encounter generator `E`. The signal and stationary row generator are

```text
f = (c,0),       Q(E) = [ −dI     dI  ],
                       [  aI   E−aI  ]
π_Bi = a/[n(a+d)],       π_Ui = d/[n(a+d)].
```

The return target is `P=a(aI−E)⁻¹`. Degree `r` information means **all** stationary moments

```text
E[∏ⱼ f(X_tⱼ)^pⱼ],       Σⱼ pⱼ ≤ r,
```

at all ordered observation times. Total scalar degree includes powers at repeated times. It is not the number of time points, binding events, or qubit shots.

For this declared model class, the proposed sharp worst-case degree for identifying the entire labeled `P` matrix is `2n−2`:

1. Exact moments through that degree recover `a,d,E`, and hence `P`.
2. At every lower degree, valid reversible models exist whose entire lower-degree moment hierarchies agree at all times while every entry of `P` differs.

This is a worst-case statement. Particular models and individual targets may require less information.

## Upper construction

Let `B(t)` be the bound-to-bound block of `exp(Qt)`, `J=11ᵀ/n`, and

```text
b(t) = a/(a+d) + d/(a+d) exp[−(a+d)t].
```

The homogeneous binding hazards imply `B(t)1=b(t)1`; symmetry also gives `1ᵀB(t)=b(t)1ᵀ`.

For `k=1,…,n−1`, set `m_k=n⁻¹Σᵢc_i^k` and `v_k=c^k−m_k1`, with componentwise powers. The matrix `V=[v₁ … v_(n−1)]` spans `1` perpendicular: distinct labels make the Vandermonde matrix `[1,c,…,c^(n−1)]` invertible.

Put `α=a/[n(a+d)]`. Mixed two-time moments obey

```text
M_kl(t) = E[f(X₀)^k f(X_t)^l]
        = α [v_kᵀ B(t) v_l + n m_k m_l b(t)].
```

Their maximum required total degree is `2n−2`. After subtracting the known constant-mode term, write `A(t)=VᵀB(t)V`. Then

```text
B(t) = V(VᵀV)⁻¹ A(t) (VᵀV)⁻¹Vᵀ + b(t)J.
```

Block multiplication gives

```text
B'''(0) = ad E − d(a+d)² I,
E = [B'''(0) + d(a+d)² I]/(ad).
```

The derivative is third order **in lag time**, not third-degree scalar moment information.

When the absolute tag scale is calibrated, degree two also recovers unknown `a,d`. With `C(t)=M_11(t)`,

```text
d = −C'(0)/C(0),
q = n C(0)/(Σᵢc_i²) = a/(a+d),
a = d q/(1−q).
```

Unknown common tag gain invalidates this simple occupancy calibration. Numerical differentiation of noisy data is outside this exact-information argument.

## A direct formula for the return matrix

With `Π=I−J`, define the convergent projected integral

```text
G = ∫₀^∞ Π B(t) Π dt.
```

The same mixed moments reconstruct `G`. Restricting the zero-frequency resolvent of `Q` to `1` perpendicular gives `G=[d(I−P)]⁻¹` on that subspace. Thus

```text
P = I − G⁺/d,
```

where `+` is the Moore–Penrose pseudoinverse, zero on the constant direction. This avoids the third time derivative if only `P` is wanted. Exact infinite-time integration still assumes access to the tail; it does not give a stable finite-record estimator.

## Lower construction

Choose a unit `w` perpendicular to `1,c,…,c^(n−2)`. The Vandermonde property makes its span one dimensional and every coordinate nonzero. Take positive `λ` and sufficiently small positive `η`, for example `η<λ/n`, and set

```text
E₀ = λ(J−I),       Eη = E₀ + η w wᵀ.
```

Both generators have strictly positive off-diagonal entries and are symmetric and irreducible. They have the same stationary law and the same bound/unbound dwell laws.

Let `V_k` consist of `2n`-component vectors whose bound and encounter components are polynomials of degree at most `k` evaluated at the labels. `Q(E₀)` preserves these spaces, and the perturbation annihilates `V_(n−2)`. Their semigroups therefore agree on that subspace. Multiplication by `D=diag(f)` raises polynomial degree by at most one.

Write a stationary multi-time moment as a product of powers of `D` and semigroups, bracketed by the constant vector in the stationary inner product. Telescope its difference by replacing one semigroup at a time. At a replaced factor, total degree at most `2n−3` forces at least one endpoint to contain at most `n−2` powers. That endpoint belongs to the annihilated polynomial subspace. Reversibility makes both semigroups self-adjoint in the common stationary inner product, so the argument applies from either side. Every telescoping term vanishes.

This proves agreement of **all** subthreshold moments at arbitrary times. Nevertheless, the explicit resolvent gives

```text
Pη−P₀ = aη/[(a+λ)(a+λ−η)] · w wᵀ.
```

Every entry changes because each `w_i` is nonzero.

At degree `2n−2`, the two-time moment with powers `(n−1,n−1)` distinguishes the pair. Its difference is

```text
α [h_(λ−η)(t)−h_λ(t)] [wᵀc^(n−1)]²,
```

where `h_κ(t)` is the bound diagonal entry of the exponential of the two-by-two matrix `[[-d,d],[a,−a−κ]]`. For positive `t,η`, the bracket is positive by the positive-semigroup Duhamel formula. The Vandermonde property makes the final squared factor nonzero. This matches the upper degree.

## Checks and limits

Run `scripts/check_moment_order.py` as shown in the README. The saved [results](../results/moment_order_checks.json) cover `n=2,…,6`: sampled lower-degree moments agree at roundoff, the threshold moment differs, and reconstruction errors pass declared tolerances. These finite checks supplement the all-times algebra; they do not replace it. The polynomial inversion becomes less well conditioned with increasing `n`.

Stationary preparation, distinct calibrated labels, known partner count, homogeneous binding hazards, and the declared reversible family are essential. Known-partner initialization, restricted encounter families, repeated labels, unknown gain, and finite noisy observations require separate analyses. Nothing here proves that the chosen finite qubit library recovers the ideal moment hierarchy, establishes a shot budget, or gives quantum advantage; the same moment algebra applies to a classical tagged signal.
