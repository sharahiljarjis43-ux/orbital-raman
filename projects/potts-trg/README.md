# Potts / random-cluster tensor-network prototype

**Project owner and maintainer: Sharahil Jarjis.**

**Status: numerical prototype with an exact square-lattice benchmark and incomplete finite-D convergence.**

This project estimates the mean number of percolation clusters per lattice site using a real-q random-cluster loop tensor and U(1)-sector tensor renormalization. It compares a fixed-p finite difference, a finite difference along the critical line, and a fixed-isometry linear-response calculation.

The useful computational question is how the differentiation path and tensor truncation affect an observable obtained from a free-energy derivative. The exact identity underlying the path comparison does not make a truncated numerical calculation exact.

## Quick check

Use Python 3.10 or later and NumPy. From this project directory:

```sh
python3 -m pip install -r requirements.txt
python3 scripts/smoke_check.py
```

The smoke check takes about a second in the recorded environment. It checks critical-line formulas at q=1 and q=2, zero-coupling normalization, small-coupling Ising series, tensor charge conservation, and agreement of the dual-number derivative with a finite difference in a small untruncated square-tensor calculation. It also summarizes the saved convergence ladders. Results are written to [results/smoke-check.json](results/smoke-check.json).

The derivative check concerns the same finite contraction on both sides. It does not establish thermodynamic convergence or reproduce a high-D campaign.

## Method

The normalization is `ln Z_RC / N = 0.5 ln(q) + f(p,q)`, giving cluster density `kappa = 0.5 + partial_q f(p_c,1)`. In the exact model, `f(p,1)=0`; therefore moving along a smooth path through the target p preserves this q derivative.

The candidate critical paths use `v = p/(1-p)` with `v=sqrt(q)` on the square lattice and the positive root of `v^3+3v^2=q` on the triangular lattice. Following these paths changes truncation behavior. A fixed-isometry dual-number calculation supplies a separate numerical comparison; its good small-contraction derivative check does not imply accurate derivatives after finite-D truncation.

## Saved results and limits

The four [saved logs](logs) use 24 renormalization steps, D=8–48 and q steps of 0.05 or 0.1. No extrapolation is included.

| Quantity | Saved result | Interpretation |
| --- | ---: | --- |
| Exact square-lattice cluster-density benchmark | `(3 sqrt(3)-5)/2 = 0.098076211...` | Analytical reference |
| Square, D=48, q step 0.05, five-point stencil | 0.0980894 | Approximate result; error about 0.0000132 |
| Triangular, D=32, same stencil | 0.1119052 | Finite-D estimate |
| Triangular, D=48, same stencil | 0.1118490 | D32-to-D48 change is 0.0000562 |

The triangular change exceeds the historical quoted uncertainty of ±0.00003. That uncertainty is not adopted here. Agreement on the square benchmark does not certify the triangular error bar. Saved calibration, tangent-control and linear-response validation outputs were not recovered in the imported archive, so the old report's complete validation narrative is not reproduced by this package.

The historical fixed-isometry linear-response estimate was less accurate at finite D and remains a comparison method, not the preferred estimator. This is computational exploratory work; it does not claim a new tensor-renormalization method or a converged triangular-lattice precision result.

## Source guide

| File | Purpose |
| --- | --- |
| `src/rc_tensor.py` | Square loop tensors and triangular blocking |
| `src/trg_u1.py` | Charge-sector SVD and tensor-renormalization contractions |
| `src/run_kappa.py` | Fixed-p differentiation driver |
| `src/critline.py` | Critical paths and three/five-point comparisons |
| `src/selfdual.py` | Historical square-path comparison |
| `src/dtrg.py` | Fixed-isometry dual-number derivative |
| `src/exact.py` | Ising benchmark formulas evaluated by numerical quadrature, with truncated high-temperature series |

For an optional low-D numerical run:

```sh
python3 src/critline.py square 0.05 8
```

Larger D values can be substantially more expensive. The quick check does not launch them. See [provenance](PROVENANCE.md) for the imported source and assistance history.
