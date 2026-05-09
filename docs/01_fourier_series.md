# Session 2 — Fourier Series Curve Fitting

**Notebook:** `S_2 curve_fit_fourier_series.ipynb`

## Learning Objectives

- Understand how any well-behaved function can be approximated by a sum of sinusoids.
- Observe the trade-off between series length (model complexity) and approximation accuracy.
- Build intuition for why fixed basis functions are limited — motivating neural networks.

## Mathematical Background

The **Fourier cosine series** on [0, 1] for a function f(x) is:

```
f(x) ≈ a₀/2 + Σₙ aₙ cos(nπx)
```

For f(x) = x², the cosine coefficients are:

```
a₀ = 2/3
aₙ = 4(-1)ⁿ / (n²π²)   for n ≥ 1
```

## What the Notebook Does

| Step | Code action | Purpose |
|------|-------------|---------|
| 1 | Define `f_exact(x) = x²` | The target function to approximate |
| 2 | Define `f_fourier_cosine(x, N)` | Sum N Fourier cosine terms |
| 3 | Plot N = 1, 5, 10, 50 | Show convergence visually |

## Key Observations

- N=1 term: poor approximation; visible error everywhere.
- N=10 terms: good visual match except near x=0.
- N=50 terms: nearly indistinguishable from x².
- More terms always help here, but you must choose the basis *a priori* — you cannot adapt it to data.

## Connection to PINNs

Neural networks replace fixed Fourier basis functions with *learned* basis functions. The hidden-layer neurons adapt during training, making NNs far more flexible for approximating complex PDE solutions.

## Exercises

1. Replace f(x) = x² with f(x) = |x - 0.5| and repeat. How many terms are needed for a good fit? Why does convergence slow near the corner?
2. Try a Fourier *sine* series on [0, 1] (appropriate for functions with f(0) = f(1) = 0). Derive the coefficients for f(x) = sin(πx).
3. Plot the pointwise error |f_exact(x) - f_fourier(x)| for N = 5 and N = 50.
