# Session 4b — RK3 Numerical Solver for Linear Convection

**Notebook:** `S_4 Wave_equation_using_RK3_First_order.ipynb`

## Learning Objectives

- Understand classical finite-difference spatial discretisation (FTBS, FTFS, FTCS).
- Implement a 3rd-order Runge-Kutta time integrator.
- Use the numerical solution as a baseline to evaluate PINN accuracy.

## Governing Equation

Same as Session 4a:

```
∂u/∂t + c ∂u/∂x = 0
```

with Gaussian initial condition u(x,0) = exp(−Kx²) and periodic boundary conditions.

## Spatial Discretisation Schemes

| Scheme | Stencil | Stability | Accuracy |
|--------|---------|-----------|---------|
| FTBS (backward space) | `(uᵢ − uᵢ₋₁)/Δx` | Stable for c > 0, CFL ≤ 1 | 1st-order |
| FTFS (forward space) | `(uᵢ₊₁ − uᵢ)/Δx` | Stable for c < 0, CFL ≤ 1 | 1st-order |
| FTCS (central space) | `(uᵢ₊₁ − uᵢ₋₁)/(2Δx)` | Unconditionally unstable | 2nd-order |

```python
def spatial_bs(u, c, dx):   # FTBS: upwind for c > 0
    return -c * (u - np.roll(u, 1)) / dx

def spatial_fs(u, c, dx):   # FTFS: upwind for c < 0
    return -c * (np.roll(u, -1) - u) / dx

def spatial_cs(u, c, dx):   # FTCS: central (unstable!)
    return -c * (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
```

`np.roll(u, 1)` shifts the array right by one (implements periodic BCs).

## RK3 Time Integration

The standard 3-stage, 3rd-order Runge-Kutta scheme (TVD-RK3 of Shu & Osher):

```
Stage 1:  u* = uⁿ + Δt · L(uⁿ)
Stage 2:  u** = (3/4)uⁿ + (1/4)(u* + Δt · L(u*))
Stage 3:  uⁿ⁺¹ = (1/3)uⁿ + (2/3)(u** + Δt · L(u**))
```

where `L(u)` is the chosen spatial operator.

## CFL Condition

```
CFL = c · Δt / Δx ≤ 1   (required for stability with upwind schemes)
```

The notebook uses CFL = 0.5, giving Δt = 0.5 · Δx / c.

## Domain and Parameters

| Parameter | Value |
|-----------|-------|
| Domain | x ∈ [−2, 2] |
| Grid points | 201 |
| Wave speed c | 1.0 |
| CFL | 0.5 |
| IC width K | 100 (narrow Gaussian) |

## What to Observe

- **FTCS diverges** even at small CFL — the central scheme is unconditionally unstable for the advection equation.
- **FTBS remains bounded** and tracks the pulse, but numerical diffusion smears the Gaussian over time.
- The animated comparison shows how each scheme deviates from the exact travelling wave.

## Exercises

1. Increase CFL from 0.5 to 1.1 (violate the stability bound). What happens to FTBS?
2. Switch to a 4th-order central scheme: `(-u_{i+2} + 8u_{i+1} - 8u_{i-1} + u_{i-2}) / (12Δx)`. Does it remain unstable?
3. Compare the L2 error of FTBS vs. 2nd-order Lax-Wendroff at t = 1.0.
