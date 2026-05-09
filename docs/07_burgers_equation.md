# Session 8a — Burgers Equation PINN

**Notebook:** `S8_burg_conserve_PINN_V.ipynb`

## Learning Objectives

- Apply a PINN to a non-linear PDE in conservative form.
- Observe how training loss converges over 10,000 iterations.
- Understand why non-linearity makes PDE solving harder for both classical and learning-based methods.

## Governing Equation

**Viscous Burgers equation (conservative form):**

```
∂u/∂t + ∂/∂x(u²/2) = ν ∂²u/∂x²
```

This is equivalent to:
```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
```

For small viscosity ν, the non-linear convection term dominates and the solution develops steep gradients (shocks).

## Why Burgers Is Important

Burgers' equation shares mathematical structure with the Navier-Stokes equations (non-linear convection + viscous diffusion) but in 1-D. It is the canonical benchmark for:
- PINN non-linear PDE solving
- Shock-capturing numerical methods
- Turbulence model testing

## PINN Loss Convergence

| Iteration | Loss |
|-----------|------|
| 0 | 12.716 |
| 1,000 | ~1.2 |
| 5,000 | ~0.05 |
| 10,000 | ~2.5 × 10⁻⁴ |

The initial high loss (~12.7) reflects the network starting from a random initialisation far from the PDE solution. The loss decreasing by 5 orders of magnitude demonstrates effective convergence.

## Conservative vs. Non-Conservative Formulation

The notebook uses the **conservative form** `∂(u²/2)/∂x` rather than `u ∂u/∂x`.

- **Conservative form:** Conserves integral of u² over the domain — important for shock-capturing accuracy.
- **Non-conservative form:** Simpler to implement but can give wrong shock speeds near discontinuities.

For PINNs with smooth solutions the difference is minor; for near-inviscid problems (ν → 0) it matters significantly.

## Common Challenges

| Challenge | Mitigation |
|-----------|-----------|
| Steep gradients (small ν) | Increase collocation point density near expected shock location |
| Loss oscillations | Reduce learning rate or switch to AdamW |
| Training stagnation | Follow Adam phase with L-BFGS (see Session 8b) |
| Non-convergence | Check for vanishing gradients; try deeper architecture |

## Exercises

1. Vary ν from 0.1 to 0.001. At what value does the network start failing to resolve the shock?
2. Add a sine-wave initial condition: u(x,0) = −sin(πx). Does the network still converge?
3. Compare the conservative and non-conservative PDE residual formulations. Are the final L2 errors different for ν = 0.01?
4. Extend training to 20,000 iterations. Plot the loss curve on a log scale — does it still decrease, or has it plateaued?
