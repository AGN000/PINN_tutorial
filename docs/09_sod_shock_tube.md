# Bonus — Sod Shock Tube (Compressible Euler Equations)

**Notebook:** `sod_18.ipynb`

## Learning Objectives

- Apply PINNs to compressible flow governed by the Euler equations.
- Handle initial discontinuities (Riemann problems) using artificial viscosity.
- Use a learnable viscosity parameter to let the network control its own regularisation.
- Implement early stopping based on a convergence criterion.

## Reference

> "Discontinuity Computing using Physics-Informed Neural Networks"  
> [ResearchGate link](https://www.researchgate.net/publication/359480166_Discontinuity_Computing_using_Physics-Informed_Neural_Networks)

## Governing Equations

**1-D compressible Euler equations (conservation form):**

```
∂ρ/∂t + ∂(ρu)/∂x = 0                               (mass)
∂(ρu)/∂t + ∂(ρu² + p)/∂x = ∂(ν ∂u/∂x)/∂x          (momentum + viscosity)
∂E/∂t + ∂(u(E + p))/∂x = ∂(ν u ∂u/∂x)/∂x           (energy + viscosity)
```

where:
- ρ = density
- u = velocity
- p = pressure
- E = ρ(e + u²/2) = total energy per unit volume
- ν = artificial viscosity (learnable)

## Sod Initial Conditions (Riemann Problem)

```
x ≤ 0.5:  ρ = 1.0,   p = 1.0,   u = 0   (high-pressure left state)
x > 0.5:  ρ = 0.125, p = 0.1,   u = 0   (low-pressure right state)
```

The solution at t = 0.15 consists of:
1. Left-travelling rarefaction wave
2. Contact discontinuity
3. Right-travelling shock wave

## Neural Network Architecture

```
Inputs:  (t, x)            — 2 scalars
Outputs: (ρ, p, u, α²)    — density, pressure, velocity, artificial viscosity

Architecture: 5 hidden layers × 15 neurons, ParametricTanh activations
```

The network outputs 4 fields simultaneously, including the viscosity α². Parameterising as α² (instead of ν directly) guarantees ν ≥ 0 at all points.

## Learnable Artificial Viscosity

Classical finite-difference methods for shocks use fixed artificial viscosity:

```
ν_fixed = C_fix · Δx
```

This PINN instead learns ν(x, t) = α(x,t)², allowing the network to apply viscosity only where needed (near the shock) and stay inviscid elsewhere. This is analogous to adaptive mesh refinement in classical methods.

## Loss Functions

```python
def loss_pde(self, x):
    # Computes residuals of conservation laws with viscous terms
    # Uses torch.autograd.grad for all spatial and temporal derivatives

def loss_ic(self, x_ic, rho_ic, u_ic, p_ic):
    # MSE between network output at t=0 and Sod initial conditions
```

Total loss: `L = L_pde + L_ic`

## Two-Phase Training

| Phase | Optimiser | LR | Max epochs | Early stopping |
|-------|----------|-----|-----------|----------------|
| 1 | Adam | 0.001 | 1,001 | loss < 1e-3 |
| 2 | L-BFGS | 0.1 | 501 | loss < 1e-3 |

L-BFGS uses `max_iter=20` line search iterations per step.

## Key Configuration

```python
x_min=0, x_max=1
t_max=0.15
nx=101, nt=101
CFL=1
convergence_threshold=1e-3
```

## Expected Results

At t = 0.15, the PINN should reproduce:
- Smooth rarefaction fan on the left
- Flat plateau between rarefaction and contact
- Sharp density/pressure jump at the contact discontinuity
- Sharp jump in all variables at the shock

## Challenges and Limitations

| Challenge | Note |
|-----------|------|
| Shock sharpness | Learnable viscosity helps but the network cannot match a true discontinuity |
| Training instability | Conservation residuals can spike if ρ or p go negative during training |
| Exact shock speed | The PINN may slightly mis-predict the shock position at early epochs |

## Exercises

1. Run to t = 0.25 (longer time). Does the shock remain sharp or does the PINN smear it over time?
2. Try a different Riemann problem (e.g., two shock problem: both states moving toward each other).
3. Remove the artificial viscosity (fix ν = 0). Does the network converge? What artefacts appear?
4. Plot the learned viscosity field ν(x, t=0.15). Is it concentrated near the shock as expected?
