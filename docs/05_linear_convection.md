# Session 5 — Finite-Difference Spatial Schemes (Deep Dive)

**Notebook:** `S5_Linear_convection.ipynb`

## Learning Objectives

- Compare FTBS, FTFS, and FTCS side-by-side in a clean, modular code structure.
- Understand numerical dissipation, dispersion, and stability through animated visualisations.
- Establish the classical numerical baseline that motivates learning-based solvers.

## Core Functions

```python
def simulate_rk3(spatial_operator):
    """
    Advance the solution in time using RK3 with the given spatial operator.
    Returns u at all time steps.
    """

def exact_solution(x, t, c):
    """
    Analytical Gaussian: exp(-K*(x - c*t)**2)
    """

def apply_periodic(u):
    """
    Enforce u[0] == u[-2] and u[-1] == u[1] (ghost cells).
    """
```

## Stability Analysis (von Neumann)

For the advection equation with FTBS and RK3 time stepping, the amplification factor for a wave mode with wavenumber k is:

```
|G| ≤ 1   when CFL ≤ 1
```

For FTCS with any explicit time integrator:

```
|G| > 1   for all CFL > 0   (always unstable)
```

## Error Sources

| Source | Cause | Effect |
|--------|-------|--------|
| Truncation error | Finite Δx, Δt | Phase / amplitude error |
| Numerical dissipation | 1st-order upwind | Gaussian smears over time |
| Numerical dispersion | Central differencing | Spurious oscillations |
| Boundary effects | Non-periodic or coarse ghost cells | Reflected waves |

## Key Visualisation

The animated plot shows:
1. **Blue line:** exact Gaussian travelling right.
2. **Orange line:** FTBS — follows the pulse but widens (dissipation).
3. **Green line:** FTFS — diverges because c > 0 (wrong upwind direction).
4. **Red line:** FTCS — diverges immediately (instability).

## Connection to PINNs

PINNs avoid the discretisation errors above because they represent u(x, t) as a continuous function. The trade-off is computational cost (optimisation) vs. accuracy guarantee (finite-difference methods have known error bounds, PINNs do not).

## Exercises

1. Try c = −1.0. Which scheme should now be stable — FTBS or FTFS? Verify by running the simulation.
2. Increase the grid to 401 points (halve Δx). By how much does the FTBS L2 error decrease at t = 1? (Expected: roughly halved for a 1st-order scheme.)
3. Implement a 2nd-order upwind (Fromm's scheme) and add it to the animated comparison.
