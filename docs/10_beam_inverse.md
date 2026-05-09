# Bonus — Cantilever Beam Inverse Problem

**Notebook:** `1D_BeamPINNs1.ipynb`

## Learning Objectives

- Distinguish forward problems (solve for u given known parameters) from inverse problems (identify unknown parameters from sparse data).
- Use logarithmic parameterisation to stabilise training over large parameter ranges.
- Apply hard boundary condition enforcement instead of penalty-based soft enforcement.
- Recover a physical material property (Young's modulus E) from displacement measurements.

## Problem Description

A cantilever beam of length L = 1 m is clamped at x = 0 and free at x = 1. A uniformly distributed load q (N/m) causes vertical deflection w(x). Given sparse measurements of w(x), identify the Young's modulus E.

## Governing Equations

**Euler-Bernoulli beam theory:**

```
EI w''(x) = M(x)       (constitutive: moment = stiffness × curvature)
M''(x) = q(x)          (equilibrium: load = second derivative of moment)
```

**Boundary conditions (clamped-free):**
```
w(0) = 0     (zero displacement at clamped end)
w'(0) = 0    (zero slope at clamped end)
M(1) = 0     (zero moment at free end)
M'(1) = 0    (zero shear at free end)
```

## Physical Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Beam length | L | 1.0 m |
| Second moment of area | I | 1 × 10⁻⁴ m⁴ |
| Distributed load | q | −500 N/m |
| True Young's modulus | E_true | 200 GPa |
| Displacement scale | w_scale | 1 × 10⁻⁶ (non-dimensionalisation) |

## Network Architecture

The PINN outputs **two fields simultaneously**:

```
Input: x ∈ [0, 1]  →  FCNN  →  Output: [w̃(x), M(x)]
```

- **w̃(x):** Unconstrained displacement proxy (raw network output)
- **M(x):** Bending moment (directly from network)

The actual displacement is recovered via hard boundary enforcement.

## Hard Boundary Condition Enforcement

Instead of adding BC terms to the loss, the displacement is constructed to satisfy the clamped conditions *exactly*:

```python
w = x**2 * w_tilde   # automatically gives w(0)=0 and w'(0)=0
```

**Why this matters:**
- Penalty-based BCs satisfy conditions only approximately (up to the loss weight).
- Hard enforcement gives exact satisfaction at zero additional computational cost.
- The network only needs to learn w̃, which has no constraints — easier optimisation.

## Logarithmic Parameterisation of E

Young's modulus spans many orders of magnitude (steel: 200 GPa, rubber: 0.01 GPa). Direct training on E can cause numerical instability. Instead, the network learns `logE`:

```python
logE = nn.Parameter(torch.tensor(torch.log(torch.tensor(E_init))))
E = torch.exp(logE)   # always positive, spans all scales equally
```

This maps the parameter space to log-scale, making gradient steps proportionally equal regardless of the magnitude of E.

## Loss Components

```
L = λ_data · L_data + λ_const · L_const + λ_equil · L_equil + λ_bc · L_bc
```

| Term | Equation enforced | Weight |
|------|-------------------|--------|
| `L_data` | `‖w_pred − w_measured‖²` | 50 |
| `L_const` | `‖M − EI w''‖²` | 5 |
| `L_equil` | `‖M'' − q‖²` | 1 |
| `L_bc`   | BC conditions on M | 10 |

The high weight on `L_data` (50) reflects that the sparse measurements are the primary information source.

## Two-Phase Training

| Phase | Optimiser | Iterations | Purpose |
|-------|----------|-----------|---------|
| 1 | Adam (lr=1e-3) | 4,000 | Rapid exploration of parameter space |
| 2 | L-BFGS | 10 | Precise convergence to minimum |

Only 10 L-BFGS steps are needed because the Adam phase has already located a good basin.

## Expected Result

The recovered E should converge to ≈ 200 GPa (within a few percent of E_true = 200 × 10⁹ Pa).

Training loss progression:
- Adam: loss decreases from ~O(10) to ~O(10⁻³)
- L-BFGS: final refinement to ~O(10⁻⁵)

## Forward vs. Inverse — Conceptual Comparison

| Aspect | Forward problem | Inverse problem (this notebook) |
|--------|----------------|--------------------------------|
| Known | E, q, BCs | q, BCs, sparse w measurements |
| Unknown | w(x), M(x) | E (and the full fields w, M) |
| Loss terms | PDE + BCs | PDE + BCs + data fit |
| Extra parameter | None | logE (learnable scalar) |

## Exercises

1. Reduce the number of measurement points from 100 to 10. Does the recovered E remain accurate?
2. Add Gaussian noise to the measurements (σ = 1% of max displacement). How does the recovered E change?
3. Try direct parameterisation (learn E directly, not logE). Does training still converge? Compare the convergence curves.
4. Extend to a beam with a point load at x = 0.5 instead of distributed load. Derive the analytical solution and compare.
