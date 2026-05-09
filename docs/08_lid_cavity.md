# Session 8b — Lid-Driven Cavity Flow (Navier-Stokes PINN)

**Notebook:** `S_8Lid_cavity.ipynb`

## Learning Objectives

- Solve the 2-D incompressible Navier-Stokes equations with a PINN.
- Apply a two-phase training strategy: Adam for rapid initial progress, L-BFGS for fine convergence.
- Enforce no-slip and lid boundary conditions through the loss function.
- Export results to a `.dat` file for external post-processing or comparison with reference data.

## Governing Equations

**Incompressible Navier-Stokes:**

```
∂u/∂x + ∂v/∂y = 0                           (continuity)
u ∂u/∂x + v ∂u/∂y = −∂p/∂x + (1/Re)∇²u    (x-momentum)
u ∂v/∂x + v ∂v/∂y = −∂p/∂y + (1/Re)∇²v    (y-momentum)
```

**Reynolds number:** Re = 100 (laminar, steady-state solution exists)

## Boundary Conditions

```
Top lid   (y=1): u = 1,  v = 0   (moving wall — drives the flow)
Other walls     : u = 0,  v = 0   (no-slip)
```

## Network Architecture

The network takes (x, y) as inputs and outputs three quantities simultaneously:

```
Input: [x, y]  →  FCNN  →  Output: [u, v, p]
```

This avoids the need for separate networks per variable and naturally couples the fields through the shared hidden layers.

## Two-Phase Training

### Phase 1 — Adam

| Metric | Value |
|--------|-------|
| Epochs | 1,000 |
| Initial loss | 0.383184 |
| Final loss | 0.114451 |
| PDE loss | 0.000036 |
| BC loss | 0.110809 |

Adam uses adaptive per-parameter learning rates. It makes fast initial progress but can stall near the minimum because the steps become very small.

### Phase 2 — L-BFGS

| Metric | Value |
|--------|-------|
| Max epochs | 500 |
| Initial loss | 0.102838 |
| Final loss | 0.001834 |
| Convergence epoch | ~180 |

L-BFGS is a quasi-Newton method. It approximates the Hessian and takes large, well-directed steps near the minimum — much more efficient than Adam at high accuracy. The trade-off: it requires evaluating the loss multiple times per step (line search) and can fail if the loss landscape is not smooth.

### Why This Combination Works

```
Adam:   loss → 0.1  (fast, noisy descent)
L-BFGS: loss → 0.001 (precise, stable refinement)
```

Adam escapes poor initialisations; L-BFGS finds the precise local minimum.

## L-BFGS in PyTorch

```python
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(),
                                     lr=1.0,
                                     max_iter=20,
                                     history_size=50)

def closure():
    optimizer_lbfgs.zero_grad()
    loss = compute_loss()
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
```

The `closure` pattern is mandatory for L-BFGS — it re-evaluates the loss during the line search.

## Output Files

Results are saved to `lid_driven_cavity_re100_pinn_results.dat` in a format compatible with standard CFD post-processors (e.g., Tecplot, ParaView with CSV reader).

## Validation

Compare PINN velocity profiles against the Ghia et al. (1982) benchmark data:
- u-velocity along the vertical centreline (x = 0.5)
- v-velocity along the horizontal centreline (y = 0.5)

These are the standard validation plots for lid-driven cavity solvers.

## Exercises

1. Increase Re to 400. Does the PINN converge? What changes are needed (more collocation points, longer training, deeper network)?
2. Plot the streamlines and vorticity field from the PINN output. Identify the primary vortex and corner eddies.
3. Swap the Phase 2 optimiser from L-BFGS to Adam with a reduced learning rate (1e-4). Compare the final loss and wall-clock time.
4. Add pressure boundary conditions (pin pressure at one corner). Does this improve the convergence of the pressure field?
