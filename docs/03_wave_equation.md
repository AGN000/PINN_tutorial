# Session 4a — Linear Convection PINN

**Notebook:** `S_4 Wave_equation_using_First_order.ipynb`

## Learning Objectives

- Formulate a PINN for a first-order hyperbolic PDE.
- Use `torch.autograd.grad` to compute PDE residuals automatically.
- Explore the effect of architecture, initialisation, and wave speed via interactive widgets.

## Governing Equation

**1-D linear convection (advection):**

```
∂u/∂t + c ∂u/∂x = 0,   x ∈ [−1, 1], t ∈ [0, T]
```

**Initial condition:**

```
u(x, 0) = exp(−Kx²)   (Gaussian pulse)
```

**Exact solution:** the Gaussian translates at speed c without changing shape:

```
u(x, t) = exp(−K(x − ct)²)
```

## PINN Formulation

The total loss combines:

```
L = λ_pde · L_pde + λ_ic · L_ic
```

| Loss term | Expression | Points |
|-----------|-----------|--------|
| `L_pde` | `‖∂u/∂t + c ∂u/∂x‖²` | 20,000 interior collocation points |
| `L_ic`  | `‖u(x,0) − exp(−Kx²)‖²` | 1,000 points at t = 0 |

### Automatic Differentiation

```python
u = model(torch.cat([x, t], dim=1))
u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
pde_residual = u_t + c * u_x
```

`create_graph=True` is essential — it tells PyTorch to build the computational graph of the derivatives so that second calls to `.backward()` can differentiate through them.

## Interactive Widgets

The notebook exposes sliders and dropdowns for:

| Widget | Controls |
|--------|---------|
| Wave speed C | Convection speed (0.5 – 2.0) |
| Epochs | Number of training iterations |
| Hidden layers / neurons | Network depth and width |
| Initialisation | Xavier uniform/normal, Kaiming, normal, zeros |
| Formulation type | First-order vs. other formulations |

## Weight Initialisation Guide

| Method | When to use |
|--------|------------|
| Xavier uniform | Default for Tanh / Sigmoid — keeps variance stable across layers |
| Xavier normal | Same benefit, slightly different distribution |
| Kaiming uniform | Designed for ReLU — rarely best for PINNs |
| Normal / Zeros | Baseline comparisons; zeros leads to symmetry breaking failure |

## Common Pitfalls

- **Large wave speed + coarse collocation grid:** The network may fail to track fast-moving features. Increase collocation points or reduce the time horizon.
- **Zero initialisation:** All neurons produce identical outputs throughout training (symmetry problem) — avoid for hidden layers.
- **Missing `create_graph=True`:** Gradients are computed correctly on the forward pass but the graph is discarded, so second-order losses or L-BFGS will fail.

## Exercises

1. Set c = 2.0 and compare the final L2 error to c = 0.5. What changes?
2. Double the number of collocation points. Does the error improve proportionally?
3. Add a boundary condition loss at x = −1 and x = +1 using periodic BCs: `u(−1, t) = u(+1, t)`. Does it help for c = 1.0?
