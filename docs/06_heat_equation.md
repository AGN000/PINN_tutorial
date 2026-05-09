# Session 6 — Heat Equation PINN

**Notebook:** `S6_heat_eqn.ipynb`

## Learning Objectives

- Solve a parabolic (diffusion) PDE with a PINN.
- Use learnable activation functions (`ParametricTanh`) for improved expressivity.
- Compare different optimisers (Adam, SGD, AdamW) and weight initialisations.
- Evaluate accuracy against the analytical solution via relative L2 error.

## Governing Equation

**1-D heat (diffusion) equation:**

```
∂u/∂t = α ∂²u/∂x²,   x ∈ [0, 1], t ∈ [0, T]
```

**Boundary conditions:**
```
u(0, t) = 0,   u(1, t) = 0   (Dirichlet, homogeneous)
```

**Initial condition:**
```
u(x, 0) = sin(πx)
```

**Exact solution:**
```
u(x, t) = exp(−α π² t) sin(πx)
```

## Learnable Activation: ParametricTanh

Standard Tanh has a fixed slope at the origin. `ParametricTanh` introduces a learnable scale parameter α:

```python
class ParametricTanh(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return torch.tanh(self.alpha * x)
```

During training, α is updated by gradient descent alongside the network weights. This allows the network to adaptively control the "sharpness" of activation for each layer.

## PINN Loss Function

```
L = (1/N_pde) Σ |∂u/∂t − α ∂²u/∂x²|²   [PDE residual]
  + (1/N_ic)  Σ |u(x,0) − sin(πx)|²      [initial condition]
  + (1/N_bc)  Σ |u(0,t)|² + |u(1,t)|²    [boundary conditions]
```

| Term | Collocation points |
|------|--------------------|
| PDE  | 10,000 |
| IC   | 100 |
| BC   | 100 (50 per boundary) |

## Second Derivative via Autograd

```python
u_x  = torch.autograd.grad(u, x,  grad_outputs=torch.ones_like(u), create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
u_t  = torch.autograd.grad(u, t,  grad_outputs=torch.ones_like(u), create_graph=True)[0]
pde_residual = u_t - alpha * u_xx
```

Note: `create_graph=True` must be set on the *first* differentiation to allow computing the second derivative.

## Optimiser Comparison

| Optimiser | Strength | Weakness |
|-----------|----------|---------|
| Adam | Adaptive learning rates; fast initial convergence | Can stagnate near the minimum |
| SGD | Simple; good generalisation with tuned LR | Slow without momentum |
| AdamW | Adam + weight decay (L2 regularisation) | Extra hyperparameter (λ) |

For most PINN problems, Adam is the practical default, often followed by L-BFGS for final refinement.

## Evaluation Metric

**Relative L2 error:**

```
ε = ‖u_pred − u_exact‖₂ / ‖u_exact‖₂
```

The notebook reports this across the full (x, t) grid and produces an animated comparison.

## Exercises

1. Change the thermal diffusivity α. For larger α (fast diffusion), does the PINN need more or fewer training iterations?
2. Use non-homogeneous BCs: u(0,t) = 1, u(1,t) = 0. Update the loss and the exact solution accordingly.
3. Replace `ParametricTanh` with a fixed `nn.Tanh()`. Compare the final relative L2 error after the same number of epochs.
4. Add a regularisation term to the loss: `λ · Σ |weights|²`. How does this affect the learned solution?
