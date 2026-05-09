# Physics-Informed Neural Networks (PINNs) — Tutorial Series

A progressive, hands-on tutorial series accompanying the YouTube playlist **"Introduction to PINNs"**.  
Each notebook is self-contained and builds on the previous one, taking you from basic neural-network curve fitting all the way to solving compressible Euler equations with discontinuities.

▶ **YouTube playlist:** [Introduction to PINNs](https://www.youtube.com/watch?v=ZdEuASgRKOc&list=PLeAxGrK8ucC6jOacrHLZgh0vJHoQS8xxv)

---

## What are PINNs?

Physics-Informed Neural Networks embed physical laws (expressed as PDEs) directly into the training loss of a neural network. Instead of learning purely from labelled data, the network is penalised whenever it violates the governing equations, enabling accurate solutions with little or no simulation data.

```
Total loss = λ_pde · L_pde + λ_ic · L_ic + λ_bc · L_bc + λ_data · L_data
```

| Term | Role |
|------|------|
| `L_pde` | PDE residual at interior collocation points |
| `L_ic`  | Mismatch with initial conditions |
| `L_bc`  | Mismatch with boundary conditions |
| `L_data`| (Optional) Mismatch with available measurements |

---

## Repository Layout

```
PINN_tutorial/
├── S_2 curve_fit_fourier_series.ipynb   # Session 2 – Fourier series warm-up
├── S_3 curve_fit.ipynb                  # Session 3 – NN as a universal approximator
├── S_4 Wave_equation_using_First_order.ipynb   # Session 4a – Linear convection PINN
├── S_4 Wave_equation_using_RK3_First_order.ipynb # Session 4b – RK3 numerical solver
├── S5_Linear_convection.ipynb           # Session 5 – Finite-difference spatial schemes
├── S6_heat_eqn.ipynb                    # Session 6 – Heat equation PINN
├── S8_burg_conserve_PINN_V.ipynb        # Session 8a – Burgers equation PINN
├── S_8Lid_cavity.ipynb                  # Session 8b – Lid-driven cavity (Navier-Stokes)
├── sod_18.ipynb                         # Bonus – Sod shock tube (compressible Euler)
├── 1D_BeamPINNs1.ipynb                  # Bonus – Inverse problem: cantilever beam
└── docs/
    ├── 01_fourier_series.md
    ├── 02_curve_fitting.md
    ├── 03_wave_equation.md
    ├── 04_rk3_convection.md
    ├── 05_linear_convection.md
    ├── 06_heat_equation.md
    ├── 07_burgers_equation.md
    ├── 08_lid_cavity.md
    ├── 09_sod_shock_tube.md
    └── 10_beam_inverse.md
```

---

## Learning Path

Work through the notebooks in the order below. Each builds intuition needed for the next.

| Step | Notebook | Topic | Key concept |
|------|----------|--------|-------------|
| 1 | `S_2 curve_fit_fourier_series` | Fourier series | Basis function approximation |
| 2 | `S_3 curve_fit` | NN curve fitting | Neural nets as universal approximators |
| 3 | `S5_Linear_convection` | Finite-difference schemes | Classical PDE numerics (baseline comparison) |
| 4 | `S_4 Wave_equation_using_RK3_First_order` | RK3 time integrator | High-order time stepping |
| 5 | `S_4 Wave_equation_using_First_order` | Linear convection PINN | Your first PINN |
| 6 | `S6_heat_eqn` | Heat equation PINN | Parabolic PDE, learnable activations |
| 7 | `S8_burg_conserve_PINN_V` | Burgers equation PINN | Non-linear PDE, shock formation |
| 8 | `S_8Lid_cavity` | Lid-driven cavity | 2-D Navier-Stokes PINN |
| 9 | `sod_18` | Sod shock tube | Compressible Euler + artificial viscosity |
| 10 | `1D_BeamPINNs1` | Cantilever beam (inverse) | Inverse problem: identify material properties |

---

## Prerequisites

**Mathematics**
- Calculus (partial derivatives, chain rule)
- Linear algebra (matrix operations)
- Ordinary and partial differential equations (introductory level)

**Programming**
- Python 3.8+
- Familiarity with NumPy and Matplotlib

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/AGN000/PINN_tutorial.git
cd PINN_tutorial

# 2. Create a virtual environment (recommended)
python -m venv pinn_env
source pinn_env/bin/activate        # Linux / macOS
# pinn_env\Scripts\activate         # Windows

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib scipy ipywidgets jupyter

# 4. Launch Jupyter
jupyter notebook
```

> **GPU (recommended):** The notebooks detect and use CUDA automatically. If no GPU is available they fall back to CPU — training will be slower but results are identical.

---

## Quick Start

```python
# Minimal PINN skeleton used throughout the series
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers) - 1):
            net += [nn.Linear(layers[i], layers[i+1]), nn.Tanh()]
        self.net = nn.Sequential(*net[:-1])  # drop last Tanh

    def forward(self, x):
        return self.net(x)

# Automatic differentiation — the core of any PINN
def gradient(u, x):
    return torch.autograd.grad(u, x,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
```

---

## Notebook Summaries

### Session 2 — Fourier Series (`S_2 curve_fit_fourier_series.ipynb`)
Approximates f(x) = x² on [0, 1] using a Fourier cosine series with N = 1, 5, 10, 50 terms. Motivates why neural networks are better universal approximators than fixed basis functions.

### Session 3 — Neural Network Curve Fitting (`S_3 curve_fit.ipynb`)
Trains a 3-layer MLP (Tanh activations) to fit f(x) = x². Introduces the PyTorch training loop: forward pass → loss → backward → step.

### Session 4a — Linear Convection PINN (`S_4 Wave_equation_using_First_order.ipynb`)
Solves ∂u/∂t + c ∂u/∂x = 0 with a Gaussian initial condition. Features interactive widgets to explore wave speed, architecture, weight initialisation, and training epochs.

### Session 4b — RK3 Numerical Solver (`S_4 Wave_equation_using_RK3_First_order.ipynb`)
Classical finite-difference solver using 3rd-order Runge-Kutta time stepping. Compares FTBS, FTFS, and FTCS spatial discretisations against the analytical Gaussian solution — providing a numerical baseline to benchmark the PINN.

### Session 5 — Finite-Difference Spatial Schemes (`S5_Linear_convection.ipynb`)
Deep-dive into the three finite-difference stencils (FTBS, FTFS, FTCS) for the linear convection equation. Demonstrates stability/accuracy tradeoffs with animated comparisons.

### Session 6 — Heat Equation PINN (`S6_heat_eqn.ipynb`)
Solves the 1-D heat equation ∂u/∂t = α ∂²u/∂x² with Dirichlet boundary conditions. Introduces learnable `ParametricTanh` activation functions and compares multiple optimisers (Adam, SGD, AdamW).

### Session 8a — Burgers Equation (`S8_burg_conserve_PINN_V.ipynb`)
Solves the viscous Burgers equation in conservative form, a classic non-linear benchmark. Demonstrates how the PINN loss decreases from ~12.7 to ~2.5 × 10⁻⁴ over 10,000 iterations.

### Session 8b — Lid-Driven Cavity (`S_8Lid_cavity.ipynb`)
Full 2-D Navier-Stokes PINN for lid-driven cavity flow at Re = 100. Uses a two-phase training strategy (Adam then L-BFGS) and saves results to a `.dat` file for post-processing.

### Bonus — Sod Shock Tube (`sod_18.ipynb`)
Applies PINNs to the compressible Euler equations with initial discontinuity (Riemann problem). Introduces artificial viscosity as a learnable parameter to capture shocks without Gibbs oscillations. Based on: *"Discontinuity Computing using Physics-Informed Neural Networks"*, [ResearchGate](https://www.researchgate.net/publication/359480166).

### Bonus — Cantilever Beam Inverse Problem (`1D_BeamPINNs1.ipynb`)
Uses a PINN to *identify* the Young's modulus E of a beam from sparse displacement measurements — an inverse problem. Key techniques: log-parameterisation of E for numerical stability, hard boundary condition enforcement (w = x² · w̃).

---

## Key Techniques Covered

| Technique | Where introduced |
|-----------|-----------------|
| Automatic differentiation via `torch.autograd` | Session 4a |
| Collocation point sampling | Session 4a |
| Hard boundary condition enforcement | Beam notebook |
| Learnable activation functions (`ParametricTanh`) | Session 6 |
| Two-phase training (Adam → L-BFGS) | Session 8b, Beam |
| Artificial viscosity for discontinuities | Sod shock tube |
| Log-parameterisation for stability | Beam inverse problem |
| Interactive widgets for hyperparameter exploration | Sessions 4a, 6 |

---

## Citation

If you use this material in your work or teaching, please cite the associated YouTube series:

```
@misc{pinn_tutorial_youtube,
  author  = {Arun Govind Neelan},
  title   = {Introduction to PINNs},
  year    = {2024},
  url     = {https://www.youtube.com/playlist?list=PLeAxGrK8ucC6jOacrHLZgh0vJHoQS8xxv}
}
```

---

## License

This repository is provided for educational purposes. Feel free to use and adapt the code with attribution.

---

## Contributing

Found a bug or have a suggestion? Open an issue or submit a pull request — contributions are welcome.
