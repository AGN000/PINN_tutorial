# Session 3 — Neural Network Curve Fitting

**Notebook:** `S_3 curve_fit.ipynb`

## Learning Objectives

- Build and train a minimal PyTorch neural network from scratch.
- Understand the supervised learning loop: forward pass → loss → backward → parameter update.
- Appreciate that a small MLP can approximate f(x) = x² without any hand-crafted basis.

## Architecture

```
Input (1) → Linear(1→10) → Tanh → Linear(10→10) → Tanh → Linear(10→1) → Output (1)
```

| Layer | Parameters |
|-------|-----------|
| Linear 1→10 | 10 weights + 10 biases |
| Linear 10→10 | 100 weights + 10 biases |
| Linear 10→1 | 10 weights + 1 bias |
| **Total** | **141 parameters** |

## Training Configuration

| Setting | Value |
|---------|-------|
| Training points | 100 (uniform on [0, 1]) |
| Target | f(x) = x² |
| Optimiser | Adam |
| Learning rate | 0.01 |
| Loss | Mean Squared Error (MSE) |
| Epochs | 1000 |

## PyTorch Training Loop

```python
for epoch in range(1000):
    optimizer.zero_grad()       # clear previous gradients
    y_pred = model(x_train)     # forward pass
    loss = mse_loss(y_pred, y_train)  # compute loss
    loss.backward()             # backpropagation
    optimizer.step()            # update weights
```

## Key Concepts

**Universal Approximation Theorem** — A single hidden-layer network with enough neurons and a non-linear activation can approximate any continuous function on a compact domain to arbitrary precision. In practice, depth (multiple layers) is more efficient than width.

**Tanh activation** — Smooth and bounded (output ∈ (-1, 1)), making gradients well-behaved. A common default for PINNs because the network output and all its derivatives remain smooth, which is required when differentiating through the network to compute PDE residuals.

## Expected Results

Loss should decrease below 1 × 10⁻⁴ by epoch 1000. The scatter plot of NN predictions vs. true values should show near-perfect alignment.

## Exercises

1. Change the target to f(x) = sin(2πx). Does the network still converge? Try increasing the number of neurons.
2. Remove one hidden layer (use only one `Linear(1→10) → Tanh → Linear(10→1)`). How does the final loss change?
3. Replace `nn.Tanh()` with `nn.ReLU()`. Plot the learned function. Why does ReLU produce a piecewise-linear output?
4. Add noise to the training data: `y_train = x_train**2 + 0.01 * torch.randn_like(x_train)`. Does the network overfit?
