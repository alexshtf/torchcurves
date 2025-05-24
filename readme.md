# torch-bspline

A PyTorch module for differentiable parametric with learnable coefficients, such as a B-Spline curve with learnable control points.

This package provides a fully differentiable B-spline implementation that integrates seamlessly with PyTorch's autograd system, making experimental research such as Kolmogov-Arnold networks, or path-planning in robotics easy.

## Features

- **Fully Differentiable**: Custom autograd function ensures gradients flow properly through the curve evaluation.
- **Batch Processing**: Vectorized operations for efficient batch evaluation

## Installation

```bash
pip install torchcurves
```

### Development Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Clone the repository
git clone https://github.com/alexshtf/torchcurves.git
cd torchcurves

# Create virtual environment and install
uv venv
uv pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from torch_bspline import BSpline

# Create a B-spline curve in 3D with 5 control points
bspline = BSpline(n_control_points=5, dim=3, degree=3)

# Evaluate the curve at parameter values
u = torch.linspace(0, 1, 100)
points = bspline(u)  # Shape: (100, 3)

# Use in a neural network
import torch.nn as nn

class CurveNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 32)
        self.bspline = BSpline(n_control_points=6, dim=2)

    def forward(self, x):
        # Encode input to parameter values
        u = torch.sigmoid(self.encoder(x))
        # Generate points on the curve
        return self.bspline(u)

# The B-spline control points will be optimized during training
model = CurveNetwork()
optimizer = torch.optim.Adam(model.parameters())
```

## Advanced Usage

### Custom Knot Vectors

```python
# Create custom knot vector for non-uniform B-spline
knots = torch.tensor([0., 0., 0., 0., 0.3, 0.7, 1., 1., 1., 1.])
bspline = BSpline(n_control_points=6, dim=2, degree=3, knots=knots)
```

### Different Degrees

```python
# Linear B-spline (degree 1)
linear_spline = BSpline(n_control_points=4, dim=2, degree=1)

# Quadratic B-spline (degree 2)
quadratic_spline = BSpline(n_control_points=5, dim=2, degree=2)

# Cubic B-spline (degree 3) - default
cubic_spline = BSpline(n_control_points=6, dim=2, degree=3)
```

### Integration with Neural Networks

```python
class SplineAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, curve_dim=3):
        super().__init__()
        # Encoder: maps input to curve parameter
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Ensure output in [0, 1]
        )

        # B-spline decoder: maps parameter to curve point
        self.decoder = BSpline(
            n_control_points=8,
            dim=curve_dim,
            degree=3
        )

    def forward(self, x):
        u = self.encoder(x)
        return self.decoder(u)

# Train the model
model = SplineAutoencoder(input_dim=20, curve_dim=3)
data = torch.randn(100, 20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    points = model(data)
    loss = some_loss_function(points)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=torchcurves

# Run specific test file
pytest tests/test_bspline.py -v
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{torchcurves,
  author = {Shtoff, Alex},
  title = {torchcurves: Differentiable B-spline Curves in PyTorch},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/alexshtf/torchcurves}
}
```
