# TorchCurves

A PyTorch module for differentiable parametric with learnable coefficients, such as a B-Spline curve with learnable control points.

This package provides a fully differentiable curve implementations that integrates seamlessly with PyTorch's autograd system, streamlining use cases such as continuous numerical embeddings for transformers, Kolmogorov-Arnold networks, or path planning in robotics.

## Features

- **Fully Differentiable**: Custom autograd function ensures gradients flow properly through the curve evaluation.
- **Batch Processing**: Vectorized operations for efficient batch evaluation

## Installation

```bash
pip install torchcurves
```

# Use case 1 - continuous embeddings
```python
import torchcurves as tc
from torch import nn
import torch


def Net(nn.Module):
    def __init__(self, num_categorical, num_numerical, dim, num_knots=10):
        self.cat_emb = nn.Embedding(num_categorical, dim)
        self.num_emb = tc.BSplineEmbeddings(num_numerical, dim, knot_config=num_knots)
        self.my_super_duper_transformer = MySuperDuperTransformer()

    def forward(self, x_categorical, x_numerical):
        embeddings = torch.cat([self.cat_emb(x_categorical), self.num_emb(x_numerical)], axis=-2)
        return self.my_super_duper_transformer(embeddings)
```

# Use case 2 - Kolmogorov-Arnold networks
A KAN based on the B-Spline basis, along the lines of the original paper:
```python
import torchcurves as tc
from torch import nn

input_dim = 2
intermediate_dim = 5
knots = 10

kan = nn.Sequential(
    # layer 1
    tc.BSplineCurve(input_dim, intermediate_dim, knots_config=knots, normalize_fn='rational'),
    tc.Sum(dim=-2),
    # layer 2
    tc.BSplineCurve(intermediate_dim, intermediate_dim, knots_config=knots, normalize_fn='rational'),
    tc.Sum(dim=-2),
    # layer 3
    tc.BSplineCurve(intermediate_dim, 1, knots_config=knots, normalize_fn='rational'),
    tc.Sum(dim=-2),
)
```
Yes, we know the original KAN paper used a different curve parametrization, B-Spline + arcsinh, but the whole point
of this repo is showing that KAN activations can be parametrized in arbitrary ways.

For example, here is A KAN based on Legendre polynomials of degree 5, with rational normalization:
```python
import torchcurves as tc
from torch import nn

input_dim = 2
intermediate_dim = 5
degree = 5

kan = nn.Sequential(
    # layer 1
    tc.LegendreCurve(input_dim, intermediate_dim, degree=degree, normalize_fn='rational'),
    tc.Sum(dim=-2),
    # layer 2
    tc.LegendreCurve(intermediate_dim, intermediate_dim, degree=degree, normalize_fn='rational'),
    tc.Sum(dim=-2),
    # layer 3
    tc.LegendreCurve(intermediate_dim, 1, degree=degree, normalize_fn='rational'),
    tc.Sum(dim=-2),
)
```
Since KANs are the primary use-case for the `tc.Sum()` layer, we can omit the `dim=2` argument, but it is provided
here for clarity.

# Advanced features
The curves we provide here typically rely on their inputs to lie in a compact interval, typically [-1, 1]. So arbitrary
inputs need to be normalized to this interval. We provide two simple out-of-the-box normalizations stragies described
below.

## Clamping
This is the default strategy - the inputs are simply clipped to [-1, 1] after scaling, i.e.
```math
x \to \max(\min(1, x / s), -1)
```
In Python it looks like this:
```python
tc.BSplineCurve(curve_dim, normalization_fn='clamp', normalization_scale=s)
```

## Rational scaling
This strategy computes
```math
x \to \frac{x}{\sqrt{s^2 + x^2}},
```
and is based on the paper
>Wang, Z.Q. and Guo, B.Y., 2004. Modified Legendre rational spectral method for the whole line. Journal of Computational Mathematics, pp.457-474.

In Python it looks like this:
```python
tc.BSplineCurve(curve_dim, normalization_fn='rational', normalization_scale=s)
```

## Example
A KAN based on rationally-scaled B-Spline basis with the default scale of $s=1$:
```python
spline_kan = nn.Sequential([
    # layer 1
    tc.BSplineCurve(input_dim, intermediate_dim, knot_config=knots, normalization_fn='rational'),
    tc.Sum()
    # layer 2
    tc.BSplineCurve(intermediate_dim, intermediate_dim, knot_config=knots, normalization_fn='rational'),
    tc.Sum()
    # layer 3
    tc.BSplineCurve(intermediate_dim, 1, knot_config=knots, normalization_fn='rational'),
    tc.Sum()
])
```

# Development


## Development Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Clone the repository
git clone https://github.com/alexshtf/torchcurves.git
cd torchcurves

# Create virtual environment and install
uv venv
uv sync --all-groups
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=torchcurves

# Run specific test file
uv run pytest tests/test_bspline.py -v
```

# Citation

If you use this package in your research, please cite:

```bibtex
@software{torchcurves,
  author = {Shtoff, Alex},
  title = {torchcurves: Differentiable Parametric Curves in PyTorch},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/alexshtf/torchcurves}
}
```
