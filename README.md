# torchcurves

![torchcurves logo](https://raw.githubusercontent.com/alexshtf/torchcurves/master/logo.png)

A PyTorch module for differentiable parametric curves with learnable coefficients,
such as a B-Spline curve with learnable control points.

This package provides fully differentiable curve implementations that integrate
seamlessly with PyTorch's autograd system. It streamlines use cases such as
continuous numerical embeddings for factorization machines [6] or transformers
[2,3], Kolmogorov-Arnold networks [1], or path planning in robotics.

Full documentation is available at
[torchcurves.readthedocs.io](https://torchcurves.readthedocs.io/en/latest/).

## Features

- **Fully Differentiable**: Custom autograd function ensures gradients flow
  properly through the curve evaluation.
- **Batch Processing**: Vectorized operations for efficient batch evaluation.

## Installation

```bash
pip install torchcurves
```

```bash
uv add torchcurves
```

## Use cases

There are examples in the `examples` directory showing how to build models using
this library. Here we show some simple code snippets to appreciate the library.

## Use case 1 - continuous embeddings

```python
import torchcurves as tc
from torch import nn
import torch


def Net(nn.Module):
    def __init__(self, num_categorical, num_numerical, dim, num_knots=10):
        super().__init__()
        self.cat_emb = nn.Embedding(num_categorical, dim)
        self.num_emb = tc.BSplineEmbeddings(num_numerical, dim, knots_config=num_knots)
        self.my_super_duper_transformer = MySuperDuperTransformer()

    def forward(self, x_categorical, x_numerical):
        embeddings = torch.cat([self.cat_emb(x_categorical), self.num_emb(x_numerical)], axis=-2)
        return self.my_super_duper_transformer(embeddings)
```

## Use case 2 - Kolmogorov-Arnold networks

A KAN [1] based on the B-Spline basis, along the lines of the original paper:

```python
import torchcurves as tc
from torch import nn

input_dim = 2
intermediate_dim = 5
num_control_points = 10

kan = nn.Sequential(
    # layer 1
    tc.BSplineCurve(input_dim, intermediate_dim, knots_config=num_control_points),
    tc.Sum(dim=-2),
    # layer 2
    tc.BSplineCurve(intermediate_dim, intermediate_dim, knots_config=num_control_points),
    tc.Sum(dim=-2),
    # layer 3
    tc.BSplineCurve(intermediate_dim, 1, knots_config=num_control_points),
    tc.Sum(dim=-2),
)
```
Yes, we know the original KAN paper used a different curve parametrization,
B-Spline + arcsinh, but the whole point of this repo is showing that KAN
activations can be parametrized in arbitrary ways.

For example, here is a KAN based on Legendre polynomials of degree 5:

```python
import torchcurves as tc
from torch import nn

input_dim = 2
intermediate_dim = 5
degree = 5

kan = nn.Sequential(
    # layer 1
    tc.LegendreCurve(input_dim, intermediate_dim, degree=degree),
    tc.Sum(dim=-2),
    # layer 2
    tc.LegendreCurve(intermediate_dim, intermediate_dim, degree=degree),
    tc.Sum(dim=-2),
    # layer 3
    tc.LegendreCurve(intermediate_dim, 1, degree=degree),
    tc.Sum(dim=-2),
)
```

Since KANs are the primary use case for the `tc.Sum()` layer, we can omit the `dim=-2` argument, but it is provided
here for clarity.

## Advanced features

The curves we provide here typically rely on their inputs to lie in a compact
interval, typically [-1, 1]. Arbitrary inputs need to be normalized to this
interval. We provide two simple out-of-the-box normalization strategies
described below.

## Rational scaling

This is the default strategy — this strategy computes

```math
x \to \frac{x}{\sqrt{s^2 + x^2}},
```

and is based on the paper
>Wang, Z.Q. and Guo, B.Y., 2004. Modified Legendre rational spectral method for the whole line. Journal of Computational Mathematics, pp.457-474.

In Python it looks like this:

```python
tc.BSplineCurve(curve_dim, normalization_fn='rational', normalization_scale=s)
```

## Clamping

The inputs are simply clipped to [-1, 1] after scaling, i.e.

```math
x \to \max(\min(1, x / s), -1)
```

In Python it looks like this:

```python
tc.BSplineCurve(curve_dim, normalization_fn='clamp', normalization_scale=s)
```

## Custom normalization

Provide a custom function that maps its input to the designated range after
scaling. Example:

```python
def erf_clamp(x: Tensor, scale: float = 1, out_min: float = -1, out_max: float = 1) -> Tensor:
    mapped = torch.special.erf(x / scale)
    return ((mapped + 1) * (out_max - out_min)) / 2 + out_min

tc.BSplineCurve(curve_dim, normalization_fn=erf_clamp, normalization_scale=s)
```

## Example: B-Spline KAN with clamping

A KAN based on rationally scaled B-Spline basis with the default scale of $s=1$:

```python
spline_kan = nn.Sequential(
    # layer 1
    tc.BSplineCurve(input_dim, intermediate_dim, knots_config=knots, normalization_fn='clamp'),
    tc.Sum(),
    # layer 2
    tc.BSplineCurve(intermediate_dim, intermediate_dim, knots_config=knots, normalization_fn='clamp'),
    tc.Sum(),
    # layer 3
    tc.BSplineCurve(intermediate_dim, 1, knots_config=knots, normalization_fn='clamp'),
    tc.Sum(),
)
```

### Legendre KAN with rational clamping

```python
import torchcurves as tc
from torch import nn

input_dim = 2
intermediate_dim = 5
degree = 5

config = dict(degree=degree, normalization_fn="clamp")
kan = nn.Sequential(
    # layer 1
    tc.LegendreCurve(input_dim, intermediate_dim, **config),
    tc.Sum(),
    # layer 2
    tc.LegendreCurve(intermediate_dim, intermediate_dim, **config),
    tc.Sum(),
    # layer 3
    tc.LegendreCurve(intermediate_dim, 1, **config),
    tc.Sum(),
)
```


## Development

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

## Building the docs

```bash
# Prepare API docs
cd docs
make html
```

## Citation

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

## References

[1]: Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljacic, Thomas Y. Hou, Max Tegmark. "KAN: Kolmogorov–Arnold Networks." *ICLR* (2025). \
[2]: Juergen Schmidhuber. "Learning to control fast-weight memories: An alternative to dynamic recurrent networks." *Neural Computation*, 4(1), pp.131-139. (1992) \
[3]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." *Advances in neural information processing systems* 30 (2017). \
[4]: Alex Shtoff, Elie Abboud, Rotem Stram, and Oren Somekh. "Function Basis Encoding of Numerical Features in Factorization Machines." *Transactions on Machine Learning Research*. \
[5]: Rügamer, David. "Scalable Higher-Order Tensor Product Spline Models." In *International Conference on Artificial Intelligence and Statistics*, pp. 1-9. PMLR, 2024. \
[6]: Steffen Rendle. "Factorization machines." In *2010 IEEE International conference on data mining*, pp. 995-1000. IEEE, 2010.
