import torch

from ..types import TensorLike


def _scale(x: torch.Tensor, in_min: float, in_max: float, out_min: float, out_max: float) -> torch.Tensor:
    if in_min == in_max and out_min == out_max:
        return x

    s = (out_max - out_min) / (in_max - in_min)
    offset = out_min - s * in_min
    return torch.add(offset, x, alpha=s)


def arctan(x: TensorLike, scale: float = 1, out_min: float = -1, out_max: float = 1) -> torch.Tensor:
    r"""Normalize values using the "arctan" normalization method.

    The normalization is performed with the formula

    .. math::
        x_{\mathrm{norm}} = \frac{2}{\pi} \arctan(x / \mathrm{scale}),

    where `scale` is a scaling factor.

    Args:
        x: Input tensor to be normalized.
        scale: Scale factor for normalization. (default=1)
        out_min: Lower bound of the output interval (default=-1)
        out_max: Upper bound of the output interval (default=1)

    """
    x = torch.as_tensor(x)
    tr = torch.arctan(x / scale)
    return _scale(tr, -torch.pi / 2, torch.pi / 2, out_min, out_max)


def rational(x: TensorLike, scale: float = 1, out_min: float = -1, out_max: float = 1) -> torch.Tensor:
    r"""Normalize values using the "Legendre Rational Functions" [1] normalization method.

    The normalization is performed with the formula

    .. math::
        x_{\mathrm{norm}} = \frac{x}{\sqrt{\mathrm{scale}^2 + x^2}},

    where `scale` is a scaling factor.

    Args:
        x: Input tensor to be normalized.
        scale: Scale factor for normalization. (default=1)
        out_min: Lower bound of the output interval (default=-1)
        out_max: Upper bound of the output interval (default=1)

    Returns:
        Normalized tensor.

    **References**

    [1] Wang, Z.Q. and Guo, B.Y., 2004.
    *Modified Legendre rational spectral method for the whole line.*
    Journal of Computational Mathematics, pp.457-474.

    """
    x = torch.as_tensor(x)
    result = x / torch.sqrt(scale**2 + x.square())
    return _scale(result, -1, 1, out_min, out_max)


def clamp(x: TensorLike, scale: float = 1, out_min: float = -1, out_max: float = 1) -> torch.Tensor:
    r"""Clamp values in a tensor to a specified range.

    The function clamps the values of the input tensor `x` to be within the output range, after scaling by the
    `scale` factor, by the formula:

    .. math::
        x_{\mathrm{norm}} = \min(1, \max(0, x / \mathrm{scale}))

    Args:
        x: Input tensor to be normalized.
        scale: Scale factor for normalization. (default=1)
        out_min: Lower bound of the output interval (default=-1)
        out_max: Upper bound of the output interval (default=1)

    Returns:
        Normalized tensor.

    """
    x = torch.as_tensor(x)
    return torch.clip(x / scale, out_min, out_max)
