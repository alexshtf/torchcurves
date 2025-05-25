import torch


def rational(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    r"""Normalize values using the "Legendre Rational Functions" [1] normalization method.

    The normalization is performed using the formula:
    :math::
        x_{\mathrm{norm}} = \frac{x}{\sqrt{\mathrm{scale}^2 + x^2}}

    where `scale` is a scaling factor.

    Args:
        x (torch.Tensor): Input tensor to be normalized.
        scale (float): Scale factor for normalization.

    Returns:
        torch.Tensor: Normalized tensor.

    References:
        [1]: Wang, Z.Q. and Guo, B.Y., 2004. Modified Legendre rational spectral method for the whole line.
            Journal of Computational Mathematics, pp.457-474.

    """
    return x / torch.sqrt(scale**2 + torch.square(x))


def clamp(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    r"""Clamp values in a tensor to a specified range.

    The function clamps the values of the input tensor `x` to be within the range [0, 1], after scaling by the
    `scale` factor.

    Args:
        x (torch.Tensor): Input tensor to be clamped.
        scale (float): The scale factor before clamping.

    Returns:
        torch.Tensor: Clamped tensor.

    """
    return torch.clamp(x / scale, min=0, max=1)


normalizations = {
    "rational": rational,
    "clamp": clamp,
}
