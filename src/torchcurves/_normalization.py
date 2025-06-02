import torch


def rational(x: torch.Tensor, scale: float = 1, out_min: float = -1, out_max: float = 1):
    r"""Normalize values using the "Legendre Rational Functions" [1] normalization method.

    The normalization is performed using the formula:
    :math::
        x_{\mathrm{norm}} = \frac{x}{\sqrt{\mathrm{scale}^2 + x^2}}

    where `scale` is a scaling factor.

    Args:
        x (torch.Tensor): Input tensor to be normalized.
        scale (float): Scale factor for normalization. (default=1)
        out_min (float): Lower bound of the output interval (default=-1)
        out_max (float): Upper bound of the output interval (default=1)

    Returns:
        torch.Tensor: Normalized tensor.

    References:
        [1]: Wang, Z.Q. and Guo, B.Y., 2004. Modified Legendre rational spectral method for the whole line.
            Journal of Computational Mathematics, pp.457-474.

    """
    result = x / torch.sqrt(scale**2 + x.square())
    out_scaled = ((out_max - out_min) * result + out_max + out_min) / 2
    return torch.clip(out_scaled, out_min, out_max)


def clamp(x: torch.Tensor, scale: float = 1, out_min: float = -1, out_max: float = 1):
    r"""Clamp values in a tensor to a specified range.

    The function clamps the values of the input tensor `x` to be within the output range, after scaling by the
    `scale` factor.

    Args:
        x (torch.Tensor): Input tensor to be normalized.
        scale (float): Scale factor for normalization. (default=1)
        out_min (float): Lower bound of the output interval (default=-1)
        out_max (float): Upper bound of the output interval (default=1)

    Returns:
        torch.Tensor: Clamped tensor.

    """
    return torch.clip(x * scale, out_min, out_max)


normalization_catalogue = {
    "rational": rational,
    "clamp": clamp,
}
