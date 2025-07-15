from typing import Literal, Union

import torch
import torch.nn as nn

from ..functional import legendre_curves
from ..types import NormalizationFn
from ._normalization import _normalization_catalogue


class LegendreCurve(nn.Module):
    r"""PyTorch module for a batch of parametrized curves using Legendre polynomial basis.

    The learnable parameters are the control points (coefficients) of the
    `Legendre series <https://en.wikipedia.org/wiki/Legendre_polynomials>`_ for each curve.
    All curves share the same degree. The input of this layer is normalized to :math:`[-1, 1]`.
    Each curve is:

    .. math::

        \mathbf{C}_m(u) = \sum_{k=0}^{\mathrm{degree}} \mathbf{C}_{m,k} \cdot P_k(u),

    where :math:`P_k` is the :math:`k`-th Legendre polynomial.

    Args:
        num_curves: Number of Legendre curves to define (:math:`M`).
        dim: Dimension of each curve's output points (:math:`D`).
        degree: Degree of the Legendre polynomial basis (shared by all curves).
            The number of coefficients per curve will be `degree + 1`.
        normalize_fn:
            Normalization method this layer's input. (default: "rational")
        normalization_scale (float):
            Scale factor for normalization (default: 1.0).

    """

    def __init__(
        self,
        num_curves: int,
        dim: int,
        degree: int,
        normalize_fn: Union[Literal["clamp", "rational"], NormalizationFn] = "rational",
        normalization_scale: float = 1.0,
    ):
        super().__init__()

        if not isinstance(num_curves, int) or num_curves <= 0:
            raise ValueError("num_curves must be a positive integer.")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer.")
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer.")

        self.num_curves = num_curves  # M
        self.dim = dim  # D
        self.degree = degree
        self.n_coefficients = self.degree + 1  # C (coefficients per curve)

        if isinstance(normalize_fn, str):
            normalize_fn_from_catalogue = _normalization_catalogue.get(normalize_fn)
            if normalize_fn_from_catalogue is None:
                raise ValueError(f"Unknown normalization {normalize_fn}")
            self.normalize_fn = normalize_fn_from_catalogue
        else:
            self.normalize_fn = normalize_fn

        self.normalization_scale = normalization_scale
        if self.normalization_scale <= 0:
            raise ValueError(f"Normalization scale must be positive, but {normalization_scale} was given.")

        # Coefficients shape: (M, C, D)
        self.coefficients = nn.Parameter(torch.empty(self.n_coefficients, self.num_curves, self.dim))
        nn.init.xavier_uniform_(self.coefficients)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Evaluate the batch of Legendre curves.

        Args:
            u: Parameter values of size :math:`(B, C)`, where :math:`B` is the mini-batch size, and `C` is the number
                of curves, and must be equal to `self.num_curves`.

        Returns:
            Points on the Legendre curves of shape :math:`(B, C, D)`.

        """
        if u.ndim != 2 or u.shape[1] != self.num_curves:
            raise ValueError(
                f"Input u must be a 2D tensor of shape (N, num_curves={self.num_curves}). Got shape: {u.shape}"
            )

        u_normalized = self.normalize_fn(u, self.normalization_scale, out_min=-1.0, out_max=1.0)
        return legendre_curves(u_normalized, self.coefficients)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_curves={self.num_curves}, "
            f"dim={self.dim}, degree={self.degree}, "
            f"n_coefficients_per_curve={self.n_coefficients})"
        )
