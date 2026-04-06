from typing import Optional, Union

import torch
import torch.nn as nn

from ..functional import legendre_curves
from ..maps import resolve_input_map
from ..types import InputMap


class LegendreCurve(nn.Module):
    r"""PyTorch module for a batch of parametrized curves using Legendre polynomial basis.

    The learnable parameters are the control points (coefficients) of the
    `Legendre series <https://en.wikipedia.org/wiki/Legendre_polynomials>`_ for each curve.
    All curves share the same degree. The input of this layer is mapped to :math:`[-1, 1]`.
    Each curve is:

    .. math::

        \mathbf{C}_m(u) = \sum_{k=0}^{\mathrm{degree}} \mathbf{C}_{m,k} \cdot P_k(u),

    where :math:`P_k` is the :math:`k`-th Legendre polynomial.

    Args:
        num_curves: Number of Legendre curves to define (:math:`M`).
        dim: Dimension of each curve's output points (:math:`D`).
        degree: Degree of the Legendre polynomial basis (shared by all curves).
            The number of coefficients per curve will be `degree + 1`.
        input_map:
            Map from raw inputs to :math:`[-1, 1]`. Can be a dotted preset
            string like `"real.rational"`, a map object from
            `torchcurves.maps`, or a callable with signature
            `f(x, out_min, out_max)`.
        checkpoint_segments:
            Optional number of segments for gradient checkpointing. Larger values save memory but increase compute.
            Only used when gradients are enabled.

    """

    def __init__(
        self,
        num_curves: int,
        dim: int,
        degree: int,
        input_map: Union[str, InputMap] = "real.rational",
        checkpoint_segments: Optional[int] = None,
    ):
        super().__init__()

        if not isinstance(num_curves, int) or num_curves <= 0:
            raise ValueError("num_curves must be a positive integer.")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer.")
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer.")
        if checkpoint_segments is not None and (not isinstance(checkpoint_segments, int) or checkpoint_segments <= 0):
            raise ValueError("checkpoint_segments must be a positive integer or None.")

        self.num_curves = num_curves  # M
        self.dim = dim  # D
        self.degree = degree
        self.n_coefficients = self.degree + 1  # C (coefficients per curve)
        self.input_map = resolve_input_map(input_map)

        self.checkpoint_segments = checkpoint_segments

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

        u_normalized = self.input_map(u, -1.0, 1.0)
        return legendre_curves(
            u_normalized,
            self.coefficients,
            checkpoint_segments=self.checkpoint_segments,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_curves={self.num_curves}, "
            f"dim={self.dim}, degree={self.degree}, "
            f"n_coefficients_per_curve={self.n_coefficients}, "
            f"checkpoint_segments={self.checkpoint_segments})"
        )
