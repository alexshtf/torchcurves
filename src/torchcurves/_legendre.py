from typing import Literal, Union

import torch
import torch.nn as nn

from .utils import NormalizationFn, _normalization_catalogue


def _legendre_batch_curves(x, coefs):
    """Evaluate curves parametrized by Legendre polynomials using Clenshaw's recursion.

    Args:
        coefs: A tensor of size (n, c, m) of curve coefficients, of a set of c polynomial curves in m dimensions of
        degree n-1, represented in the Legendre basis.
        x: Batch of size (b, c), where x[:, j] is the batch of inputs for the j-th curve in the batch.

    Returns:
        A tensor of size (b, c, m) of the batch points on c curves in m-dimensions.

    """
    n, c, m = coefs.shape  # n - number of coefficients, c - number of curves, m - curve dimension
    x = x.unsqueeze(-1).expand(-1, -1, m)  # (b × c × m), b = batch size
    b2 = torch.zeros_like(x)  # (b × c × m)
    b1 = torch.zeros_like(x)  # (b × c × m)
    for k in reversed(range(n)):
        alpha = (2 * k + 1) / (k + 1)
        beta = (k + 1) / (k + 2)
        curr_coef = coefs[k].unsqueeze(0)  # (1 x c x m)
        bnext = torch.add(torch.addcmul(curr_coef, x, b1, value=alpha), b2, alpha=-beta)
        b2, b1 = b1, bnext
    return b1


def legendre_curves(x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Evaluate curves parametrized by Legendre polynomials.

    Args:
        coefficients: A tensor of size (N, C, M) of curve coefficients, of a set of C polynomial curves in M dimensions
        of degree N-1, represented in the Legendre basis.
        x: Batch of size (B, C), where x[:, j] is the batch of inputs for the j-th curve in the batch.

    Returns:
        points: Evaluated points on the curves, shape (B, C, M).

    """
    return _legendre_batch_curves(x, coefficients)


class LegendreCurve(nn.Module):
    r"""PyTorch module for a batch of parametrized curves using Legendre polynomial basis.

    The learnable parameters are the control points (coefficients) of the Legendre series for each curve.
    All curves share the same degree.
    The input parameter `u` to the forward method is normalized to [-1, 1].
    Each curve is C_m(u) = sum_{k=0}^{degree} CP_{m,k} * P_k(u_norm).

    Args:
        num_curves (int): Number of Legendre curves to define (M).
        dim (int): Dimension of each curve's output points (D).
        degree (int): Degree of the Legendre polynomial basis (shared by all curves).
                      The number of coefficients per curve will be `degree + 1`.
        normalize_fn (Union[Literal["clamp", "rational"], NormalizationFn]):
            Normalization method for inputs `u`. (default: "clamp")
        normalization_scale (float):
            Scale factor for normalization (default: 1.0).

    """

    def __init__(
        self,
        num_curves: int,
        dim: int,
        degree: int,
        normalize_fn: Union[Literal["clamp", "rational"], NormalizationFn] = "clamp",
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
            u (torch.Tensor): A 2D tensor of parameter values, shape (N, num_curves).
                              N is the number of samples per curve.
                              u.shape[1] must match self.num_curves.
                              Values will be normalized to [-1, 1].

        Returns:
            torch.Tensor: Points on the Legendre curves, shape (N, num_curves, dim).

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
