from typing import Literal, Tuple

import torch
import torch.nn as nn

from ._normalization import normalization_catalogue
from .types import NormalizationFn


class LegendreCurveFunction(torch.autograd.Function):
    """Custom autograd function for Legendre polynomial curve evaluation and differentiation."""

    @staticmethod
    def _eval_legendre_polys(x: torch.Tensor, degree: int) -> torch.Tensor:
        """Evaluate Legendre polynomials P_k(x) for k from 0 to degree.

        Args:
            x: Input tensor of shape (N, M). N samples, M curves. Values typically in [-1, 1].
            degree: Maximum degree of Legendre polynomials to compute.

        Returns:
            basis_values: Tensor of shape (N, M, degree + 1) where basis_values[n, m, k] = P_k(x[n,m]).

        """
        num_samples_n, num_curves_m = x.shape
        device, dtype = x.device, x.dtype

        # basis_values[n, m, k] will store P_k(x[n,m])
        basis_values = torch.zeros(num_samples_n, num_curves_m, degree + 1, device=device, dtype=dtype)

        if degree >= 0:  # P_0(x) = 1
            basis_values[..., 0] = 1.0

        if degree >= 1:  # P_1(x) = x
            basis_values[..., 1] = x  # x has shape (N,M), basis_values[...,1] has shape (N,M)

        # Recurrence relations for n >= 1 for P_{n+1}(x)
        # (idx+1)P_{idx+1}(x) = (2*idx+1)xP_idx(x) - idx*P_{idx-1}(x)
        # Here, 'idx' refers to the degree of the polynomial.
        for poly_degree_idx in range(1, degree):  # Computes P_2 up to P_degree
            # x is (N,M), basis_values[..., poly_degree_idx] is (N,M)
            # x.unsqueeze(-1) would be (N,M,1) if we needed broadcasting with a (degree+1) dim, but not here.
            term_1 = (2 * poly_degree_idx + 1) * x * basis_values[..., poly_degree_idx]
            term_2 = poly_degree_idx * basis_values[..., poly_degree_idx - 1]
            basis_values[..., poly_degree_idx + 1] = (term_1 - term_2) / (poly_degree_idx + 1)

        return basis_values

    @staticmethod
    def _eval_legendre_derivs(degree: int, polys: torch.Tensor) -> torch.Tensor:
        """Evaluate derivatives of Legendre polynomials P'_k(x) using precomputed P_k(x).

        Args:
            degree: Maximum degree of Legendre polynomial derivatives to compute.
            polys: Precomputed Legendre polynomial values P_k(x), shape (N, M, degree + 1).

        Returns:
            basis_deriv_values: Tensor of shape (N, M, degree + 1)
                                where basis_deriv_values[n, m, k] = P'_k(x[n,m]).

        """
        num_samples_n, num_curves_m, _ = polys.shape
        device, dtype = polys.device, polys.dtype
        deriv_basis = torch.zeros(num_samples_n, num_curves_m, degree + 1, device=device, dtype=dtype)

        if degree >= 0:  # P'_0(x) = 0
            deriv_basis[..., 0] = 0.0

        if degree >= 1:  # P'_1(x) = 1
            deriv_basis[..., 1] = 1.0

        # Recurrence for P'_{idx+1}(x) = (2*idx+1)P_idx(x) + P'_{idx-1}(x)
        for poly_degree_idx in range(1, degree):  # Computes P'_2 up to P'_degree
            term_1 = (2 * poly_degree_idx + 1) * polys[..., poly_degree_idx]
            term_2 = deriv_basis[..., poly_degree_idx - 1]
            deriv_basis[..., poly_degree_idx + 1] = term_1 + term_2
        return deriv_basis

    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficients: torch.Tensor, degree: int) -> torch.Tensor:
        """Forward pass for Legendre curve (batched for multiple curves).

        Args:
            ctx: Context object to save tensors for backward pass.
            x: Parameter values, shape (N, M). N samples, M curves. Expected in [-1, 1].
            coefficients: Legendre polynomial coefficients, shape (M, C, D). C = degree + 1.
            degree: Degree of the Legendre polynomial basis.

        Returns:
            points: Evaluated points on the curves, shape (N, M, D).

        """
        basis_funcs = LegendreCurveFunction._eval_legendre_polys(x, degree)  # (N, M, C)

        # points[n,m,d] = sum_c basis_funcs[n,m,c] * control_points[m,c,d]
        points = torch.einsum("nmc,mcd->nmd", basis_funcs, coefficients)  # (N, M, D)

        ctx.save_for_backward(x, coefficients, basis_funcs)  # Save x for derivative calculation
        ctx.degree = degree
        return points

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Backward pass for Legendre curve (batched).

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Gradient of the loss w.r.t. output points, shape (N, M, D).

        Returns:
            grad_x: Gradient w.r.t. input x, shape (N, M).
            grad_control_points: Gradient w.r.t. control_points, shape (M, C, D).
            None: For degree.

        """
        x, control_points, basis_funcs = ctx.saved_tensors
        # x: (N,M), control_points: (M,C,D), basis_funcs: (N,M,C)
        degree = ctx.degree

        # Gradient w.r.t. x
        # P_k(x) are in basis_funcs. We need P'_k(x).
        deriv_basis = LegendreCurveFunction._eval_legendre_derivs(degree, basis_funcs)  # (N, M, C)

        # d_points_dx[n,m,d] = sum_c deriv_basis[n,m,c] * control_points[m,c,d]
        d_points_dx = torch.einsum("nmc,mcd->nmd", deriv_basis, control_points)  # (N, M, D)

        # grad_x[n,m] = sum_d grad_output[n,m,d] * d_points_dx[n,m,d]
        grad_x = (grad_output * d_points_dx).sum(dim=-1)  # (N, M)

        # Gradient w.r.t. control_points
        # grad_control_points[m,c,d] = sum_n basis_funcs[n,m,c] * grad_output[n,m,d]
        grad_control_points = torch.einsum("nmc,nmd->mcd", basis_funcs, grad_output)  # (M, C, D)

        return grad_x, grad_control_points, None


def legendre_curves(x: torch.Tensor, coefficients: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate Legendre curve (batched for multiple curves).

    Args:
        ctx: Context object to save tensors for backward pass.
        x: Parameter values, shape (N, M). N samples, M curves. Expected in [-1, 1].
        coefficients: Legendre polynomial coefficients, shape (M, C, D). C = degree + 1.
        degree: Degree of the Legendre polynomial basis. Will use (1 + degree) cofficients.

    Returns:
        points: Evaluated points on the curves, shape (N, M, D).

    """
    return LegendreCurveFunction.apply(x, coefficients, degree)


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
        normalize_fn (Literal["clamp", "rational"] | NormalizationFn):
            Normalization method for inputs `u`. (default: "clamp")
        normalization_scale (float):
            Scale factor for normalization (default: 1.0).

    """

    def __init__(
        self,
        num_curves: int,
        dim: int,
        degree: int,
        normalize_fn: Literal["clamp", "rational"] | NormalizationFn = "clamp",
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
            self.normalize_fn = normalization_catalogue.get(normalize_fn)
            if self.normalize_fn is None:
                raise ValueError(f"Unknown normalization {normalize_fn}")
        else:
            self.normalize_fn = normalize_fn

        self.normalization_scale = normalization_scale
        if self.normalization_scale <= 0:
            raise ValueError(f"Normalization scale must be positive, but {normalization_scale} was given.")

        # Coefficients shape: (M, C, D)
        self.coefficients = nn.Parameter(torch.empty(self.num_curves, self.n_coefficients, self.dim))
        nn.init.normal_(self.coefficients)

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

        # Normalize u to [-1, 1] as Legendre polynomials are typically defined on this interval.
        # The normalize_fn from _normalization.py by default normalizes to [-1,1]
        # if out_min/out_max are not specified or are -1,1.
        u_normalized = self.normalize_fn(u, self.normalization_scale, out_min=-1.0, out_max=1.0)

        # u_normalized has shape (N, M)
        # self.coefficients has shape (M, C, D)
        # self.degree is int
        return LegendreCurveFunction.apply(
            u_normalized,
            self.coefficients,
            self.degree,
        )  # Returns (N, M, D)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_curves={self.num_curves}, "
            f"dim={self.dim}, degree={self.degree}, "
            f"n_coefficients_per_curve={self.n_coefficients})"
        )
