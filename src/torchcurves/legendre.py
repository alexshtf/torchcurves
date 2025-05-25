from typing import Tuple

import torch
import torch.nn as nn


class LegendreCurveFunction(torch.autograd.Function):
    """Custom autograd function for Legendre polynomial curve evaluation and differentiation."""

    @staticmethod
    def _eval_legendre_polys(x: torch.Tensor, degree: int) -> torch.Tensor:
        """Evaluate Legendre polynomials P_k(x) for k from 0 to degree.

        Uses recurrence relations:
            P_0(x) = 1
            P_1(x) = x
            (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x) for n >= 1

        Args:
            x: Input tensor of shape (batch_size,). Values typically in [-1, 1].
            degree: Maximum degree of Legendre polynomials to compute.

        Returns:
            basis_values: Tensor of shape (batch_size, degree + 1) where basis_values[:, k] = P_k(x).

        """
        batch_size = x.shape[0]
        # basis_values[:, k] will store P_k(x)
        basis_values = torch.zeros(batch_size, degree + 1, device=x.device, dtype=x.dtype)

        # P_0(x) = 1
        if degree >= 0:
            basis_values[:, 0] = 1.0

        if degree >= 1:
            # P_1(x) = x
            basis_values[:, 1] = x

        # Recurrence relations for n >= 1 for P_{n+1}(x)
        for n in range(1, degree):  # Computes P_2 up to P_degree
            term_1 = (2 * n + 1) * x * basis_values[:, n]
            term_2 = n * basis_values[:, n - 1]
            basis_values[:, n + 1] = (term_1 - term_2) / (n + 1)

        return basis_values

    @staticmethod
    def _eval_legendre_derivs(degree: int, polys: torch.Tensor) -> torch.Tensor:
        """Evaluate derivatives of Legendre polynomials P'_k(x) using precomputed P_k(x).

        Uses recurrence relations:
            P'_0(x) = 0
            P'_1(x) = 1
            P'_{n+1}(x) = (2n+1)P_n(x) + P'_{n-1}(x) for n >= 1

        Args:
            degree: Maximum degree of Legendre polynomial derivatives to compute.
            polys: Precomputed Legendre polynomial values P_k(x),
                   shape (batch_size, degree + 1).

        Returns:
            basis_deriv_values: Tensor of shape (batch_size, degree + 1)
                                where basis_deriv_values[:, k] = P'_k(x).

        """
        batch_size = polys.shape[0]
        deriv_basis = torch.zeros(batch_size, degree + 1, device=polys.device, dtype=polys.dtype)

        # P'_0(x) = 0
        if degree >= 0:
            deriv_basis[:, 0] = 0.0

        # P'_1(x) = 1
        if degree >= 1:
            deriv_basis[:, 1] = 1.0

        # Recurrence for P'_{n+1}(x) = (2n+1)P_n(x) + P'_{n-1}(x)
        for n in range(1, degree):  # Computes P'_2 up to P'_degree
            term_1 = (2 * n + 1) * polys[:, n]
            term_2 = deriv_basis[:, n - 1]
            deriv_basis[:, n + 1] = term_1 + term_2
        return deriv_basis

    @staticmethod
    def forward(ctx, x: torch.Tensor, control_points: torch.Tensor, degree: int) -> torch.Tensor:
        """Forward pass for Legendre curve.

        Args:
            ctx: Context object to save tensors for backward pass.
            x: Parameter values, shape (batch_size,). Expected in [-1, 1].
               This is the direct argument for P_k(x).
            control_points: Control points, shape (degree + 1, dim).
            degree: Degree of the Legendre polynomial basis.

        """
        basis_funcs = LegendreCurveFunction._eval_legendre_polys(x, degree)
        points = torch.matmul(basis_funcs, control_points)
        ctx.save_for_backward(control_points, basis_funcs)
        ctx.degree = degree
        return points

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Backward pass for Legendre curve.

        Args:
            ctx: Context object containing saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output points, shape (batch_size, dim).

        """
        control_points, basis_funcs = ctx.saved_tensors
        degree = ctx.degree

        deriv_basis = LegendreCurveFunction._eval_legendre_derivs(degree, basis_funcs)
        dout_dx = torch.matmul(deriv_basis, control_points)
        grad_x = (grad_output * dout_dx).sum(dim=1)

        grad_control_points = torch.matmul(basis_funcs.transpose(0, 1), grad_output)

        return grad_x, grad_control_points, None


class LegendreCurve(nn.Module):
    """PyTorch module for parametrized curves using Legendre polynomial basis.

    The learnable parameters are the control points (coefficients) of the Legendre series.
    The input parameter `u` to the forward method is always expected to be in the range [-1, 1].
    The curve is C(u) = sum_{k=0}^{degree} CP_k * P_k(u).

    Args:
        dim (int): Dimension of the curve (output dimension of points).
        degree (int): Degree of the Legendre polynomial basis.
                      The number of control points will be `degree + 1`.

    """

    def __init__(self, dim: int, degree: int):
        super().__init__()

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer.")
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer.")

        self.dim = dim
        self.degree = degree
        self.n_control_points = self.degree + 1

        self.control_points = nn.Parameter(torch.empty(self.n_control_points, self.dim))
        nn.init.xavier_uniform_(self.control_points)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Evaluate the Legendre curve for a batch of parameter values u.

        Args:
            u (torch.Tensor): A 1D tensor of parameter values, shape (batch_size,).
                              Each value in u should be in the range [-1, 1].

        Returns:
            torch.Tensor: Points on the Legendre curve, shape (batch_size, dim).

        """
        if u.ndim != 1:
            raise ValueError("Input u must be a 1D tensor (batch_size,).")

        # Clamp u to [-1, 1] as this is the direct input for Legendre polynomials
        u_clamped = torch.clamp(u, -1.0, 1.0)

        return LegendreCurveFunction.apply(
            u_clamped,
            self.control_points,
            self.degree,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dim={self.dim}, degree={self.degree}, n_control_points={self.n_control_points})"
        )


class LegendreRationalCurve(nn.Module):
    """PyTorch module for parametrized curves using Legendre Rational function basis.

    The learnable parameters are the control points (coefficients) of the series.
    The input parameter `x` to the forward method can be any real number.
    This `x` is mapped to `y = x / sqrt(1 + x^2)`, which is in [-1, 1].
    The curve is C(x) = sum_{k=0}^{degree} CP_k * P_k(y).

    Args:
        dim (int): Dimension of the curve (output dimension of points).
        degree (int): Degree of the Legendre polynomial basis.
                      The number of control points will be `degree + 1`.

    """

    def __init__(self, dim: int, degree: int):
        super().__init__()

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer.")
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer.")

        self.dim = dim
        self.degree = degree
        self.n_control_points = self.degree + 1

        self.control_points = nn.Parameter(torch.empty(self.n_control_points, self.dim))
        nn.init.xavier_uniform_(self.control_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the Legendre Rational curve for a batch of parameter values x.

        Args:
            x (torch.Tensor): A 1D tensor of parameter values, shape (batch_size,).
                              Values can be any real number.

        Returns:
            torch.Tensor: Points on the Legendre Rational curve, shape (batch_size, dim).

        """
        if x.ndim != 1:
            raise ValueError("Input x must be a 1D tensor (batch_size,).")

        y = x / torch.sqrt(1 + x.square())
        return LegendreCurveFunction.apply(
            y,
            self.control_points,
            self.degree,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dim={self.dim}, degree={self.degree}, n_control_points={self.n_control_points})"
        )
