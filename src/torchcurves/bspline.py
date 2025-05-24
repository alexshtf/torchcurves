from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ._normalization import normalization_catalogue
from .types import NormalizationFn


def uniform_augmented_knots(
    n_control_points: int, degree: int, dtype=torch.float32, device: Union[torch.device, str] = None
) -> torch.Tensor:
    """Generate an augmented knot vector with uniform spacing in [-1, 1] for B-spline curves.

    This function returns a 1D tensor containing knot values. The internal knots are computed uniformly in the interval
    [-1, 1] for the given number of control points and degree. The head and tail, each containing (degree + 1) identical
    knots, conforming to the not-a-knot boundary conditions.

    Args:
        n_control_points (int): The total number of control points for the B-spline.
                                Must be at least (degree + 1) to have a valid knot vector.
        degree (int): The degree of the B-spline.
        dtype (torch.dtype, optional): The desired data type of the output tensor.
                                       Defaults to torch.float32.
        device (torch.device or str): The device on which the knot vector will reside.

    Returns:
        torch.Tensor: A 1D tensor of knots consisting of head knots, uniformly spaced
                      internal knots, and tail knots, all in the range [-1.0, 1.0].

    Raises:
        ValueError: If the number of control points is less than (degree + 1), indicating
                    that there are not enough points to form a valid knot vector.

    """
    if n_control_points < 1 + degree:
        raise ValueError("Not enough control points for the given degree to form internal knots.")

    # Generates knots in [-1, 1]
    k_min, k_max = -1.0, 1.0  # Target range for normalized u

    head_knots = torch.full((degree + 1,), k_min, dtype=dtype, device=device)
    tail_knots = torch.full((degree + 1,), k_max, dtype=dtype, device=device)

    num_internal_knots = n_control_points - degree - 1
    if num_internal_knots == 0:
        internal_knots = torch.empty(0, dtype=dtype, device=device)
    else:
        internal_knots = torch.linspace(k_min, k_max, num_internal_knots + 2, dtype=dtype, device=device)[1:-1]

    return torch.cat([head_knots, internal_knots, tail_knots])


class BSplineFunction(torch.autograd.Function):
    ZERO_TOLERANCE = 1e-12
    ONE_TOLERANCE = 1.0 - ZERO_TOLERANCE  # Assuming u is normalized to [0,1] for these constants

    """Custom autograd function for B-spline evaluation and differentiation (Vectorized for multiple curves)."""

    @staticmethod
    def find_spans(u: torch.Tensor, knots: torch.Tensor, degree: int, n_control_points: int) -> torch.Tensor:
        """Find the knot span index for each parameter value (vectorized).

        Args:
            u: Parameter values, shape (N, M) or (N,). N samples, M curves.
               If u is (N,), it's treated as (N,1).
               Values are expected to be in the range defined by the knots (e.g., [0,1] or [-1,1]).
            knots: Knot vector, shape (num_total_knots,). Expected to be a clamped knot vector.
            degree: B-spline degree (p).
            n_control_points: Number of control points per curve (c).

        Returns:
            Span indices, shape (N, M) or (N,). Each span_idx `s` means u falls in [knots[s], knots[s+1]).

        """
        # Note: The original ZERO_TOLERANCE and ONE_TOLERANCE assumed u in [0,1] and knots clamped to [0,1].
        # If knots are e.g. [-1,1], this specific boundary handling might need adjustment
        # or u should be pre-normalized to [0,1] if this logic is to be kept strictly.
        # For now, we assume u is in the range [knots[degree], knots[n_control_points]].
        # The torch.searchsorted and clamp largely handle this.

        spans = torch.searchsorted(knots, u, side="right") - 1

        # Handle boundaries based on the actual knot values for robustness
        # This assumes knots is sorted and clamped: knots[0]..knots[degree] are same,
        # and knots[n_control_points]..knots[n_control_points+degree] are same.
        min_knot_val = knots[degree]
        max_knot_val = knots[n_control_points]  # This is the start of the last segment of p+1 knots

        # For u values at or slightly below the minimum parameter value
        spans[u <= min_knot_val + BSplineFunction.ZERO_TOLERANCE] = degree
        # For u values at or slightly above the maximum parameter value
        spans[u >= max_knot_val - BSplineFunction.ZERO_TOLERANCE] = n_control_points - 1

        spans = torch.clamp(spans, min=degree, max=n_control_points - 1)
        return spans

    @staticmethod
    def cox_de_boor(u: torch.Tensor, knots: torch.Tensor, spans: torch.Tensor, degree: int) -> torch.Tensor:
        """Compute B-spline basis functions using Cox-de Boor recursion.

        Args:
            u: Parameter values, shape (N, M). N samples, M curves.
            knots: Knot vector, shape (num_total_knots,).
            spans: Knot span indices, shape (N, M). `spans[n,m]` is `s`.
            degree: B-spline degree (p).

        Returns:
            Basis function values N_batch, shape (N, M, degree+1).
            N_batch[n, m, j] = B_{spans[n,m]-degree+j, degree}(u[n,m]).

        """
        num_samples_n, num_curves_m = u.shape
        device, dtype = u.device, u.dtype

        # batch_nonzero_basis[n, m, k] will store B_{spans[n,m]-degree+k, degree}(u[n,m])
        batch_nonzero_basis = torch.zeros(num_samples_n, num_curves_m, degree + 1, device=device, dtype=dtype)

        left_dist_all_p = torch.zeros(num_samples_n, num_curves_m, degree + 1, device=device, dtype=dtype)
        right_dist_all_p = torch.zeros(num_samples_n, num_curves_m, degree + 1, device=device, dtype=dtype)

        batch_nonzero_basis[..., 0] = 1.0

        for p_iter in range(1, degree + 1):  # p_iter is 'j' in Piegl & Tiller A2.2
            # knots is 1D. We gather using indices derived from spans (N,M)
            # Resulting shapes for left_dist_all_p, etc. will be (N,M)
            idx_knot_left = (spans + 1 - p_iter).clamp(min=0, max=knots.shape[0] - 1)
            left_dist_all_p[..., p_iter] = u - knots[idx_knot_left]

            idx_knot_right = (spans + p_iter).clamp(min=0, max=knots.shape[0] - 1)
            right_dist_all_p[..., p_iter] = knots[idx_knot_right] - u

            saved_val = torch.zeros(num_samples_n, num_curves_m, device=device, dtype=dtype)

            for r_iter in range(p_iter):
                denominator_batch = right_dist_all_p[..., r_iter + 1] + left_dist_all_p[..., p_iter - r_iter]

                ratios = batch_nonzero_basis[..., r_iter] / denominator_batch
                ratios = torch.where(torch.isfinite(ratios), ratios, torch.zeros_like(ratios))

                batch_nonzero_basis[..., r_iter] = saved_val + right_dist_all_p[..., r_iter + 1] * ratios
                saved_val = left_dist_all_p[..., p_iter - r_iter] * ratios

            batch_nonzero_basis[..., p_iter] = saved_val
        return batch_nonzero_basis

    @staticmethod
    def evaluate_curve(
        basis: torch.Tensor,  # shape (N, M, degree+1)
        control_points: torch.Tensor,  # shape (M, C, D) C=n_control_points
        spans: torch.Tensor,  # shape (N, M)
        degree: int,
    ) -> torch.Tensor:
        """Evaluate B-spline curves (vectorized for multiple curves).

        Args:
            basis: Basis function values. basis[n,m,j] = N_{spans[n,m]-degree+j, degree}(u[n,m]).
            control_points: Control points for M curves.
            spans: Knot span indices.
            degree: B-spline degree.

        Returns:
            Points on curves, shape (N, M, D).

        """
        num_samples_n, num_curves_m = spans.shape
        # C = num_control_points_per_curve, D = dim
        # M_cp, C_cp, D_cp = control_points.shape
        # Assert M_cp == num_curves_m

        # control_point_indices: indices into C dimension of control_points
        # Shape: (N, M, degree+1)
        degrees_range = torch.arange(degree + 1, device=spans.device).view(1, 1, -1)
        control_point_indices = spans.unsqueeze(-1) - degree + degrees_range

        # Clamp indices to be valid for control_points' C dimension
        clamped_cp_indices = torch.clamp(control_point_indices, 0, control_points.shape[1] - 1)

        # Gather control points: gathered_control_points[n, m, i, d] = control_points[m, clamped_cp_indices[n,m,i], d]
        # Need to create m_indices for gathering from control_points' M dimension
        # m_indices_for_gather shape: (N, M, degree+1)
        m_indices_for_gather = torch.arange(num_curves_m, device=control_points.device).view(1, -1, 1)
        m_indices_for_gather = m_indices_for_gather.expand(num_samples_n, -1, degree + 1)

        gathered_control_points = control_points[
            m_indices_for_gather,  # Selects the curve from M dimension of control_points
            clamped_cp_indices,  # Selects the control points from C dimension
            :,  # Selects all D dimensions
        ]  # Shape (N, M, degree+1, D)

        # Compute points: points[n,m,d] = sum_i basis[n,m,i] * gathered_control_points[n,m,i,d]
        # basis.unsqueeze(-1) gives (N, M, degree+1, 1)
        return (basis.unsqueeze(-1) * gathered_control_points).sum(dim=2)  # Sum over degree+1 dim

    @staticmethod
    def basis_derivative_coefficients(
        knots: torch.Tensor, spans: torch.Tensor, degree: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute coefficients for basis function derivatives (vectorized for multiple curves).

        Args:
            knots: Knot vector.
            spans: Knot span indices, shape (N, M).
            degree: B-spline degree (p).

        Returns:
            alpha_coeffs_batch, beta_coeffs_batch: shape (N, M, degree+1).

        """
        num_samples_n, num_curves_m = spans.shape
        device, dtype = spans.device, knots.dtype  # Use knot's dtype for coeffs

        degrees_range = torch.arange(degree + 1, device=device).view(1, 1, -1)
        knots_idx = spans.unsqueeze(-1) - degree + degrees_range  # (N, M, degree+1)

        # Gather knot values - knots[knots_idx] will broadcast correctly
        knots_k = knots[knots_idx.clamp(min=0, max=knots.shape[0] - 1)]
        knots_k_plus_deg = knots[(knots_idx + degree).clamp(min=0, max=knots.shape[0] - 1)]
        knots_k_plus_1 = knots[(knots_idx + 1).clamp(min=0, max=knots.shape[0] - 1)]
        knots_k_plus_deg_plus_1 = knots[(knots_idx + degree + 1).clamp(min=0, max=knots.shape[0] - 1)]

        alpha_coeffs_batch = torch.zeros(num_samples_n, num_curves_m, degree + 1, device=device, dtype=dtype)
        beta_coeffs_batch = torch.zeros(num_samples_n, num_curves_m, degree + 1, device=device, dtype=dtype)

        denom_alpha = knots_k_plus_deg - knots_k
        mask_alpha = torch.abs(denom_alpha) > BSplineFunction.ZERO_TOLERANCE
        alpha_coeffs_batch[mask_alpha] = degree / denom_alpha[mask_alpha]

        denom_beta = knots_k_plus_deg_plus_1 - knots_k_plus_1
        mask_beta = torch.abs(denom_beta) > BSplineFunction.ZERO_TOLERANCE
        beta_coeffs_batch[mask_beta] = degree / denom_beta[mask_beta]

        return alpha_coeffs_batch, beta_coeffs_batch

    @staticmethod
    def compute_basis_derivatives(
        u: torch.Tensor, knots: torch.Tensor, spans: torch.Tensor, degree: int
    ) -> torch.Tensor:
        """Compute derivatives of B-spline basis functions (vectorized for multiple curves).

        Output basis_deriv[n,m,i] = B'_{spans[n,m]-degree+i, degree}(u[n,m]).
        Shape: (N, M, degree+1)
        """
        if degree == 0:
            return torch.zeros(*u.shape, 1, device=u.device, dtype=u.dtype)

        # lower_deg_basis shape: (N, M, degree)
        lower_deg_basis = BSplineFunction.cox_de_boor(u, knots, spans, degree - 1)

        # alpha, beta have shape (N, M, degree+1)
        alpha, beta = BSplineFunction.basis_derivative_coefficients(knots, spans, degree)

        # Pad lower_deg_basis's last dimension to (degree+1)
        # Pad (0,1) means add 1 zero to the right: [N0,...,N(deg-1), 0]
        lower_pad_right = F.pad(lower_deg_basis, (0, 1))
        # Pad (1,0) means add 1 zero to the left: [0, N0,...,N(deg-1)]
        lower_pad_left = F.pad(lower_deg_basis, (1, 0))

        basis_deriv = alpha * lower_pad_left - beta * lower_pad_right
        return basis_deriv

    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,  # shape (N, M)
        control_points: torch.Tensor,  # shape (M, C, D)
        knots: torch.Tensor,  # shape (num_total_knots,)
        degree: int,
    ) -> torch.Tensor:
        # M_cp = control_points.shape[0] # Number of curves from control_points
        # N_u, M_u = u.shape             # N samples, M curves from u
        # Assert M_cp == M_u

        n_control_points_per_curve = control_points.shape[1]  # C

        spans = BSplineFunction.find_spans(u, knots, degree, n_control_points_per_curve)  # (N,M)
        basis_funcs = BSplineFunction.cox_de_boor(u, knots, spans, degree)  # (N,M,degree+1)
        points = BSplineFunction.evaluate_curve(basis_funcs, control_points, spans, degree)  # (N,M,D)

        ctx.save_for_backward(u, control_points, knots, spans, basis_funcs)
        ctx.degree = degree
        ctx.n_control_points_per_curve = n_control_points_per_curve  # C

        # For re-computing control_point_indices in backward
        degrees_range = torch.arange(degree + 1, device=spans.device).view(1, 1, -1)
        ctx.control_point_indices = spans.unsqueeze(-1) - degree + degrees_range  # (N,M,degree+1)

        return points

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        # grad_output shape: (N, M, D)
        u, control_points, knots, spans, basis_funcs = ctx.saved_tensors
        # u: (N,M), control_points: (M,C,D), knots: (K,), spans: (N,M), basis_funcs: (N,M,deg+1)

        degree = ctx.degree
        n_control_points_per_curve = ctx.n_control_points_per_curve  # C
        control_point_indices = ctx.control_point_indices  # (N,M,deg+1)

        num_samples_n, num_curves_m = u.shape
        # _, _, dim_d = grad_output.shape

        # Gradient with respect to u
        # basis_deriv shape: (N, M, degree+1)
        basis_deriv = BSplineFunction.compute_basis_derivatives(u, knots, spans, degree)

        clamped_cp_indices = torch.clamp(control_point_indices, 0, n_control_points_per_curve - 1)  # (N,M,deg+1)

        # Gather control points for d_points_du calculation
        # m_indices_for_gather shape: (N, M, degree+1)
        m_indices_for_gather = torch.arange(num_curves_m, device=u.device).view(1, -1, 1)
        m_indices_for_gather = m_indices_for_gather.expand(num_samples_n, -1, degree + 1)

        # gathered_cps shape: (N, M, degree+1, D)
        gathered_cps = control_points[m_indices_for_gather, clamped_cp_indices, :]

        # d_points_du[n,m,d] = sum_i basis_deriv[n,m,i] * gathered_cps[n,m,i,d]
        d_points_du = torch.einsum("nmi,nmid->nmd", basis_deriv, gathered_cps)  # Shape (N, M, D)

        # grad_u[n,m] = sum_d grad_output[n,m,d] * d_points_du[n,m,d]
        grad_u = (grad_output * d_points_du).sum(dim=-1)  # Shape (N, M)

        # Gradient with respect to control_points
        # grad_control_points shape: (M, C, D)
        grad_control_points = torch.zeros_like(control_points)

        # update_values[n,m,i,d] = grad_output[n,m,d] * basis_funcs[n,m,i]
        # grad_output.unsqueeze(2): (N,M,1,D)
        # basis_funcs.unsqueeze(3): (N,M,deg+1,1)
        update_values = grad_output.unsqueeze(2) * basis_funcs.unsqueeze(3)  # (N,M,deg+1,D)

        # Permute for scatter_add_: target grad_control_points[m_idx, c_idx, d_idx]
        # update_values: (N, M, deg+1, D) -> (M, N, deg+1, D)
        update_values_perm = update_values.permute(1, 0, 2, 3)
        # clamped_cp_indices: (N, M, deg+1) -> (M, N, deg+1)
        clamped_cp_indices_perm = clamped_cp_indices.permute(1, 0, 2)

        # Flatten N and deg+1 dimensions
        # uv_flat: (M, N*(deg+1), D)
        uv_flat = update_values_perm.reshape(num_curves_m, -1, grad_output.shape[-1])
        # idx_flat: (M, N*(deg+1))
        idx_flat = clamped_cp_indices_perm.reshape(num_curves_m, -1)

        # Expand idx_flat to match uv_flat for scatter_add_
        # idx_expanded_for_scatter: (M, N*(deg+1), D)
        idx_expanded_for_scatter = idx_flat.unsqueeze(-1).expand_as(uv_flat)

        # Scatter add along dimension C (index 1)
        grad_control_points.scatter_add_(1, idx_expanded_for_scatter, uv_flat)

        return grad_u, grad_control_points, None, None


class BSplineCurveBase(nn.Module):
    r"""Base PyTorch module for B-spline curves, supporting a batch of multiple curves.

    The learnable parameters are the control points for a batch of `num_curves`.
    Each curve in the batch shares the same degree and knot configuration.
    The input parameter `u` to the forward method is normalized to the range [-1, 1]
    (or the range of the knots if specified differently) using the specified normalization strategy.

    Args:
        num_curves (int): Number of B-spline curves to define in this module (m).
        dim (int): Dimension of each curve's output points (d).
        degree (int): Degree of the B-spline (p) (default: 3).
        knots_config (Union[int, torch.Tensor]):
            If an int, it specifies the number of control points per curve (c).
            A uniformly-spaced knot vector will be automatically generated in [-1, 1].
            If a torch.Tensor, it explicitly specifies the knot values. The number
            of control points will be inferred. The tensor should be 1D.
        normalize_fn (Literal["clamp", "rational"] | NormalizationFn):
            Normalization method for inputs `u`. (default: "clamp")
        normalization_scale (float):
            Scale factor for normalization (default: 1.0).

    """

    def __init__(
        self,
        num_curves: int,
        dim: int,
        degree: int = 3,
        knots_config: Union[int, torch.Tensor] = 10,  # This is n_control_points_per_curve
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

        self.num_curves = num_curves  # m
        self.dim = dim  # d
        self.degree = degree  # p

        if isinstance(normalize_fn, str):
            self.normalize_fn = normalization_catalogue.get(normalize_fn)
            if self.normalize_fn is None:
                raise ValueError(f"Unknown normalization {normalize_fn}")
        else:
            self.normalize_fn = normalize_fn

        self.normalization_scale = normalization_scale
        if self.normalization_scale <= 0:
            raise ValueError(f"Normalization scale must be positive, but {normalization_scale} was given.")

        if isinstance(knots_config, int):
            n_control_points_per_curve = knots_config  # c
        elif isinstance(knots_config, torch.Tensor):
            if knots_config.ndim != 1:
                raise ValueError("Provided knots_config tensor must be 1D.")
            num_knots_from_tensor = knots_config.shape[0]
            n_control_points_per_curve = num_knots_from_tensor - self.degree - 1
        else:
            raise TypeError(
                "knots_config must be an int (number of control points per curve) or a torch.Tensor (knot vector)."
            )

        if n_control_points_per_curve <= self.degree:
            raise ValueError(
                f"Number of control points per curve ({n_control_points_per_curve}) must be greater "
                f"than the degree ({self.degree})."
            )
        self.n_control_points_per_curve = n_control_points_per_curve  # c

        # Control points shape: (m, c, d)
        self.control_points = nn.Parameter(torch.empty(self.num_curves, self.n_control_points_per_curve, self.dim))
        nn.init.xavier_uniform_(self.control_points)

        if isinstance(knots_config, int):
            # Knots are shared by all m curves
            knot_buffer = uniform_augmented_knots(
                self.n_control_points_per_curve, self.degree, dtype=self.control_points.dtype
            )
        else:  # knots_config is a torch.Tensor
            knot_buffer = knots_config.to(dtype=self.control_points.dtype, copy=True)

        self.register_buffer("knots", knot_buffer)
        # Determine knot range for normalization, assuming knots are sorted.
        # Effective parameter range for B-spline is [knots[degree], knots[n_control_points_per_curve]]
        self._knot_min = self.knots[self.degree].item()
        self._knot_max = self.knots[self.n_control_points_per_curve].item()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_curves={self.num_curves}, "
            f"n_control_points_per_curve={self.n_control_points_per_curve}, "
            f"dim={self.dim}, degree={self.degree}, "
            f"knots_shape={self.knots.shape if hasattr(self, 'knots') else None})"
        )

    def _prepare_arg(self, u: torch.Tensor) -> torch.Tensor:
        return self.normalize_fn(u, self.normalization_scale, out_min=self._knot_min, out_max=self._knot_max)

    def forward(self, u: torch.Tensor):
        """Evaluate a batch of B-spline curves.

        Args:
            u (torch.Tensor): A tensor of parameter values, shape (N, num_curves).
                              N is the number of samples per curve.
                              u.shape[1] must match self.num_curves.

        Returns:
            torch.Tensor: Points on the B-spline curves, shape (N, num_curves, dim).

        """
        if u.ndim != 2 or u.shape[1] != self.num_curves:
            raise ValueError(
                f"Input u must be a 2D tensor of shape (N, num_curves={self.num_curves}). Got shape: {u.shape}"
            )

        u_prepared = self._prepare_arg(u)
        return self._forward_core(u_prepared)

    def _forward_core(self, u_prepared: torch.Tensor) -> torch.Tensor:
        # u_prepared has shape (N, M)
        # self.control_points has shape (M, C, D)
        # Should return tensor of shape (N, M, D)
        raise NotImplementedError("This method should be implemented in derived classes")


def bspline_curves(
    u: torch.Tensor, control_points: torch.Tensor, knots: Optional[torch.Tensor] = None, degree: int = 3
):
    r"""Evaluate multiple B-Spline curves, each with its own control points, sharing the same knots and degree.

    This function allow back-propagating both through the control points and the argument. Useful as a layer in
    a neural network.

    Args:
        u (torch.Tensor): A tensor of size B x C of values between ``knots.min()`` and ``knots.max()``, representing
            a mini-batch of ``B`` arguments for sampling each of the ``C`` curves.
        control_points (torch.Tensor): A tensor of size ``M x C x D`` describing ``M`` curves with ``C`` control
            points each, embedded in ``D``-dimensional space.
        knots (torch.Tensor, optional): A 1D tensor of size ``M + degree + 1`` representing the spline function's
            knot vector. ``None`` means uniformly-spaced knots between ``-1`` and ``1`` with the not-a-knot boundary
            conditions. (default: ``None``)
        degree (int): The degree of the B-Spline function. (default: ``3`` meaning a cubic spline)

    Returns:
        A tensor of size B x C x D, representing a mini-batch of size B, corresponding to samples from C curves in
        D-dimensional space.

    """
    if knots is None:
        n_control_points = control_points.shape[1]
        knots = uniform_augmented_knots(
            n_control_points, degree, dtype=control_points.dtype, device=control_points.device
        )

    return BSplineFunction.apply(
        u,
        control_points,
        knots,
        degree,
    )


def bspline_embeddings(
    u: torch.Tensor, control_points: torch.Tensor, knots: Optional[torch.Tensor] = None, degree: int = 3
):
    r"""Evaluate multiple B-Spline curves, each with its own control points, sharing the same knots and degree.

    This function allow back-propagating only through the control points and the argument. Useful as the input layer
    in a neural network, whose arguments come from a data-set that requires no back-prop, while allowing a cheaper
    computation for this usecase than `bspline_curves`.

    Args:
        u (torch.Tensor): A tensor of size B x C of values between ``knots.min()`` and ``knots.max()``, representing
            a mini-batch of ``B`` arguments for sampling each of the ``C`` curves.
        control_points (torch.Tensor): A tensor of size ``M x C x D`` describing ``M`` curves with ``C`` control
            points each, embedded in ``D``-dimensional space.
        knots (torch.Tensor, optional): A 1D tensor of size ``M + degree + 1`` representing the spline function's knot
            vector. ``None`` means uniformly-spaced knots between ``-1`` and ``1`` with the not-a-knot boundary
            conditions. (default: ``None``)
        degree (int): The degree of the B-Spline function. (default: ``3`` meaning a cubic spline)

    Returns:
        A tensor of size B x C x D, representing a mini-batch of size B, corresponding to samples from C curves in
        D-dimensional space.

    """
    n_control_points = control_points.shape[1]
    if knots is None:
        knots = uniform_augmented_knots(
            n_control_points, degree, dtype=control_points.dtype, device=control_points.device
        )

    spans = BSplineFunction.find_spans(u, knots, degree, n_control_points)  # (N,M)
    basis_funcs = BSplineFunction.cox_de_boor(u, knots, spans, degree)  # (N,M,deg+1)
    return BSplineFunction.evaluate_curve(basis_funcs, control_points, spans, degree)  # (N,M,D)


class BSplineEmbeddings(BSplineCurveBase):
    """PyTorch module for B-spline embeddings (batch of m curves, no backprop to the input).

    Learnable control points, no backpropagation through the curve parameter `u`.
    This means gradients are not computed for `u`. This module is useful as an embedding layer in a neural network,
    where `u` comes from a data-set, and no need to compute gradients w.r.t `u`. This facilitates a slightly faster
    evaluation of the B-Spline curve.
    """

    def _forward_core(self, u_prepared: torch.Tensor) -> torch.Tensor:
        return bspline_embeddings(u_prepared, self.control_points, self.knots, self.degree)


class BSplineCurve(BSplineCurveBase):
    """PyTorch module for a batch of B-Spline curves.

    Learnable control points, and backpropagation through the curve parameter `u`.
    """

    def _forward_core(self, u_prepared: torch.Tensor) -> torch.Tensor:
        return bspline_curves(u_prepared, self.control_points, self.knots, self.degree)
