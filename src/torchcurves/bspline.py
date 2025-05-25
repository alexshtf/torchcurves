from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class BSplineFunction(torch.autograd.Function):
    ZERO_TOLERANCE = 1e-12
    ONE_TOLERANCE = 1.0 - ZERO_TOLERANCE

    """Custom autograd function for B-spline evaluation and differentiation (Vectorized)."""

    @staticmethod
    def find_spans(u: torch.Tensor, knots: torch.Tensor, degree: int, n_control_points: int) -> torch.Tensor:
        """Find the knot span index for each parameter value (vectorized).

        Args:
            u: Parameter values in [0, 1], shape (batch_size,).
            knots: Knot vector, shape (m,). Expected to be a clamped knot vector.
            degree: B-spline degree (p).
            n_control_points: Number of control points (n).

        Returns:
            Span indices, shape (batch_size,). Each span_idx `s` means u falls in [knots[s], knots[s+1]).

        """
        spans = torch.searchsorted(knots, u, side="right") - 1
        spans[u < BSplineFunction.ZERO_TOLERANCE] = degree
        spans[u >= BSplineFunction.ONE_TOLERANCE] = n_control_points
        spans = torch.clamp(spans, min=degree, max=n_control_points - 1)
        return spans

    @staticmethod
    def cox_de_boor(u: torch.Tensor, knots: torch.Tensor, spans: torch.Tensor, degree: int) -> torch.Tensor:
        """Compute B-spline basis functions using Cox-de Boor recursion. Algorithm A2.2 from Piegl & Tiller.

        Args:
            u: Parameter values, shape (batch_size,).
            knots: Knot vector, shape (m,).
            spans: Knot span indices, shape (batch_size,). `spans[b]` is `s`.
            degree: B-spline degree (p).

        Returns:
            Basis function values N_batch, shape (batch_size, degree+1).
            N_batch[b, j] = B_{spans[b]-degree+j, degree}(u[b]).

        """
        batch_size = u.shape[0]

        # batch_nonzero_basis[b, k] will store B_{spans[b]-degree+k, degree}(u[b])
        batch_nonzero_basis = torch.zeros(batch_size, degree + 1, device=u.device, dtype=u.dtype)

        left_dist_all_p = torch.zeros(batch_size, degree + 1, device=u.device, dtype=u.dtype)
        right_dist_all_p = torch.zeros(batch_size, degree + 1, device=u.device, dtype=u.dtype)

        # Initialize batch_nonzero_basis for degree 0: B_{s,0}(u) = 1.0 if u is in [knots_s, knots_{s+1}), else 0.
        batch_nonzero_basis[:, 0] = 1.0

        # Loop for degree p_iter from 1 to p (degree)
        for p_iter in range(1, degree + 1):  # p_iter is 'j' in Piegl & Tiller A2.2
            idx_knot_left = (spans + 1 - p_iter).clamp(min=0, max=knots.shape[0] - 1)
            left_dist_all_p[:, p_iter] = u - knots.gather(0, idx_knot_left)

            idx_knot_right = (spans + p_iter).clamp(min=0, max=knots.shape[0] - 1)
            right_dist_all_p[:, p_iter] = knots.gather(0, idx_knot_right) - u

            saved_val = torch.zeros(batch_size, device=u.device, dtype=u.dtype)

            # Loop for r_iter from 0 to p_iter-1 (r_iter is 'r' in Piegl & Tiller A2.2)
            for r_iter in range(p_iter):
                denominator_batch = right_dist_all_p[:, r_iter + 1] + left_dist_all_p[:, p_iter - r_iter]

                ratios = batch_nonzero_basis[:, r_iter] / denominator_batch
                ratios = torch.where(torch.isfinite(ratios), ratios, torch.zeros_like(ratios))

                batch_nonzero_basis[:, r_iter] = saved_val + right_dist_all_p[:, r_iter + 1] * ratios
                saved_val = left_dist_all_p[:, p_iter - r_iter] * ratios

            batch_nonzero_basis[:, p_iter] = saved_val

        return batch_nonzero_basis

    @staticmethod
    def evaluate_curve(
        basis: torch.Tensor,  # shape (batch_size, degree+1)
        control_points: torch.Tensor,  # shape (n_control_points, dim)
        spans: torch.Tensor,  # shape (batch_size,)
        degree: int,
    ) -> torch.Tensor:
        """Evaluate B-spline curve (vectorized).

        Args:
            basis: Basis function values. basis[b,j] = N_{spans[b]-degree+j, degree}(u[b]).
            control_points: Control points.
            spans: Knot span indices.
            degree: B-spline degree.

        Returns:
            Points on curve, shape (batch_size, dim).

        """
        n_control_points, dim = control_points.shape

        # Calculate indices of control points to use for each batch item
        # For each u[b] (in span s[b]), we need CP_{s[b]-degree}, ..., CP_{s[b]}
        # basis[b,i] corresponds to N_{s[b]-degree+i, degree}, which multiplies CP_{s[b]-degree+i}
        # cp_indices_batch[b, i] = spans[b] - degree + i
        degrees_range = torch.arange(degree + 1, device=spans.device).unsqueeze(0)  # Shape (1, degree+1)
        control_point_indices = spans.unsqueeze(1) - degree + degrees_range  # Shape (batch_size, degree+1)

        # Clamp indices to be valid for control_points tensor
        control_point_indices = torch.clamp(control_point_indices, 0, n_control_points - 1)

        # Gather control points: gathered_control_points[b, i, d] = control_points[control_point_indices[b,i], d]
        gathered_control_points = control_points[control_point_indices]  # Shape (batch_size, degree+1, dim)

        # Compute points: points[b,d] = sum_i basis[b,i] * gathered_control_points[b,i,d]
        points = (basis.unsqueeze(2) * gathered_control_points).sum(dim=1)

        return points

    @staticmethod
    def basis_derivative_coefficients(
        knots: torch.Tensor, spans: torch.Tensor, degree: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute coefficients for basis function derivatives.

           B'_{k,p}(u) = alpha_coeff * B_{k,p-1}(u) - beta_coeff * B_{k+1,p-1}(u)
           alpha_coeff = p / (knots[k+p] - knots[k])
           beta_coeff  = p / (knots[k+p+1] - knots[k+1])

        Args:
            knots: Knot vector.
            spans: Knot span indices, shape (batch_size,).
            degree: B-spline degree (p).

        Returns:
            alpha_coeffs_batch, beta_coeffs_batch: shape (batch_size, degree+1).

        """
        batch_size = spans.shape[0]
        degrees_range = torch.arange(degree + 1, device=spans.device).unsqueeze(0)

        # knot_idx_k_batch[b,i] = spans[b] - degree + i (this is 'k' in B'_{k,p})
        knots_idx = spans.unsqueeze(1) - degree + degrees_range  # (batch_size, degree+1)

        # Gather knot values
        knots_k = knots[knots_idx]
        knots_k_plus_deg = knots[knots_idx + degree]
        knots_k_plus_1 = knots[knots_idx + 1]
        knots_k_plus_deg_plus_1 = knots[knots_idx + degree + 1]

        alpha_coeffs_batch = torch.zeros(batch_size, degree + 1, device=spans.device, dtype=knots.dtype)
        beta_coeffs_batch = torch.zeros(batch_size, degree + 1, device=spans.device, dtype=knots.dtype)

        # Alpha: p / (knots[k+p] - knots[k])
        denom_alpha = knots_k_plus_deg - knots_k
        mask_alpha = torch.abs(denom_alpha) > BSplineFunction.ZERO_TOLERANCE
        alpha_coeffs_batch[mask_alpha] = degree / denom_alpha[mask_alpha]

        # Beta: p / (knots[k+p+1] - knots[k+1])
        denom_beta = knots_k_plus_deg_plus_1 - knots_k_plus_1
        mask_beta = torch.abs(denom_beta) > BSplineFunction.ZERO_TOLERANCE
        beta_coeffs_batch[mask_beta] = degree / denom_beta[mask_beta]

        return alpha_coeffs_batch, beta_coeffs_batch

    @staticmethod
    def compute_basis_derivatives(
        u: torch.Tensor, knots: torch.Tensor, spans: torch.Tensor, degree: int
    ) -> torch.Tensor:
        """Compute derivatives of B-spline basis functions (vectorized).

        Output basis_deriv[b,i] = B'_{spans[b]-degree+i, degree}(u[b]).
        """
        if degree == 0:
            return torch.zeros(u.shape[0], 1, device=u.device, dtype=u.dtype)

        # lower_deg_basis[b,j] = B_{spans[b]-(degree-1)+j, degree-1}(u[b])
        # Shape: (batch_size, degree)
        lower_deg_basis = BSplineFunction.cox_de_boor(u, knots, spans, degree - 1)

        # alpha, beta have shape (batch_size, degree+1)
        alpha, beta = BSplineFunction.basis_derivative_coefficients(knots, spans, degree)

        # Pad lower_deg_basis to align for multiplication
        lower_pad_left = F.pad(lower_deg_basis, (1, 0))  # Shape (batch_size, degree+1) -> [0, N0, ..., N(deg-1)]
        lower_pad_right = F.pad(lower_deg_basis, (0, 1))  # Shape (batch_size, degree+1) -> [N0, ..., N(deg-1), 0]

        basis_deriv = alpha * lower_pad_left - beta * lower_pad_right
        return basis_deriv

    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        control_points: torch.Tensor,
        knots: torch.Tensor,
        degree: int,
    ) -> torch.Tensor:
        n_control_points = control_points.shape[0]

        spans = BSplineFunction.find_spans(u, knots, degree, n_control_points)
        basis_funcs = BSplineFunction.cox_de_boor(u, knots, spans, degree)
        points = BSplineFunction.evaluate_curve(basis_funcs, control_points, spans, degree)

        ctx.save_for_backward(u, control_points, knots, spans, basis_funcs)
        ctx.degree = degree
        ctx.n_control_points = n_control_points

        # For re-computing control_point_indices in backward if needed, or pass them
        degree_range = torch.arange(degree + 1, device=spans.device).unsqueeze(0)
        ctx.control_point_indices = spans.unsqueeze(1) - degree + degree_range

        return points

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        u, control_points, knots, spans, basis_funcs = ctx.saved_tensors
        degree = ctx.degree
        n_control_points = ctx.n_control_points
        control_point_indices = ctx.control_point_indices  # (batch_size, degree+1)

        _, dim = grad_output.shape

        # Gradient with respect to u
        basis_deriv = BSplineFunction.compute_basis_derivatives(u, knots, spans, degree)  # Shape (batch_size, degree+1)

        clamped_cp_indices = torch.clamp(control_point_indices, 0, n_control_points - 1)
        gathered_cps = control_points[clamped_cp_indices]  # Shape (batch_size, degree+1, dim)

        # d_points_du[b,d] = sum_i basis_deriv[b,i] * gathered_cps[b,i,d]
        d_points_du = torch.einsum("bi,bid->bd", basis_deriv, gathered_cps)  # Shape (batch_size, dim)

        # grad_u[b] = sum_d grad_output[b,d] * d_points_du[b,d]
        grad_u = (grad_output * d_points_du).sum(dim=1)  # Shape (batch_size,)

        # Gradient with respect to control points
        grad_control_points = torch.zeros_like(control_points)

        # update_values[b,i,d] = grad_output[b,d] * basis_funcs[b,i]
        update_values = grad_output.unsqueeze(1) * basis_funcs.unsqueeze(2)
        # Shape: (batch_size, 1, dim) * (batch_size, degree+1, 1) -> (batch_size, degree+1, dim)

        # Flatten for scatter_add_
        # update_values_flat has shape (batch_size*(degree+1), dim)
        # indices_flat has shape (batch_size*(degree+1))
        update_values_flat = update_values.reshape(-1, dim)
        indices_flat = clamped_cp_indices.reshape(-1)

        # Expand indices_flat for scatter_add_ to match dim of update_values_flat
        # indices_flat_expanded shape (batch_size*(degree+1), dim)
        indices_flat_expanded = indices_flat.unsqueeze(1).expand_as(update_values_flat)

        grad_control_points.scatter_add_(0, indices_flat_expanded, update_values_flat)

        return grad_u, grad_control_points, None, None


class BSplineCurve(nn.Module):
    """PyTorch module for parametrized B-spline curves.

    The learnable parameters are the control points of the curve.
    The input parameter `u` to the forward method is always expected to be in the range [0, 1].

    Args:
        dim (int): Dimension of the curve (output dimension of points).
        degree (int): Degree of the B-spline (default: 3).
        knots_config (Union[int, torch.Tensor]):
            If an int, it specifies the number of control points. A clamped knot
            vector will be automatically generated (using uniformly spaced internal knots),
            ensuring the curve interpolates the first and last control points.
            If a torch.Tensor, it explicitly specifies the knot values. The number
            of control points will be inferred from the knot vector and degree.
            The provided tensor should be a 1D tensor of knot values.

    """

    def __init__(self, dim: int, degree: int = 3, knots_config: Union[int, torch.Tensor] = 10):
        super().__init__()

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer.")
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer.")

        self.dim = dim
        self.degree = degree

        if isinstance(knots_config, int):
            n_control_points = knots_config
        elif isinstance(knots_config, torch.Tensor):
            if knots_config.ndim != 1:
                raise ValueError("Provided knots_config tensor must be 1D.")
            num_knots_from_tensor = knots_config.shape[0]
            n_control_points = num_knots_from_tensor - self.degree - 1
        else:
            raise TypeError("knots_config must be an int (number of control points) or a torch.Tensor (knot vector).")

        if n_control_points <= self.degree:
            raise ValueError(
                f"Number of control points ({n_control_points}) must be greater than the degree ({self.degree}). "
                f"If providing a knot tensor, ensure its length is appropriate for the degree."
            )
        self.n_control_points = n_control_points

        self.control_points = nn.Parameter(torch.empty(self.n_control_points, self.dim))
        nn.init.xavier_uniform_(self.control_points)

        if isinstance(knots_config, int):
            knot_buffer = self._generate_clamped_knot_vector(
                self.n_control_points, self.degree, dtype=self.control_points.dtype
            )
        else:  # knots_config is a torch.Tensor
            knot_buffer = knots_config.to(dtype=self.control_points.dtype, copy=True)

        self.register_buffer("knots", knot_buffer)

    @staticmethod
    def _generate_clamped_knot_vector(n_control_points: int, degree: int, dtype=torch.float32) -> torch.Tensor:
        num_knots = n_control_points + degree + 1
        knots = torch.zeros(num_knots, dtype=dtype)

        knots[n_control_points:] = 1.0

        num_inner_knots = n_control_points - degree - 1
        if num_inner_knots > 0:
            inner_knots = torch.linspace(0, 1, n_control_points - degree + 1, dtype=dtype)
            knots[degree + 1 : n_control_points] = inner_knots[1:-1]

        return knots

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Evaluate the B-spline curve for a batch of parameter values u.

        Args:
            u (torch.Tensor): A 1D tensor of parameter values, shape (batch_size,).
                              Each value in u should be in the range [0, 1].

        Returns:
            torch.Tensor: Points on the B-spline curve, shape (batch_size, dim).

        """
        if u.ndim != 1:
            raise ValueError("Input u must be a 1D tensor (batch_size,).")

        u_clamped = torch.clamp(u, 0.0, 1.0)
        return BSplineFunction.apply(
            u_clamped,
            self.control_points,
            self.knots,  # type: ignore[arg-type] # self.knots is a Tensor buffer
            self.degree,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_control_points={self.n_control_points}, "
            f"dim={self.dim}, degree={self.degree}, "
            f"knots_shape={self.knots.shape if hasattr(self, 'knots') else None})"
        )
