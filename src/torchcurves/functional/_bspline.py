from typing import Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F  # noqa: N812


def uniform_augmented_knots(
    n_control_points: int,
    degree: int,
    dtype=torch.float32,
    device: Union[torch.device, str, None] = None,
    k_min: float = -1,
    k_max: float = 1,
) -> torch.Tensor:
    """Generate an augmented knot vector with uniform spacing in [-1, 1] for B-spline curves.

    This function returns a 1D tensor containing knot values. The internal knots are computed uniformly in the interval
    [-1, 1] for the given number of control points and degree. The head and tail, each containing (degree + 1) identical
    knots, produce an augmented knot vector with repeated boundary knots.

    Args:
        n_control_points (int): The total number of control points for the B-spline.
                                Must be at least (degree + 1) to have a valid knot vector.
        degree (int): The degree of the B-spline.
        dtype (torch.dtype, optional): The desired data type of the output tensor.
                                       Defaults to torch.float32.
        device (torch.device or str): The device on which the knot vector will reside.
        k_min (float, optional): The minimum value for the internal knots. Defaults to -1.0.
        k_max (float, optional): The maximum value for the internal knots. Defaults to 1.0.

    Returns:
        torch.Tensor: A 1D tensor of knots consisting of head knots, uniformly spaced
            internal knots, and tail knots, all in the range [-1.0, 1.0].

    Raises:
        ValueError: If the number of control points is less than (degree + 1), indicating
            that there are not enough points to form a valid knot vector.

    """
    num_internal_knots = n_control_points - degree - 1
    if num_internal_knots < 0:
        raise ValueError("Not enough control points for the given degree to form internal knots.")

    head_knots = torch.full((degree + 1,), k_min, dtype=dtype, device=device)
    tail_knots = torch.full((degree + 1,), k_max, dtype=dtype, device=device)

    if num_internal_knots > 0:
        internal_knots = torch.linspace(k_min, k_max, num_internal_knots + 2, dtype=dtype, device=device)[1:-1]
        return torch.cat((head_knots, internal_knots, tail_knots))
    else:
        return torch.cat((head_knots, tail_knots))


class _BSplineFunction(torch.autograd.Function):
    ZERO_TOLERANCE = 1e-12

    """Custom autograd function for vectorized B-spline evaluation."""

    @staticmethod
    def _control_point_indices(spans: torch.Tensor, degree: int) -> torch.Tensor:
        offsets = torch.arange(-degree, 1, device=spans.device).view(1, 1, -1)
        return spans.unsqueeze(-1) + offsets

    @staticmethod
    def _curve_indices(num_samples: int, num_curves: int, width: int, device: torch.device) -> torch.Tensor:
        return torch.arange(num_curves, device=device).view(1, -1, 1).expand(num_samples, -1, width)

    @staticmethod
    def _gather_control_points(control_points: torch.Tensor, cp_indices: torch.Tensor) -> torch.Tensor:
        num_samples, num_curves, width = cp_indices.shape
        curve_indices = _BSplineFunction._curve_indices(
            num_samples=num_samples,
            num_curves=num_curves,
            width=width,
            device=control_points.device,
        )
        return control_points[curve_indices, cp_indices, :]

    @staticmethod
    def find_spans(u: torch.Tensor, knots: torch.Tensor, degree: int, n_control_points: int) -> torch.Tensor:
        """Find the knot span index for each parameter value.

        Args:
            u: Parameter values of shape (N, M).
            knots: Knot vector of shape (K,).
            degree: B-spline degree.
            n_control_points: Number of control points per curve.

        Returns:
            Span indices of shape (N, M).

        """
        spans = torch.searchsorted(knots, u, side="right") - 1

        min_knot = knots[degree]
        max_knot = knots[n_control_points]
        spans[u <= min_knot + _BSplineFunction.ZERO_TOLERANCE] = degree
        spans[u >= max_knot - _BSplineFunction.ZERO_TOLERANCE] = n_control_points - 1
        spans.clamp_(min=degree, max=n_control_points - 1)
        return spans

    @staticmethod
    def cox_de_boor(u: torch.Tensor, knots: torch.Tensor, spans: torch.Tensor, degree: int) -> torch.Tensor:
        """Compute non-zero B-spline basis values using Cox-de Boor recursion.

        Args:
            u: Parameter values of shape (N, M).
            knots: Knot vector of shape (K,).
            spans: Knot span indices of shape (N, M).
            degree: B-spline degree.

        Returns:
            Basis values of shape (N, M, degree + 1).

        """
        num_samples, num_curves = u.shape
        basis = torch.zeros(num_samples, num_curves, degree + 1, device=u.device, dtype=u.dtype)
        left = torch.empty(num_samples, num_curves, degree + 1, device=u.device, dtype=u.dtype)
        right = torch.empty(num_samples, num_curves, degree + 1, device=u.device, dtype=u.dtype)

        basis[..., 0].fill_(1)
        zero = torch.zeros((), device=u.device, dtype=u.dtype)

        for p_iter in range(1, degree + 1):
            left_idx = spans + 1 - p_iter
            right_idx = spans + p_iter
            left[..., p_iter] = u - knots[left_idx]
            right[..., p_iter] = knots[right_idx] - u

            saved = zero
            for r_iter in range(p_iter):
                denominator = right[..., r_iter + 1] + left[..., p_iter - r_iter]
                ratios = basis[..., r_iter] / denominator
                ratios.nan_to_num_(0, 0, 0)

                basis[..., r_iter] = torch.addcmul(
                    saved,
                    right[..., r_iter + 1],
                    ratios,
                )
                saved = left[..., p_iter - r_iter] * ratios

            basis[..., p_iter] = saved

        return basis

    @staticmethod
    def evaluate_curve(
        basis: torch.Tensor,
        control_points: torch.Tensor,
        cp_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate vectorized B-spline curves.

        Args:
            basis: Basis values of shape (N, M, degree + 1).
            control_points: Control points of shape (M, C, D).
            cp_indices: Control-point indices of shape (N, M, degree + 1).

        Returns:
            Points on curves, shape (N, M, D).

        """
        gathered_control_points = _BSplineFunction._gather_control_points(control_points, cp_indices)
        return (basis.unsqueeze(-1) * gathered_control_points).sum(dim=2)

    @staticmethod
    def basis_derivative_coefficients(
        knots: torch.Tensor, spans: torch.Tensor, degree: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute coefficients used in basis-derivative recursion.

        Args:
            knots: Knot vector.
            spans: Knot span indices of shape (N, M).
            degree: B-spline degree.

        Returns:
            A tuple `(alpha, beta)` each of shape (N, M, degree + 1).

        """
        offsets = torch.arange(-degree, 1, device=spans.device).view(1, 1, -1)
        knot_indices = spans.unsqueeze(-1) + offsets

        knots_k = knots[knot_indices]
        knots_k_plus_degree = knots[knot_indices + degree]
        knots_k_plus_one = knots[knot_indices + 1]
        knots_k_plus_degree_plus_one = knots[knot_indices + degree + 1]

        alpha = degree / (knots_k_plus_degree - knots_k)
        alpha.nan_to_num_(0, 0, 0)

        beta = degree / (knots_k_plus_degree_plus_one - knots_k_plus_one)
        beta.nan_to_num_(0, 0, 0)

        return alpha, beta

    @staticmethod
    def compute_basis_derivatives(
        u: torch.Tensor, knots: torch.Tensor, spans: torch.Tensor, degree: int
    ) -> torch.Tensor:
        """Compute derivatives of non-zero B-spline basis values.

        Returns:
            A tensor of shape (N, M, degree + 1).

        """
        if degree == 0:
            return torch.zeros(*u.shape, 1, device=u.device, dtype=u.dtype)

        lower_deg_basis = _BSplineFunction.cox_de_boor(u, knots, spans, degree - 1)
        alpha, beta = _BSplineFunction.basis_derivative_coefficients(knots, spans, degree)
        lower_pad_right = F.pad(lower_deg_basis, (0, 1))
        lower_pad_left = F.pad(lower_deg_basis, (1, 0))
        return torch.addcmul(alpha * lower_pad_left, beta, lower_pad_right, value=-1)

    @staticmethod
    def _accumulate_control_point_grads(
        grad_output: torch.Tensor,
        basis: torch.Tensor,
        cp_indices: torch.Tensor,
        control_points_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        num_curves = grad_output.shape[1]
        grad_control_points = torch.zeros(
            control_points_shape,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )

        updates = grad_output.unsqueeze(2) * basis.unsqueeze(3)
        updates_flat = updates.permute(1, 0, 2, 3).reshape(num_curves, -1, grad_output.shape[-1])
        indices_flat = cp_indices.permute(1, 0, 2).reshape(num_curves, -1)
        indices_expanded = indices_flat.unsqueeze(-1).expand_as(updates_flat)
        grad_control_points.scatter_add_(1, indices_expanded, updates_flat)
        return grad_control_points

    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        control_points: torch.Tensor,
        knots: torch.Tensor,
        degree: int,
    ) -> torch.Tensor:
        need_grad_u = ctx.needs_input_grad[0]
        need_grad_cp = ctx.needs_input_grad[1]

        spans = _BSplineFunction.find_spans(u, knots, degree, control_points.shape[1])
        basis = _BSplineFunction.cox_de_boor(u, knots, spans, degree)
        cp_indices = _BSplineFunction._control_point_indices(spans, degree)
        points = _BSplineFunction.evaluate_curve(
            basis=basis,
            control_points=control_points,
            cp_indices=cp_indices,
        )

        saved = [cp_indices]
        if need_grad_u:
            saved.extend([spans, u, control_points, knots])
        if need_grad_cp:
            saved.append(basis)
        ctx.save_for_backward(*saved)

        ctx.need_grad_u = need_grad_u
        ctx.need_grad_cp = need_grad_cp
        ctx.control_points_shape = tuple(control_points.shape)
        ctx.degree = degree
        return points

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
        need_grad_u = ctx.need_grad_u
        need_grad_cp = ctx.need_grad_cp

        saved_iter = iter(ctx.saved_tensors)
        cp_indices = next(saved_iter)

        degree = ctx.degree

        grad_u = None
        if need_grad_u:
            spans = next(saved_iter)
            u = next(saved_iter)
            control_points = next(saved_iter)
            knots = next(saved_iter)
            basis_deriv = _BSplineFunction.compute_basis_derivatives(u, knots, spans, degree)
            gathered_control_points = _BSplineFunction._gather_control_points(control_points, cp_indices)
            d_points_du = torch.einsum("nmi,nmid->nmd", basis_deriv, gathered_control_points)
            grad_u = (grad_output * d_points_du).sum(dim=-1)

        grad_control_points = None
        if need_grad_cp:
            basis = next(saved_iter)
            grad_control_points = _BSplineFunction._accumulate_control_point_grads(
                grad_output=grad_output,
                basis=basis,
                cp_indices=cp_indices,
                control_points_shape=ctx.control_points_shape,
            )

        return grad_u, grad_control_points, None, None


def bspline_curves(
    u: torch.Tensor,
    control_points: torch.Tensor,
    knots: Optional[torch.Tensor] = None,
    degree: int = 3,
) -> torch.Tensor:
    r"""Evaluate multiple B-Spline curves, each with its own control points, sharing the same knots and degree.

    This function automatically handles backpropagation based on whether inputs require gradients:
    - Computes gradients only for inputs that require them using custom autograd.

    Args:
        u: A tensor of size :math:`(B, C)` of values between ``knots.min()`` and ``knots.max()``, representing
            a mini-batch of :math:`B` arguments for sampling each of the :math:`C` curves.
        control_points: A tensor of size :math:`(M, C, D)` describing :math:`M` curves with :math:`C` control
            points each, embedded in :math:`\mathbb{R}^D`.
        knots: A 1D tensor of size :math:`M + P + 1` representing the spline function's
            knot vector, where :math:`P` is the degree of the piecewise polynomials defining the spline function.
            ``None`` means uniformly spaced augmented knots in :math:`[-1, 1]` with
            repeated boundary knots. (default: ``None``)
        degree: The degree :math:`P` of the B-Spline function. (default: ``3`` meaning a cubic spline)

    Returns:
        A tensor of size :math:`(B, C, D)`, representing a mini-batch of size :math:`B`, corresponding to samples from
        :math:`C` curves in :math:`\mathbb{R}^D`.

    """
    if knots is None:
        n_control_points = control_points.shape[1]
        knots = uniform_augmented_knots(
            n_control_points,
            degree,
            dtype=control_points.dtype,
            device=control_points.device,
        )

    return cast(torch.Tensor, _BSplineFunction.apply(u, control_points, knots, degree))
