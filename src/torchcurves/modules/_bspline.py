from typing import Literal, Union

import torch
import torch.nn as nn

from ..functional import bspline_curves, bspline_embeddings, uniform_augmented_knots
from ..types import NormalizationFn
from ._normalization import _normalization_catalogue


class BSplineCurveBase(nn.Module):
    """Base PyTorch module for B-spline curves, supporting a batch of multiple curves."""

    knots: torch.Tensor  # explicit annotation for type-checking

    def __init__(
        self,
        num_curves: int,
        dim: int,
        degree: int = 3,
        knots_config: Union[int, torch.Tensor] = 10,  # This is n_control_points_per_curve
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

        self.num_curves = num_curves  # m
        self.dim = dim  # d
        self.degree = degree  # p

        if isinstance(normalize_fn, str):
            normalize_fn_callable = _normalization_catalogue.get(normalize_fn)
            if normalize_fn_callable is None:
                raise ValueError(f"Unknown normalization {normalize_fn}")
            self.normalize_fn = normalize_fn_callable
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
        self._knot_min = knot_buffer[self.degree].item()
        self._knot_max = knot_buffer[self.n_control_points_per_curve].item()

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
            u: Parameter values of size :math:`(B, C)`, where :math:`B` is the mini-batch size, and `C` is the number
                of curves, and must be equal to `self.num_curves`.

        Returns:
            Points on the B-spline curves of shape :math:`(B, C, D)`.

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


class BSplineEmbeddings(BSplineCurveBase):
    r"""Embeddings layer based on B-Spline curves.

    Useful as the first layer in a neural network, where the input comes from a data-set.

    The learnable parameters are the control points of :math:`M` curves in :math:`\mathbb{R}^D`.
    All curves share the same degree and knot configuration.

    The input of this layer normalized to the range :math:`[-1, 1]` (or the range of the knots if specified differently)
    using the specified normalization strategy.

    Args:
        num_curves: Number of B-spline curves to define in this module (:math:`M`).
        dim: Dimension of each curve's output points (:math:`D`).
        degree: Degree of the B-spline (default: 3).
        knots_config:
            If an int, it specifies the number of control points per curve (:math:`C`).
            A uniformly-spaced knot vector will be automatically generated in [-1, 1].
            If a torch.Tensor, it explicitly specifies the knot values. The number
            of control points will be inferred. The tensor should be 1D.
        normalize_fn: Normalization method layer's input. (default: "rational")
        normalization_scale: Scale factor for normalization (default: 1.0).

    Note:
        Assumes the input of this layer is not learnable, and thus doesn't require computing gradients.

    """

    def _forward_core(self, u_prepared: torch.Tensor) -> torch.Tensor:
        return bspline_embeddings(u_prepared, self.control_points, self.knots, self.degree)


class BSplineCurve(BSplineCurveBase):
    r"""B-Spline curves layer that allows back-propagating through its input.

    The learnable parameters are the control points of :math:`M` curves in :math:`\mathbb{R}^D`.
    All curves share the same degree and knot configuration.

    The input of this layer normalized to the range :math:`[-1, 1]` (or the range of the knots if specified differently)
    using the specified normalization strategy.

    Args:
        num_curves: Number of B-spline curves to define in this module (:math:`M`).
        dim: Dimension of each curve's output points (:math:`D`).
        degree: Degree of the B-spline (default: 3).
        knots_config:
            If an int, it specifies the number of control points per curve (:math:`C`).
            A uniformly-spaced knot vector will be automatically generated in [-1, 1].
            If a torch.Tensor, it explicitly specifies the knot values. The number
            of control points will be inferred. The tensor should be 1D.
        normalize_fn: Normalization method layer's input. (default: "rational")
        normalization_scale: Scale factor for normalization (default: 1.0).

    """

    def _forward_core(self, u_prepared: torch.Tensor) -> torch.Tensor:
        return bspline_curves(u_prepared, self.control_points, self.knots, self.degree)
