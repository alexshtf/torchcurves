from typing import Literal, Union

import torch
import torch.nn as nn

from ..functional import bspline_curves, uniform_augmented_knots
from ..types import NormalizationFn
from ._normalization import _normalization_catalogue


class BSplineBasis(nn.Module):
    r"""PyTorch module for a fixed B-spline basis.

    This module stores the knot vector, spline degree, and input normalization,
    but does not own coefficients. Callers provide coefficients at evaluation
    time, which makes this module useful when another network predicts them.

    Args:
        degree: Degree of the B-spline (default: 3).
        knots_config:
            If an int, it specifies the number of control points per curve.
            A uniformly spaced augmented knot vector will be generated
            automatically.
            If a torch.Tensor, it explicitly specifies the knot values. The
            number of control points will be inferred. The tensor must be 1D.
        normalize_fn: Normalization method for the module input. (default: "rational")
        normalization_scale: Scale factor for normalization (default: 1.0).

    """

    knots: torch.Tensor

    def __init__(
        self,
        degree: int = 3,
        knots_config: Union[int, torch.Tensor] = 10,
        normalize_fn: Union[Literal["clamp", "rational", "arctan"], NormalizationFn] = "rational",
        normalization_scale: float = 1.0,
    ):
        super().__init__()

        if not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer.")

        self.degree = degree

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
            n_control_points_per_curve = knots_config
            if n_control_points_per_curve <= self.degree:
                raise ValueError(
                    f"Number of control points ({n_control_points_per_curve}) must be greater "
                    f"than the degree ({self.degree})."
                )
            knot_buffer = uniform_augmented_knots(
                n_control_points_per_curve,
                self.degree,
            )
        elif isinstance(knots_config, torch.Tensor):
            if knots_config.ndim != 1:
                raise ValueError("Provided knots_config tensor must be 1D.")
            num_knots = knots_config.shape[0]
            n_control_points_per_curve = num_knots - self.degree - 1
            if n_control_points_per_curve <= self.degree:
                raise ValueError(
                    f"Number of control points ({n_control_points_per_curve}) must be greater "
                    f"than the degree ({self.degree})."
                )
            knot_buffer = knots_config.detach().clone()
        else:
            raise TypeError("knots_config must be an int (number of control points) or a torch.Tensor (knot vector).")

        self.n_control_points_per_curve = n_control_points_per_curve
        self.register_buffer("knots", knot_buffer)

        # Effective parameter range for a degree-p spline with C control points is
        # [knots[p], knots[C]], assuming a sorted knot vector.
        self._knot_min = self.knots[self.degree].item()
        self._knot_max = self.knots[self.n_control_points_per_curve].item()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"degree={self.degree}, "
            f"n_control_points_per_curve={self.n_control_points_per_curve}, "
            f"knots_shape={self.knots.shape if hasattr(self, 'knots') else None})"
        )

    def _prepare_arg(self, u: torch.Tensor) -> torch.Tensor:
        return self.normalize_fn(
            u,
            self.normalization_scale,
            out_min=self._knot_min,
            out_max=self._knot_max,
        )

    def forward(self, u: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """Evaluate caller-supplied coefficients in this B-spline basis.

        Args:
            u: Parameter values of shape :math:`(B, M)`, where :math:`B` is the
                mini-batch size and :math:`M` is the number of curves.
            coefficients: Coefficients of shape :math:`(M, C, D)`, where :math:`C`
                is the number of control points and :math:`D` is the output
                dimension.

        Returns:
            Points on the B-spline curves of shape :math:`(B, M, D)`.

        """
        if u.ndim != 2:
            raise ValueError(f"Input u must be a 2D tensor of shape (batch_size, num_curves). Got shape: {u.shape}")
        if coefficients.ndim != 3:
            raise ValueError(
                "Input coefficients must be a 3D tensor of shape "
                f"(num_curves, n_control_points_per_curve={self.n_control_points_per_curve}, dim). "
                f"Got shape: {coefficients.shape}"
            )
        if u.shape[1] != coefficients.shape[0]:
            raise ValueError(
                "The number of curves must match between u and coefficients. "
                f"Got u.shape[1]={u.shape[1]} and coefficients.shape[0]={coefficients.shape[0]}."
            )
        if coefficients.shape[1] != self.n_control_points_per_curve:
            raise ValueError(
                "The number of control points in coefficients must match this basis. "
                f"Expected {self.n_control_points_per_curve}, got {coefficients.shape[1]}."
            )

        u_prepared = self._prepare_arg(u)
        return bspline_curves(u_prepared, coefficients, self.knots, self.degree)


class BSplineCurve(nn.Module):
    r"""PyTorch module for B-spline curves with learnable control points, supporting a batch of multiple curves.

    The learnable parameters are the control points of :math:`M` curves in :math:`\mathbb{R}^D`.
    All curves share the same degree and knot configuration. Internally, this
    module stores the shared spline basis in a :class:`BSplineBasis` module.

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

    def __init__(
        self,
        num_curves: int,
        dim: int,
        degree: int = 3,
        knots_config: Union[int, torch.Tensor] = 10,  # This is n_control_points_per_curve
        normalize_fn: Union[Literal["clamp", "rational", "arctan"], NormalizationFn] = "rational",
        normalization_scale: float = 1.0,
    ):
        super().__init__()

        if not isinstance(num_curves, int) or num_curves <= 0:
            raise ValueError("num_curves must be a positive integer.")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer.")

        self.num_curves = num_curves  # m
        self.dim = dim  # d
        self.basis = BSplineBasis(
            degree=degree,
            knots_config=knots_config,
            normalize_fn=normalize_fn,
            normalization_scale=normalization_scale,
        )

        # Control points shape: (m, c, d)
        self.control_points = nn.Parameter(
            torch.empty(self.num_curves, self.basis.n_control_points_per_curve, self.dim)
        )
        nn.init.xavier_uniform_(self.control_points)
        self.basis = self.basis.to(
            dtype=self.control_points.dtype,
            device=self.control_points.device,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_curves={self.num_curves}, "
            f"n_control_points_per_curve={self.n_control_points_per_curve}, "
            f"dim={self.dim}, degree={self.degree}, "
            f"knots_shape={self.knots.shape if hasattr(self, 'knots') else None})"
        )

    @property
    def degree(self) -> int:
        return self.basis.degree

    @property
    def knots(self) -> torch.Tensor:
        return self.basis.knots

    @property
    def normalize_fn(self) -> NormalizationFn:
        return self.basis.normalize_fn

    @property
    def normalization_scale(self) -> float:
        return self.basis.normalization_scale

    @property
    def n_control_points_per_curve(self) -> int:
        return self.basis.n_control_points_per_curve

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

        return self.basis(u, self.control_points)
