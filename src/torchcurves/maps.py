"""Input-map factories for curve modules.

The target interval is chosen by the curve family:

- :class:`torchcurves.LegendreCurve` always maps to ``[-1, 1]``.
- :class:`torchcurves.BSplineBasis` and :class:`torchcurves.BSplineCurve`
  map to their effective knot interval.

Built-in dotted presets:

- ``"real.rational"``
- ``"real.arctan"``
- ``"real.clamp"``
- ``"nonneg.rational"``
- ``"nonneg.arctan"``

Example usage includes ``tc.BSplineCurve(4, 8, input_map="real.clamp")``,
``tc.LegendreCurve(4, 8, degree=5, input_map=tc.maps.Real.rational(scale=2.0))``,
and ``tc.BSplineBasis(knots_config=8, parameter_range=(0.0, 1.0), input_map="nonneg.rational")``.
"""

from dataclasses import dataclass
from typing import Callable, Union

import torch

from .functional import arctan as _arctan
from .functional import clamp as _clamp
from .functional import rational as _rational
from .types import InputMap, TensorLike


def _validate_scale(scale: float) -> float:
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"Input map scale must be positive, but {scale} was given.")
    return scale


def _scale_from_unit_interval(x: torch.Tensor, out_min: float, out_max: float) -> torch.Tensor:
    return x * (out_max - out_min) + out_min


def _nonnegative_part(x: TensorLike) -> torch.Tensor:
    return torch.clamp_min(torch.as_tensor(x), 0)


def _map_nonnegative_with_symmetric_helper(
    x: TensorLike,
    map_fn: Callable[..., torch.Tensor],
    scale: float,
    out_min: float,
    out_max: float,
) -> torch.Tensor:
    x_nonnegative = _nonnegative_part(x)
    mapped = map_fn(x_nonnegative, scale=scale, out_min=-1, out_max=1)
    return _scale_from_unit_interval(mapped, out_min=out_min, out_max=out_max)


@dataclass(frozen=True)
class _RealRationalMap:
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _validate_scale(self.scale))

    def __call__(self, x: TensorLike, out_min: float, out_max: float) -> torch.Tensor:
        return _rational(x, scale=self.scale, out_min=out_min, out_max=out_max)


@dataclass(frozen=True)
class _RealArctanMap:
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _validate_scale(self.scale))

    def __call__(self, x: TensorLike, out_min: float, out_max: float) -> torch.Tensor:
        return _arctan(x, scale=self.scale, out_min=out_min, out_max=out_max)


@dataclass(frozen=True)
class _RealClampMap:
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _validate_scale(self.scale))

    def __call__(self, x: TensorLike, out_min: float, out_max: float) -> torch.Tensor:
        return _clamp(x, scale=self.scale, out_min=out_min, out_max=out_max)


@dataclass(frozen=True)
class _NonnegRationalMap:
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _validate_scale(self.scale))

    def __call__(self, x: TensorLike, out_min: float, out_max: float) -> torch.Tensor:
        return _map_nonnegative_with_symmetric_helper(
            x,
            _rational,
            scale=self.scale,
            out_min=out_min,
            out_max=out_max,
        )


@dataclass(frozen=True)
class _NonnegArctanMap:
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _validate_scale(self.scale))

    def __call__(self, x: TensorLike, out_min: float, out_max: float) -> torch.Tensor:
        return _map_nonnegative_with_symmetric_helper(
            x,
            _arctan,
            scale=self.scale,
            out_min=out_min,
            out_max=out_max,
        )


class _RealNamespace:
    """Factories for maps from the real line to a curve parameter interval."""

    def rational(self, scale: float = 1.0) -> InputMap:
        return _RealRationalMap(scale=scale)

    def arctan(self, scale: float = 1.0) -> InputMap:
        return _RealArctanMap(scale=scale)

    def clamp(self, scale: float = 1.0) -> InputMap:
        return _RealClampMap(scale=scale)


class _NonnegNamespace:
    """Factories for maps from the nonnegative reals to a curve parameter interval."""

    def rational(self, scale: float = 1.0) -> InputMap:
        return _NonnegRationalMap(scale=scale)

    def arctan(self, scale: float = 1.0) -> InputMap:
        return _NonnegArctanMap(scale=scale)


Real = _RealNamespace()
Nonneg = _NonnegNamespace()

_input_map_catalogue: dict[str, Callable[[], InputMap]] = {
    "real.rational": Real.rational,
    "real.arctan": Real.arctan,
    "real.clamp": Real.clamp,
    "nonneg.rational": Nonneg.rational,
    "nonneg.arctan": Nonneg.arctan,
}


def resolve_input_map(input_map: Union[InputMap, str]) -> InputMap:
    """Resolve a dotted preset string or validate a callable input map."""
    if isinstance(input_map, str):
        input_map_factory = _input_map_catalogue.get(input_map)
        if input_map_factory is None:
            raise ValueError(f"Unknown input_map {input_map}")
        return input_map_factory()

    if not callable(input_map):
        raise TypeError("input_map must be a dotted preset string or a callable.")

    return input_map


__all__ = ["Nonneg", "Real"]
