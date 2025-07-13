"""Differentiable parametric curves in arbitrary dimensions."""

from ._bspline import BSplineCurve, BSplineEmbeddings, bspline_curves, uniform_augmented_knots
from ._kan_tools import Sum
from ._legendre import LegendreCurve, legendre_curves

__version__ = "0.1.0"
__all__ = [
    "BSplineCurve",
    "BSplineEmbeddings",
    "LegendreCurve",
    "Sum",
    "legendre_curves",
    "uniform_augmented_knots",
    "bspline_curves",
]
