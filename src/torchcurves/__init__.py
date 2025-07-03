"""torchcurves: Differentiable parametric curves in PyTorch."""

from ._bspline import BSplineCurve, BSplineEmbeddings, _BSplineFunction, bspline_curves, uniform_augmented_knots
from ._kan_tools import Sum
from ._legendre import LegendreCurve, legendre_curves

__version__ = "0.1.0"
__all__ = [
    "BSplineCurve",
    "BSplineEmbeddings",
    "_BSplineFunction",
    "uniform_augmented_knots",
    "bspline_curves",
    "LegendreCurve",
    "legendre_curves",
    "Sum",
]
