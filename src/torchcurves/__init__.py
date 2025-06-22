"""torchcurves: Differentiable parametric curves in PyTorch."""

from .bspline import BSplineCurve, BSplineEmbeddings, BSplineFunction, bspline_curves, uniform_augmented_knots
from .kan_tools import Sum
from .legendre import LegendreCurve, legendre_curves

__version__ = "0.1.0"
__all__ = [
    "BSplineCurve",
    "BSplineEmbeddings",
    "BSplineFunction",
    "uniform_augmented_knots",
    "bspline_curves",
    "LegendreCurve",
    "legendre_curves",
    "Sum",
]
