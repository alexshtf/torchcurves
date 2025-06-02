"""torchcurves: Differentiable parametric curves in PyTorch."""

from .bspline import BSplineCurve, BSplineEmbeddings, BSplineFunction, bspline_curves, uniform_augmented_knots
from .kan_tools import Sum
from .legendre import LegendreCurve, LegendreCurveFunction

__version__ = "0.1.0"
__all__ = [
    "uniform_augmented_knots",
    "bspline_curves",
    "BSplineCurve",
    "BSplineEmbeddings",
    "BSplineFunction",
    "LegendreCurve",
    "LegendreCurveFunction",
    "Sum",
]
