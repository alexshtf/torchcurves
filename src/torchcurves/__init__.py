"""torchcurves: Differentiable parametric curves in PyTorch."""

from .bspline import BSplineCurve, BSplineEmbeddings, BSplineFunction
from .kan_tools import Replicate, Sum
from .legendre import LegendreCurve, LegendreCurveFunction

__version__ = "0.1.0"
__all__ = [
    "BSplineCurve",
    "BSplineEmbeddings",
    "BSplineFunction",
    "LegendreCurve",
    "LegendreCurveFunction",
    "Replicate",
    "Sum",
]
