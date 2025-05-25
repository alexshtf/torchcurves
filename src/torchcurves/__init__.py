"""torchcurves: Differentiable parametric curves in PyTorch."""

from .bspline import BSplineCurve, BSplineFunction
from .legendre import LegendreCurve, LegendreCurveFunction, LegendreRationalCurve

__version__ = "0.1.0"
__all__ = ["BSplineCurve", "BSplineFunction", "LegendreCurve", "LegendreCurveFunction", "LegendreRationalCurve"]
