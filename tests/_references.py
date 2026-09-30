from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline


def numpy64(tensor: torch.Tensor) -> np.ndarray:
    """Return the represented tensor values as detached CPU FP64 data."""
    return tensor.detach().to(device="cpu", dtype=torch.float64).numpy()


def assert_reference(actual, expected, precision, *, device="cpu", scale=None):
    """Check the tensor contract before comparing with an independent FP64 reference."""
    dtype, rtol, atol = precision
    assert actual.dtype == dtype and actual.device == torch.device(device)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    if scale is None:
        np.testing.assert_allclose(numpy64(actual), expected, rtol=rtol, atol=atol)
    else:
        # Cancellation in a VJP needs the sum of absolute terms as its error scale.
        np.testing.assert_array_less(np.abs(numpy64(actual) - expected), atol + rtol * scale)


def legendre_reference(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    upstream: torch.Tensor | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """NumPy values and optional (VJP, absolute contraction scale) for each input."""
    args, coeffs = numpy64(x), numpy64(coefficients)

    def evaluate(c):
        return np.stack(
            [np.polynomial.legendre.legval(args[:, curve], c[:, curve, :]).T for curve in range(args.shape[1])],
            axis=1,
        )

    values = evaluate(coeffs)
    if upstream is None:
        return values, []
    weights = numpy64(upstream)
    basis = np.polynomial.legendre.legvander(args, coeffs.shape[0] - 1)
    derivative = evaluate(np.polynomial.legendre.legder(coeffs, axis=0))
    terms = weights * derivative
    grad_x, scale_x = terms.sum(axis=-1), np.abs(terms).sum(axis=-1)
    grad_coeffs = np.einsum("bcn,bcd->ncd", basis, weights)
    scale_coeffs = np.einsum("bcn,bcd->ncd", np.abs(basis), np.abs(weights))
    return values, [(grad_x, scale_x), (grad_coeffs, scale_coeffs)]


def bspline_reference(
    u: torch.Tensor,
    coefficients: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    upstream: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Evaluate SciPy values and analytical vector-Jacobian products in FP64."""
    args, coeffs, knot_values = numpy64(u), numpy64(coefficients), numpy64(knots)
    spline = BSpline(knot_values, np.eye(coeffs.shape[1]), degree, extrapolate=False)
    basis = spline(args)  # (samples, curves, control points)
    values = np.einsum("bmc,mcd->bmd", basis, coeffs)
    if upstream is None:
        return values, None, None
    weights = numpy64(upstream)
    derivative = np.zeros_like(basis) if degree == 0 else spline(args, nu=1)
    grad_u = np.einsum("bmc,mcd,bmd->bm", derivative, coeffs, weights)
    grad_coefficients = np.einsum("bmc,bmd->mcd", basis, weights)
    return values, grad_u, grad_coefficients
