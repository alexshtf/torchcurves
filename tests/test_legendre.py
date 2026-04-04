import re

import numpy as np
import pytest
import torch

from torchcurves import LegendreCurve
from torchcurves.functional import legendre_curves

DTYPE = torch.float64


def _seeded_coefficients(num_curves: int, degree: int, dim: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(degree + 1, num_curves, dim, dtype=DTYPE, generator=generator)


def _torch_single_curve_single_sample(x_value: float, coefficients_single: torch.Tensor) -> torch.Tensor:
    x_single = torch.tensor([[x_value]], dtype=DTYPE)  # (batch=1, curves=1)
    return legendre_curves(x_single, coefficients_single)  # (1, 1, dim)


def _numpy_single_curve_single_sample(x_value: float, coefficients_single: torch.Tensor) -> torch.Tensor:
    coeff_np = coefficients_single[:, 0, :].detach().cpu().numpy()  # (degree+1, dim)
    values = [
        float(np.polynomial.legendre.legval(x_value, coeff_np[:, dim_index])) for dim_index in range(coeff_np.shape[1])
    ]
    return torch.tensor(values, dtype=coefficients_single.dtype).view(1, 1, -1)


@pytest.mark.parametrize(
    ("degree", "dim", "x_value"),
    [
        (0, 1, -0.8),
        (2, 2, -0.3),
        (4, 3, 0.4),
        (7, 1, 0.95),
    ],
)
def test_single_curve_single_sample_matches_numpy(degree: int, dim: int, x_value: float) -> None:
    coefficients_single = _seeded_coefficients(1, degree, dim)
    actual = _torch_single_curve_single_sample(x_value, coefficients_single)
    expected = _numpy_single_curve_single_sample(x_value, coefficients_single)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_multiple_curves_one_sample_is_concat_over_curve_dimension() -> None:
    degree = 5
    dim = 2
    num_curves = 3

    coefficients = _seeded_coefficients(num_curves, degree, dim)
    x = torch.tensor([[-0.7, 0.1, 0.9]], dtype=DTYPE)  # (batch=1, curves=3)

    batched = legendre_curves(x, coefficients)  # (1, 3, dim)

    per_curve = []
    for curve_index in range(num_curves):
        x_single = x[:, curve_index : curve_index + 1]  # (1, 1)
        coeff_single = coefficients[:, curve_index : curve_index + 1, :]  # (degree+1, 1, dim)
        per_curve.append(legendre_curves(x_single, coeff_single))  # each (1, 1, dim)

    expected = torch.cat(per_curve, dim=1)  # concat curves -> (1, 3, dim)
    torch.testing.assert_close(batched, expected, rtol=1e-12, atol=1e-12)


def test_one_curve_multiple_samples_is_concat_over_batch_dimension() -> None:
    degree = 6
    dim = 2

    coefficients_single = _seeded_coefficients(1, degree, dim)
    x = torch.tensor([[-0.9], [-0.4], [0.2], [0.85]], dtype=DTYPE)  # (batch=4, curves=1)

    batched = legendre_curves(x, coefficients_single)  # (4, 1, dim)

    per_sample = []
    for sample_index in range(x.shape[0]):
        x_single = x[sample_index : sample_index + 1, :]  # (1, 1)
        per_sample.append(legendre_curves(x_single, coefficients_single))  # each (1, 1, dim)

    expected = torch.cat(per_sample, dim=0)  # concat batch -> (4, 1, dim)
    torch.testing.assert_close(batched, expected, rtol=1e-12, atol=1e-12)


def test_multiple_curves_multiple_samples_matches_nested_single_single_concatenation() -> None:
    degree = 4
    dim = 3
    num_curves = 2

    coefficients = _seeded_coefficients(num_curves, degree, dim)
    x = torch.tensor(
        [
            [-0.8, 0.3],
            [-0.1, 0.7],
            [0.5, 0.95],
        ],
        dtype=DTYPE,
    )  # (batch=3, curves=2)

    batched = legendre_curves(x, coefficients)  # (3, 2, dim)

    rows = []
    for sample_index in range(x.shape[0]):
        row_curves = []
        for curve_index in range(num_curves):
            x_single = x[sample_index : sample_index + 1, curve_index : curve_index + 1]  # (1, 1)
            coeff_single = coefficients[:, curve_index : curve_index + 1, :]  # (degree+1, 1, dim)
            row_curves.append(legendre_curves(x_single, coeff_single))  # (1, 1, dim)
        rows.append(torch.cat(row_curves, dim=1))  # (1, curves, dim)

    expected = torch.cat(rows, dim=0)  # (batch, curves, dim)
    torch.testing.assert_close(batched, expected, rtol=1e-12, atol=1e-12)


def test_legendre_module_accepts_batched_curve_inputs() -> None:
    model = LegendreCurve(num_curves=3, dim=2, degree=5).double()
    u = torch.tensor(
        [
            [-0.9, -0.1, 0.2],
            [0.0, 0.3, 0.7],
            [0.4, -0.8, 1.0],
            [0.9, 0.1, -0.4],
        ],
        dtype=DTYPE,
    )

    actual = model(u)

    assert actual.shape == (4, 3, 2)


@pytest.mark.parametrize(
    "shape",
    [(4,), (4, 3, 1), (4, 2)],
    ids=["rank-1", "rank-3", "wrong-num-curves"],
)
def test_legendre_module_rejects_invalid_input_shapes(shape: tuple[int, ...]) -> None:
    model = LegendreCurve(num_curves=3, dim=2, degree=5)
    u = torch.rand(shape)
    expected_message = f"Input u must be a 2D tensor of shape (N, num_curves={model.num_curves}). Got shape: {u.shape}"

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        model(u)
