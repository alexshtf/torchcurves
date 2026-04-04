import re

import pytest
import torch
from scipy.interpolate import BSpline as SciPyBSpline

from torchcurves import BSplineCurve
from torchcurves.functional import bspline_curves, uniform_augmented_knots

DTYPE = torch.float64


def _seeded_control_points(num_curves: int, n_control_points: int, dim: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(num_curves, n_control_points, dim, dtype=DTYPE, generator=generator)


def _torch_single_curve_single_sample(
    u_value: float,
    control_points_single: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    u_single = torch.tensor([[u_value]], dtype=DTYPE)  # (batch=1, curves=1)
    return bspline_curves(u_single, control_points_single, knots, degree)  # (1, 1, dim)


def _scipy_single_curve_single_sample(
    u_value: float,
    control_points_single: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    cp_np = control_points_single[0].detach().cpu().numpy()  # (n_control_points, dim)
    knots_np = knots.detach().cpu().numpy()
    values = [
        float(
            SciPyBSpline(
                knots_np,
                cp_np[:, dim_index],
                degree,
                extrapolate=False,
            )(u_value)
        )
        for dim_index in range(cp_np.shape[1])
    ]
    return torch.tensor(values, dtype=control_points_single.dtype).view(1, 1, -1)


@pytest.mark.parametrize(
    ("degree", "n_control_points", "dim", "u_value"),
    [
        (1, 4, 1, -1.0),
        (2, 5, 2, -0.25),
        (3, 6, 3, 0.3),
        (3, 6, 2, 1.0),
    ],
)
def test_single_curve_single_sample_matches_scipy(degree: int, n_control_points: int, dim: int, u_value: float) -> None:
    control_points_single = _seeded_control_points(1, n_control_points, dim)
    knots = uniform_augmented_knots(n_control_points, degree, dtype=DTYPE)

    actual = _torch_single_curve_single_sample(u_value, control_points_single, knots, degree)
    expected = _scipy_single_curve_single_sample(u_value, control_points_single, knots, degree)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_multiple_curves_one_sample_is_concat_over_curve_dimension() -> None:
    degree = 3
    n_control_points = 7
    dim = 2
    num_curves = 3

    control_points = _seeded_control_points(num_curves, n_control_points, dim)
    knots = uniform_augmented_knots(n_control_points, degree, dtype=DTYPE)
    u = torch.tensor([[-0.8, -0.1, 0.6]], dtype=DTYPE)  # (batch=1, curves=3)

    batched = bspline_curves(u, control_points, knots, degree)  # (1, 3, dim)

    per_curve = []
    for curve_index in range(num_curves):
        u_single = u[:, curve_index : curve_index + 1]  # (1, 1)
        cp_single = control_points[curve_index : curve_index + 1, :, :]  # (1, n_control_points, dim)
        per_curve.append(bspline_curves(u_single, cp_single, knots, degree))  # each (1, 1, dim)

    expected = torch.cat(per_curve, dim=1)  # concat curves -> (1, 3, dim)
    torch.testing.assert_close(batched, expected, rtol=1e-12, atol=1e-12)


def test_one_curve_multiple_samples_is_concat_over_batch_dimension() -> None:
    degree = 2
    n_control_points = 6
    dim = 2

    control_points_single = _seeded_control_points(1, n_control_points, dim)
    knots = uniform_augmented_knots(n_control_points, degree, dtype=DTYPE)
    u = torch.tensor([[-0.9], [-0.2], [0.1], [0.8]], dtype=DTYPE)  # (batch=4, curves=1)

    batched = bspline_curves(u, control_points_single, knots, degree)  # (4, 1, dim)

    per_sample = []
    for sample_index in range(u.shape[0]):
        u_single = u[sample_index : sample_index + 1, :]  # (1, 1)
        per_sample.append(bspline_curves(u_single, control_points_single, knots, degree))  # each (1, 1, dim)

    expected = torch.cat(per_sample, dim=0)  # concat batch -> (4, 1, dim)
    torch.testing.assert_close(batched, expected, rtol=1e-12, atol=1e-12)


def test_multiple_curves_multiple_samples_matches_nested_single_single_concatenation() -> None:
    degree = 3
    n_control_points = 8
    dim = 3
    num_curves = 2

    control_points = _seeded_control_points(num_curves, n_control_points, dim)
    knots = uniform_augmented_knots(n_control_points, degree, dtype=DTYPE)
    u = torch.tensor(
        [
            [-0.9, -0.1],
            [0.2, 0.7],
            [0.5, 1.0],
        ],
        dtype=DTYPE,
    )  # (batch=3, curves=2)

    batched = bspline_curves(u, control_points, knots, degree)  # (3, 2, dim)

    rows = []
    for sample_index in range(u.shape[0]):
        row_curves = []
        for curve_index in range(num_curves):
            u_single = u[sample_index : sample_index + 1, curve_index : curve_index + 1]  # (1, 1)
            cp_single = control_points[curve_index : curve_index + 1, :, :]  # (1, n_control_points, dim)
            row_curves.append(bspline_curves(u_single, cp_single, knots, degree))  # (1, 1, dim)
        rows.append(torch.cat(row_curves, dim=1))  # (1, curves, dim)

    expected = torch.cat(rows, dim=0)  # (batch, curves, dim)
    torch.testing.assert_close(batched, expected, rtol=1e-12, atol=1e-12)


def test_bspline_module_accepts_batched_curve_inputs() -> None:
    model = BSplineCurve(num_curves=3, dim=2, degree=3, knots_config=7).double()
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
def test_bspline_module_rejects_invalid_input_shapes(shape: tuple[int, ...]) -> None:
    model = BSplineCurve(num_curves=3, dim=2)
    u = torch.rand(shape)
    expected_message = f"Input u must be a 2D tensor of shape (N, num_curves={model.num_curves}). Got shape: {u.shape}"

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        model(u)
