import re

import pytest
import torch
from scipy.interpolate import BSpline as SciPyBSpline
from torch.autograd import gradcheck

import torchcurves as tc
from torchcurves import BSplineBasis, BSplineCurve
from torchcurves.functional import arctan, bspline_curves, uniform_augmented_knots

DTYPE = torch.float64
GRADCHECK_EPS = 1e-6
GRADCHECK_ATOL = 1e-4
GRADCHECK_RTOL = 1e-3


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


def _run_bspline_gradcheck(
    u: torch.Tensor,
    *,
    degree: int,
    n_control_points: int,
    dim: int,
) -> None:
    num_curves = u.shape[1]
    control_points = _seeded_control_points(num_curves, n_control_points, dim).requires_grad_()
    knots = uniform_augmented_knots(n_control_points, degree, dtype=DTYPE)

    # Use float64 CPU tensors with explicit tolerances to keep finite-difference
    # checks stable across representative spline degrees and boundary-adjacent inputs.
    passed = gradcheck(
        lambda u_arg, control_points_arg: bspline_curves(u_arg, control_points_arg, knots, degree),
        (u.requires_grad_(), control_points),
        eps=GRADCHECK_EPS,
        atol=GRADCHECK_ATOL,
        rtol=GRADCHECK_RTOL,
    )

    assert passed


def _tanh_map(x: torch.Tensor, out_min: float, out_max: float) -> torch.Tensor:
    mapped = torch.tanh(x)
    return 0.5 * (mapped + 1.0) * (out_max - out_min) + out_min


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


@pytest.mark.parametrize(
    ("degree", "n_control_points", "dim", "u"),
    [
        (
            1,
            4,
            1,
            torch.tensor([[-0.75], [0.15]], dtype=DTYPE),
        ),
        (
            1,
            4,
            4,
            torch.tensor([[-0.75], [0.15]], dtype=DTYPE),
        ),
        (
            2,
            5,
            2,
            torch.tensor(
                [
                    [-0.6, 0.1],
                    [0.35, 0.8],
                ],
                dtype=DTYPE,
            ),
        ),
    ],
    ids=["degree-1-single-curve", "degree-1-high-dim", "degree-2-two-curves"],
)
def test_bspline_gradcheck_interior_inputs(
    degree: int,
    n_control_points: int,
    dim: int,
    u: torch.Tensor,
) -> None:
    _run_bspline_gradcheck(
        u,
        degree=degree,
        n_control_points=n_control_points,
        dim=dim,
    )


def test_bspline_gradcheck_near_knot_boundaries() -> None:
    degree = 3
    n_control_points = 6
    dim = 1
    knots = uniform_augmented_knots(n_control_points, degree, dtype=DTYPE)
    left = knots[degree].item()
    right = knots[n_control_points].item()
    offset = 1e-4
    u = torch.tensor(
        [
            [left + offset, right - offset],
            [left + 2 * offset, 0.0],
        ],
        dtype=DTYPE,
    )

    _run_bspline_gradcheck(
        u,
        degree=degree,
        n_control_points=n_control_points,
        dim=dim,
    )


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


def test_bspline_module_accepts_explicit_knot_tensor() -> None:
    knots = uniform_augmented_knots(7, 3, dtype=DTYPE)
    model = BSplineCurve(num_curves=2, dim=3, degree=3, knots_config=knots).double()
    u = torch.tensor(
        [
            [-0.9, 0.2],
            [0.1, 0.8],
        ],
        dtype=DTYPE,
    )

    actual = model(u)

    assert actual.shape == (2, 2, 3)
    assert model.n_control_points_per_curve == 7
    torch.testing.assert_close(model.knots, knots)


def test_bspline_curve_uses_basis_for_forward() -> None:
    model = BSplineCurve(num_curves=3, dim=2, degree=3, knots_config=7).double()
    u = torch.tensor(
        [
            [-0.7, -0.1, 0.2],
            [0.0, 0.5, 0.9],
        ],
        dtype=DTYPE,
    )

    actual = model(u)
    expected = model.basis(u, model.control_points)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_bspline_curve_exposes_basis_aliases() -> None:
    input_map = tc.maps.Real.clamp(scale=0.25)
    model = BSplineCurve(
        num_curves=2,
        dim=3,
        degree=2,
        knots_config=6,
        parameter_range=(0.0, 2.0),
        input_map=input_map,
    )

    assert isinstance(model.basis, BSplineBasis)
    assert model.degree == model.basis.degree == 2
    assert model.knots is model.basis.knots
    assert model.parameter_range == model.basis.parameter_range == (0.0, 2.0)
    assert model.input_map is model.basis.input_map is input_map
    assert model.n_control_points_per_curve == model.basis.n_control_points_per_curve == 6


def test_bspline_basis_with_uniform_knots_matches_manual_functional_path() -> None:
    degree = 3
    n_control_points = 7
    dim = 2
    num_curves = 3

    basis = BSplineBasis(
        degree=degree,
        knots_config=n_control_points,
        input_map="real.clamp",
    ).double()
    coefficients = _seeded_control_points(num_curves, n_control_points, dim)
    u = torch.tensor(
        [
            [-0.8, -0.1, 0.6],
            [0.1, 0.2, 0.9],
        ],
        dtype=DTYPE,
    )

    actual = basis(u, coefficients)
    expected = bspline_curves(u, coefficients, basis.knots, degree)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_bspline_basis_with_parameter_range_generates_matching_uniform_knots() -> None:
    degree = 2
    n_control_points = 6
    parameter_range = (0.0, 1.0)

    basis = BSplineBasis(
        degree=degree,
        knots_config=n_control_points,
        parameter_range=parameter_range,
        input_map="real.clamp",
    ).double()

    expected_knots = uniform_augmented_knots(
        n_control_points,
        degree,
        dtype=DTYPE,
        k_min=parameter_range[0],
        k_max=parameter_range[1],
    )

    torch.testing.assert_close(basis.knots, expected_knots, rtol=1e-12, atol=1e-12)
    assert basis.parameter_range == parameter_range


def test_bspline_basis_with_explicit_knots_matches_manual_functional_path() -> None:
    degree = 2
    knots = uniform_augmented_knots(6, degree, dtype=DTYPE, k_min=0, k_max=1)
    basis = BSplineBasis(degree=degree, knots_config=knots, input_map="real.clamp").double()
    coefficients = _seeded_control_points(2, 6, 1)
    u = torch.tensor(
        [
            [0.0, 0.2],
            [0.4, 1.0],
        ],
        dtype=DTYPE,
    )

    actual = basis(u, coefficients)
    expected = bspline_curves(u, coefficients, knots, degree)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_bspline_basis_normalizes_to_custom_knot_interval() -> None:
    degree = 3
    knots = uniform_augmented_knots(6, degree, dtype=DTYPE, k_min=0, k_max=1)
    basis = BSplineBasis(
        degree=degree,
        knots_config=knots,
        input_map=tc.maps.Real.arctan(scale=1.5),
    ).double()
    coefficients = _seeded_control_points(2, 6, 1)
    raw_u = torch.tensor(
        [
            [-3.0, 0.0],
            [1.0, 2.5],
        ],
        dtype=DTYPE,
    )

    actual = basis(raw_u, coefficients)
    expected = bspline_curves(
        arctan(raw_u, scale=1.5, out_min=0, out_max=1),
        coefficients,
        knots,
        degree,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_bspline_basis_supports_plain_callable_input_map() -> None:
    degree = 3
    basis = BSplineBasis(
        degree=degree,
        knots_config=6,
        input_map=_tanh_map,
    ).double()
    coefficients = _seeded_control_points(2, 6, 1)
    raw_u = torch.tensor(
        [
            [-3.0, 0.0],
            [1.0, 2.5],
        ],
        dtype=DTYPE,
    )

    actual = basis(raw_u, coefficients)
    expected = bspline_curves(
        _tanh_map(raw_u, *basis.parameter_range),
        coefficients,
        basis.knots,
        degree,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_bspline_basis_nonnegative_rational_maps_to_custom_parameter_range() -> None:
    degree = 3
    basis = BSplineBasis(
        degree=degree,
        knots_config=6,
        parameter_range=(0.0, 1.0),
        input_map="nonneg.rational",
    ).double()
    coefficients = _seeded_control_points(2, 6, 1)
    raw_u = torch.tensor(
        [
            [-3.0, 0.0],
            [1.0, 2.5],
        ],
        dtype=DTYPE,
    )

    actual = basis(raw_u, coefficients)
    expected = bspline_curves(
        tc.maps.Nonneg.rational()(raw_u, *basis.parameter_range),
        coefficients,
        basis.knots,
        degree,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"degree": -1}, "degree must be a non-negative integer."),
        (
            {"parameter_range": (1.0, 0.0)},
            "parameter_range must satisfy min < max. Got (1.0, 0.0).",
        ),
        (
            {"parameter_range": (0.0, 1.0), "knots_config": uniform_augmented_knots(6, 3)},
            "parameter_range can only be set when knots_config is an int.",
        ),
        (
            {"knots_config": torch.ones(2, 2)},
            "Provided knots_config tensor must be 1D.",
        ),
        (
            {"knots_config": 3, "degree": 3},
            "Number of control points (3) must be greater than the degree (3).",
        ),
        (
            {"knots_config": [1, 2, 3]},
            "knots_config must be an int (number of control points) or a torch.Tensor (knot vector).",
        ),
        (
            {"input_map": "rational"},
            "Unknown input_map rational",
        ),
    ],
)
def test_bspline_basis_rejects_invalid_constructor_inputs(kwargs: dict[str, object], expected_message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=re.escape(expected_message)):
        BSplineBasis(**kwargs)


def test_bspline_basis_rejects_input_map_that_is_not_string_or_callable() -> None:
    with pytest.raises(TypeError, match=re.escape("input_map must be a dotted preset string or a callable.")):
        BSplineBasis(input_map=123)  # type: ignore[arg-type]


def test_bspline_basis_rejects_invalid_u_rank() -> None:
    basis = BSplineBasis(degree=3, knots_config=7)
    coefficients = torch.randn(2, 7, 3)
    u = torch.randn(2, 1, 1)
    expected_message = f"Input u must be a 2D tensor of shape (batch_size, num_curves). Got shape: {u.shape}"

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        basis(u, coefficients)


def test_bspline_basis_rejects_invalid_coefficients_rank() -> None:
    basis = BSplineBasis(degree=3, knots_config=7)
    u = torch.randn(2, 3)
    coefficients = torch.randn(3, 7)
    expected_message = (
        "Input coefficients must be a 3D tensor of shape "
        "(num_curves, "
        f"n_control_points_per_curve={basis.n_control_points_per_curve}, dim). "
        f"Got shape: {coefficients.shape}"
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        basis(u, coefficients)


def test_bspline_basis_rejects_curve_count_mismatch() -> None:
    basis = BSplineBasis(degree=3, knots_config=7)
    u = torch.randn(2, 3)
    coefficients = torch.randn(2, 7, 1)
    expected_message = (
        "The number of curves must match between u and coefficients. "
        f"Got u.shape[1]={u.shape[1]} and coefficients.shape[0]={coefficients.shape[0]}."
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        basis(u, coefficients)


def test_bspline_basis_rejects_control_point_count_mismatch() -> None:
    basis = BSplineBasis(degree=3, knots_config=7)
    u = torch.randn(2, 3)
    coefficients = torch.randn(3, 6, 1)
    expected_message = (
        "The number of control points in coefficients must match this basis. "
        f"Expected {basis.n_control_points_per_curve}, got {coefficients.shape[1]}."
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        basis(u, coefficients)


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
