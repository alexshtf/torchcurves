import re

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

import torchcurves as tc
from torchcurves import BSplineBasis, BSplineCurve
from torchcurves.functional import bspline_curves, uniform_augmented_knots

from ._references import assert_reference, bspline_reference, numpy64

DTYPE = torch.float64
GRADCHECK_EPS = 1e-6
GRADCHECK_ATOL = 1e-4
GRADCHECK_RTOL = 1e-3
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
]


def _seeded_control_points(num_curves: int, n_control_points: int, dim: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(num_curves, n_control_points, dim, dtype=DTYPE, generator=generator)


def _nonuniform_knots(degree: int, dtype: torch.dtype) -> torch.Tensor:
    # Full multiplicity at -0.2 exercises discontinuous splines for positive degrees.
    internal_knots = [-0.6] + [-0.2] * (degree + 1) + [0.45]
    return torch.tensor([-1.0] * (degree + 1) + internal_knots + [1.0] * (degree + 1), dtype=dtype)


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


@pytest.mark.parametrize("degree", [0, 5])
@pytest.mark.parametrize("device", DEVICES)
def test_nonuniform_repeated_knots_and_noncontiguous_inputs_match_scipy(
    degree: int, device: str, precision: tuple[torch.dtype, float, float]
) -> None:
    dtype, rtol, atol = precision
    knots = _nonuniform_knots(degree, DTYPE).to(device=device, dtype=dtype)
    n_control_points = len(knots) - degree - 1
    control_points = (_seeded_control_points(2, n_control_points, 4) / 3).to(device=device, dtype=dtype)[..., ::2]
    interior = torch.unique_consecutive(knots[degree + 1 : -degree - 1])
    assert interior.numel() == 3  # Intended distinct knots must survive input rounding.
    below = torch.nextafter(interior, torch.full_like(interior, -torch.inf))
    above = torch.nextafter(interior, torch.full_like(interior, torch.inf))
    samples = torch.cat((knots[:1], below, interior, above, knots[-1:]))
    u = torch.stack((samples, samples.flip(0))).T
    assert not u.is_contiguous()
    assert not control_points.is_contiguous()

    actual = bspline_curves(u, control_points, knots, degree)
    expected, _, _ = bspline_reference(u, control_points, knots, degree)
    if dtype == torch.float32:
        rtol = 1e-6  # Retain the tighter existing FP32 repeated-knot budget.
    assert_reference(actual, expected, (dtype, rtol, atol), device=u.device)


@pytest.mark.parametrize("degree", [0, 5])
@pytest.mark.parametrize("device", DEVICES)
def test_float32_inputs_with_float64_knots_match_scipy(degree: int, device: str) -> None:
    # Preserve the existing mixed-knot precision contract separately from native precision.
    knots = _nonuniform_knots(degree, DTYPE).to(device)
    control_points = (_seeded_control_points(2, len(knots) - degree - 1, 4) / 3).to(device=device, dtype=torch.float32)[
        ..., ::2
    ]
    # A rounded FP32 argument can lie on a different side of an FP64 repeated
    # knot. Keep endpoints and those boundary samples in the mixed-knot check.
    u = (
        torch.tensor(
            [[-1.0, -0.8, -0.6, -0.2, 0.25, 0.6, 1.0], [1.0, 0.8, 0.45, 0.2, -0.2, -0.7, -1.0]],
            dtype=DTYPE,
        )
        .to(device=device, dtype=torch.float32)
        .T
    )
    assert not u.is_contiguous()
    assert not control_points.is_contiguous()
    actual = bspline_curves(u, control_points, knots, degree)
    expected, _, _ = bspline_reference(u, control_points, knots, degree)

    assert_reference(actual, expected, (torch.float32, 1e-6, 1e-6), device=u.device)


@pytest.mark.parametrize(
    ("degree", "dim", "gradient_mode", "repeated_knots"),
    [
        pytest.param(0, 2, "both", False, id="constant-both"),
        pytest.param(1, 2, "arguments", False, id="linear-arguments"),
        pytest.param(2, 3, "coefficients", False, id="quadratic-coefficients"),
        pytest.param(3, 1, "both", False, id="cubic-scalar-both"),
        pytest.param(3, 8, "both", False, id="cubic-vector-both"),
        pytest.param(3, 2, "both", True, id="repeated-knots-both"),
    ],
)
def test_bspline_values_and_gradients_match_scipy(
    degree: int,
    dim: int,
    gradient_mode: str,
    repeated_knots: bool,
    precision: tuple[torch.dtype, float, float],
) -> None:
    dtype, rtol, atol = precision
    knots_master = (
        _nonuniform_knots(degree, DTYPE) if repeated_knots else uniform_augmented_knots(degree + 4, degree, dtype=DTYPE)
    )
    knots = knots_master.to(dtype)
    coefficients = (_seeded_control_points(2, len(knots) - degree - 1, dim) / 3).to(dtype)
    coefficients.requires_grad_(gradient_mode != "arguments")
    # Interior arguments avoid undefined derivatives; duplicated rows exercise accumulation.
    u = torch.tensor(
        [[-0.75, -0.375], [-0.375, 0.125], [0.125, 0.625], [0.125, 0.625], [0.625, -0.75]], dtype=DTYPE
    ).to(dtype)
    u.requires_grad_(gradient_mode != "coefficients")
    generator = torch.Generator().manual_seed(17)
    upstream = (torch.rand(*u.shape, dim, dtype=DTYPE, generator=generator) * 1.5 - 0.75).to(dtype)
    actual = bspline_curves(u, coefficients, knots, degree)
    expected, grad_u, grad_coefficients = bspline_reference(u, coefficients, knots, degree, upstream)
    actual.backward(upstream)

    assert_reference(actual, expected, (dtype, rtol, atol), device=u.device)
    for leaf, reference in ((u, grad_u), (coefficients, grad_coefficients)):
        if not leaf.requires_grad:
            assert leaf.grad is None
            continue
        assert leaf.grad is not None
        assert_reference(leaf.grad, reference, precision, device=leaf.device)


def test_generated_knots_preserve_requested_dtype_shape_and_endpoints(
    precision: tuple[torch.dtype, float, float],
) -> None:
    dtype, _, _ = precision
    degree, n_control_points = 3, 7
    knots = uniform_augmented_knots(n_control_points, degree, dtype=dtype, k_min=-0.5, k_max=1.5)

    assert knots.dtype == dtype
    assert knots.device == torch.device("cpu")
    assert knots.shape == (n_control_points + degree + 1,)
    assert torch.isfinite(knots).all()
    assert torch.all(knots[: degree + 1] == -0.5)
    assert torch.all(knots[-degree - 1 :] == 1.5)
    assert torch.all(knots[degree + 1 : n_control_points + 1] > knots[degree:n_control_points])


@pytest.mark.parametrize("degree", [0, 5])
@pytest.mark.parametrize("device", DEVICES)
def test_nonuniform_repeated_knots_gradcheck(degree: int, device: str) -> None:
    knots = _nonuniform_knots(degree, DTYPE).to(device)
    n_control_points = len(knots) - degree - 1
    control_points = _seeded_control_points(2, n_control_points, 2).to(device).requires_grad_()
    # Stay away from the knot discontinuities while checking both differentiable inputs.
    u = torch.tensor([[-0.85, -0.4], [0.1, 0.7]], dtype=DTYPE, device=device, requires_grad=True)

    assert gradcheck(
        lambda u_arg, cp_arg: bspline_curves(u_arg, cp_arg, knots, degree),
        (u, control_points),
        eps=GRADCHECK_EPS,
        atol=GRADCHECK_ATOL,
        rtol=GRADCHECK_RTOL,
    )


@pytest.mark.parametrize(
    ("batch", "curves", "degree", "count", "dim"),
    [(1, 3, 3, 7, 2), (4, 1, 2, 6, 2), (3, 2, 3, 8, 3)],
)
def test_batching_matches_individual_evaluations(batch, curves, degree, count, dim) -> None:
    coefficients = _seeded_control_points(curves, count, dim)
    knots = uniform_augmented_knots(count, degree, dtype=DTYPE)
    u = torch.linspace(-0.9, 1.0, batch * curves, dtype=DTYPE).reshape(batch, curves)
    actual = bspline_curves(u, coefficients, knots, degree)
    assert actual.shape == (batch, curves, dim)
    for b in range(batch):
        for m in range(curves):
            expected = bspline_curves(u[b : b + 1, m : m + 1], coefficients[m : m + 1], knots, degree)
            torch.testing.assert_close(actual[b : b + 1, m : m + 1], expected, rtol=1e-12, atol=1e-12)


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
    # Compute the mapping analytically instead of calling the implementation under test.
    mapped = torch.from_numpy(np.arctan(numpy64(raw_u) / 1.5) / np.pi + 0.5)
    expected, _, _ = bspline_reference(mapped, coefficients, knots, degree)

    assert actual.dtype == DTYPE
    assert actual.device == raw_u.device
    assert torch.isfinite(actual).all()
    np.testing.assert_allclose(numpy64(actual), expected, rtol=1e-12, atol=1e-12)


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


@pytest.mark.parametrize(
    ("u_shape", "coefficient_shape", "message"),
    [
        ((2, 1, 1), (2, 7, 3), "Input u must be a 2D tensor"),
        ((2, 3), (3, 7), "Input coefficients must be a 3D tensor"),
        ((2, 3), (2, 7, 1), "The number of curves must match"),
        ((2, 3), (3, 6, 1), "The number of control points in coefficients must match"),
    ],
)
def test_bspline_basis_rejects_invalid_shapes(u_shape, coefficient_shape, message) -> None:
    basis = BSplineBasis(degree=3, knots_config=7)
    with pytest.raises(ValueError, match=message):
        basis(torch.empty(u_shape), torch.empty(coefficient_shape))


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
