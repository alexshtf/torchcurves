import re
from typing import Optional

import numpy as np
import pytest
import torch

import torchcurves as tc
import torchcurves.functional._legendre as legendre_impl
from torchcurves import LegendreCurve
from torchcurves.functional import arctan, clamp, legendre_curves, rational

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


def _tanh_map(x: torch.Tensor, out_min: float, out_max: float) -> torch.Tensor:
    mapped = torch.tanh(x)
    return 0.5 * (mapped + 1.0) * (out_max - out_min) + out_min


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
    ("input_map", "expected_fn"),
    [
        ("real.rational", rational),
        ("real.arctan", arctan),
        ("real.clamp", clamp),
    ],
)
def test_legendre_curve_with_string_input_map_matches_manual_functional_path(input_map: str, expected_fn) -> None:
    model = LegendreCurve(num_curves=2, dim=3, degree=4, input_map=input_map).double()
    u = torch.tensor(
        [
            [-3.0, -0.4],
            [0.5, 2.0],
        ],
        dtype=DTYPE,
    )

    actual = model(u)
    expected = legendre_curves(expected_fn(u, out_min=-1.0, out_max=1.0), model.coefficients)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_legendre_curve_with_object_input_map_matches_manual_functional_path() -> None:
    input_map = tc.maps.Real.rational(scale=0.5)
    model = LegendreCurve(num_curves=2, dim=3, degree=4, input_map=input_map).double()
    u = torch.tensor(
        [
            [-3.0, -0.4],
            [0.5, 2.0],
        ],
        dtype=DTYPE,
    )

    actual = model(u)
    expected = legendre_curves(
        rational(u, scale=0.5, out_min=-1.0, out_max=1.0),
        model.coefficients,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert model.input_map is input_map


def test_legendre_curve_supports_plain_callable_input_map() -> None:
    model = LegendreCurve(num_curves=2, dim=3, degree=4, input_map=_tanh_map).double()
    u = torch.tensor(
        [
            [-3.0, -0.4],
            [0.5, 2.0],
        ],
        dtype=DTYPE,
    )

    actual = model(u)
    expected = legendre_curves(_tanh_map(u, -1.0, 1.0), model.coefficients)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("checkpoint_segments", "expected_checkpoint_calls"),
    [(3, 3), (13, 10)],
    ids=["uneven-segments", "more-segments-than-degree"],
)
def test_legendre_checkpointing_preserves_values_and_gradients(
    checkpoint_segments: int,
    expected_checkpoint_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    degree = 9
    num_curves = 2
    dim = 3
    generator = torch.Generator().manual_seed(1)
    x_values = 1.8 * torch.rand(3, num_curves, dtype=DTYPE, generator=generator) - 0.9
    coefficient_values = torch.randn(degree + 1, num_curves, dim, dtype=DTYPE, generator=generator)
    output_weight = torch.randn(3, num_curves, dim, dtype=DTYPE, generator=generator)

    def evaluate(segments: Optional[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x_values.clone().requires_grad_()
        coefficients = coefficient_values.clone().requires_grad_()
        output = legendre_curves(x, coefficients, checkpoint_segments=segments)
        (output * output_weight).sum().backward()
        assert x.grad is not None
        assert coefficients.grad is not None
        return output.detach(), x.grad, coefficients.grad

    expected = evaluate(segments=None)

    checkpoint_calls = 0
    original_checkpoint = legendre_impl._checkpoint

    def counting_checkpoint(fn, *args):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return original_checkpoint(fn, *args)

    monkeypatch.setattr(legendre_impl, "_checkpoint", counting_checkpoint)
    actual = evaluate(segments=checkpoint_segments)

    assert checkpoint_calls == expected_checkpoint_calls
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-12, atol=1e-12)


def test_legendre_checkpointing_is_skipped_when_gradients_are_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.tensor([[-0.8, 0.3], [0.1, 0.9]], dtype=DTYPE, requires_grad=True)
    coefficients = _seeded_coefficients(num_curves=2, degree=4, dim=3).requires_grad_()

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("Checkpointing must be skipped when gradients are disabled.")

    monkeypatch.setattr(legendre_impl, "_checkpoint", fail_if_called)
    with torch.no_grad():
        expected = legendre_curves(x, coefficients)
        actual = legendre_curves(x, coefficients, checkpoint_segments=3)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"degree": -1}, "degree must be a non-negative integer."),
        ({"checkpoint_segments": 0}, "checkpoint_segments must be a positive integer or None."),
        ({"input_map": "rational"}, "Unknown input_map rational"),
    ],
)
def test_legendre_curve_rejects_invalid_constructor_inputs(kwargs: dict[str, object], expected_message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=re.escape(expected_message)):
        base_kwargs: dict[str, object] = {"num_curves": 2, "dim": 3, "degree": 4}
        base_kwargs.update(kwargs)
        LegendreCurve(**base_kwargs)


def test_legendre_curve_rejects_input_map_that_is_not_string_or_callable() -> None:
    with pytest.raises(TypeError, match=re.escape("input_map must be a dotted preset string or a callable.")):
        LegendreCurve(num_curves=2, dim=3, degree=4, input_map=123)  # type: ignore[arg-type]


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
