from functools import partial

import pytest
import torch

from torchcurves import BSplineBasis, BSplineCurve, LegendreCurve
from torchcurves.functional import uniform_augmented_knots

BASIS = partial(BSplineBasis, degree=3, knots_config=6)
CURVE = partial(BSplineCurve, 2, 1, degree=3, knots_config=6)


@pytest.mark.parametrize(
    ("factory", "mode"),
    [
        pytest.param(BASIS, "reset", id="basis-reset"),
        pytest.param(CURVE, "reset", id="curve-reset"),
        pytest.param(partial(LegendreCurve, 2, 1, degree=3), "reset", id="legendre-reset"),
        pytest.param(BASIS, "checkpoint", id="basis-checkpoint"),
        pytest.param(CURVE, "checkpoint", id="curve-checkpoint"),
        pytest.param(BASIS, "explicit-meta", id="basis-explicit-meta-checkpoint"),
        pytest.param(CURVE, "explicit-meta", id="curve-explicit-meta-checkpoint"),
    ],
)
def test_meta_materialization_reset_or_checkpoint(factory, mode: str) -> None:
    with torch.device("meta"):
        module = factory(**({"knots_config": torch.empty(10)} if mode == "explicit-meta" else {})).double()
    state = module.state_dict(keep_vars=True)
    assert all(tensor.is_meta and tensor.dtype == torch.float64 for tensor in state.values())
    module.to_empty(device="cpu")
    state = module.state_dict(keep_vars=True)
    if mode == "reset":
        with torch.no_grad():
            for tensor in state.values():
                tensor.fill_(torch.nan)
        module.apply(lambda child: child.reset_parameters())
        for name, tensor in module.state_dict(keep_vars=True).items():
            assert tensor is state[name]
            assert tensor.dtype == torch.float64 and tensor.device == torch.device("cpu")
            assert torch.isfinite(tensor).all()
        if not isinstance(module, LegendreCurve):
            expected_knots = torch.tensor([-1.0] * 4 + [-1 / 3, 1 / 3] + [1.0] * 4, dtype=torch.float64)
            torch.testing.assert_close(module.knots, expected_knots, rtol=0, atol=1e-15)
        return

    # Checkpoint knots change the constructor interval from [-1, 1] to [2, 5].
    knots = uniform_augmented_knots(6, 3, dtype=torch.float64, k_min=2.0, k_max=5.0)
    ordinary = factory(knots_config=knots).double()
    # Greville coefficients reproduce identity, so outputs equal the mapped input.
    coefficients = torch.stack([knots[j + 1 : j + 4].mean() for j in range(6)]).view(1, 6, 1).repeat(2, 1, 1)
    if isinstance(ordinary, BSplineCurve):
        with torch.no_grad():
            ordinary.control_points.copy_(coefficients)
    module.load_state_dict(ordinary.state_dict())
    torch.testing.assert_close(module.state_dict(), ordinary.state_dict(), rtol=0, atol=0)
    if mode == "explicit-meta":
        basis = module if isinstance(module, BSplineBasis) else module.basis
        with pytest.raises(NotImplementedError, match="meta tensor"):
            basis.reset_parameters()
    u = torch.tensor([[-4.0, -0.75], [0.0, 0.5], [1.5, 4.0]], dtype=torch.float64)
    output = module(u, coefficients) if isinstance(module, BSplineBasis) else module(u)
    expected = ordinary(u, coefficients) if isinstance(ordinary, BSplineBasis) else ordinary(u)
    mapped = 2.0 + 1.5 * (1.0 + u / torch.sqrt(1.0 + u.square()))
    assert module.parameter_range == (2.0, 5.0)
    assert output.shape == (3, 2, 1) and torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(output, mapped.unsqueeze(-1), rtol=1e-12, atol=1e-12)


def test_custom_knots_reset_and_float_bound_input_map() -> None:
    def custom_map(x, out_min, out_max):
        assert isinstance(out_min, float) and isinstance(out_max, float)
        return out_min + 0.5 * (out_max - out_min) * (torch.tanh(x) + 1.0)

    knots = torch.tensor([0.0] * 4 + [0.25, 0.75, 2.5] + [3.0] * 4, dtype=torch.float64)
    expected = knots.clone()
    basis = BSplineBasis(degree=3, knots_config=knots, input_map=custom_map)
    knots.fill_(99.0)  # Caller mutation must not change the reset recipe.
    basis.load_state_dict({"knots": expected + 1.0})
    basis.to("meta").to_empty(device="cpu")
    basis.reset_parameters()
    torch.testing.assert_close(basis.knots, expected, rtol=0, atol=0)
    assert set(basis.state_dict()) == {"knots"}
    coefficients = torch.stack([expected[j + 1 : j + 4].mean() for j in range(7)]).view(1, 7, 1)
    u = torch.tensor([[-1.5], [0.0], [2.5]], dtype=torch.float64)
    torch.testing.assert_close(basis(u, coefficients), (1.5 * (torch.tanh(u) + 1.0)).unsqueeze(-1))


def test_curve_reset_leaves_child_basis_unchanged() -> None:
    curve = BSplineCurve(2, 3, knots_config=8)
    curve.knots.add_(3.0)
    expected = curve.knots.clone()
    with torch.no_grad():
        curve.control_points.fill_(torch.nan)
    curve.reset_parameters()
    assert torch.isfinite(curve.control_points).all()
    torch.testing.assert_close(curve.knots, expected, rtol=0, atol=0)
