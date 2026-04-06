import torch

import torchcurves as tc
from torchcurves.functional import arctan, clamp, rational

DTYPE = torch.float64


def test_real_input_map_objects_match_functional_helpers() -> None:
    x = torch.tensor([-3.0, -0.5, 0.0, 2.5], dtype=DTYPE)
    out_min = -0.25
    out_max = 1.5
    scale = 0.75

    torch.testing.assert_close(
        tc.maps.Real.rational(scale=scale)(x, out_min, out_max),
        rational(x, scale=scale, out_min=out_min, out_max=out_max),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        tc.maps.Real.arctan(scale=scale)(x, out_min, out_max),
        arctan(x, scale=scale, out_min=out_min, out_max=out_max),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        tc.maps.Real.clamp(scale=scale)(x, out_min, out_max),
        clamp(x, scale=scale, out_min=out_min, out_max=out_max),
        rtol=1e-12,
        atol=1e-12,
    )


def test_nonnegative_rational_map_uses_the_full_target_interval() -> None:
    x = torch.tensor([-3.0, 0.0, 2.0, 20.0], dtype=DTYPE)
    scale = 2.0
    out_min = 0.0
    out_max = 1.0

    actual = tc.maps.Nonneg.rational(scale=scale)(x, out_min, out_max)

    expected_nonnegative = torch.clamp_min(x, 0)
    expected = expected_nonnegative / torch.sqrt(scale**2 + expected_nonnegative.square())

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[0].item() == out_min
    assert actual[1].item() == out_min
    assert out_min < actual[2].item() < actual[3].item() < out_max


def test_nonnegative_arctan_map_uses_the_full_target_interval() -> None:
    x = torch.tensor([-3.0, 0.0, 2.0, 20.0], dtype=DTYPE)
    scale = 2.0
    out_min = 0.0
    out_max = 1.0

    actual = tc.maps.Nonneg.arctan(scale=scale)(x, out_min, out_max)

    expected_nonnegative = torch.clamp_min(x, 0)
    expected = 2 * torch.arctan(expected_nonnegative / scale) / torch.pi

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[0].item() == out_min
    assert actual[1].item() == out_min
    assert out_min < actual[2].item() < actual[3].item() < out_max
