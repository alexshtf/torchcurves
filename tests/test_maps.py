import numpy as np
import pytest
import torch

import torchcurves as tc
from torchcurves.functional import arctan, clamp, rational

from ._references import assert_reference, numpy64

DTYPE = torch.float64


@pytest.mark.parametrize("kind", ["rational", "arctan", "clamp"])
def test_real_input_maps_match_numpy(kind: str, precision: tuple[torch.dtype, float, float]) -> None:
    dtype, _, _ = precision
    x = torch.tensor([-3.0, -0.5, 0.0, 0.4, 2.5], dtype=DTYPE).to(dtype)
    out_min, out_max, scale = -0.25, 1.5, 0.75
    factory = getattr(tc.maps.Real, kind)

    actual = factory(scale=scale)(x, out_min, out_max)
    x64 = numpy64(x)
    if kind == "clamp":
        expected = np.clip(x64 / scale, out_min, out_max)
    else:
        unit = x64 / np.sqrt(scale**2 + x64**2) if kind == "rational" else 2 * np.arctan(x64 / scale) / np.pi
        expected = (unit + 1) / 2 * (out_max - out_min) + out_min

    assert_reference(actual, expected, precision)


@pytest.mark.parametrize(("kind", "functional"), [("rational", rational), ("arctan", arctan), ("clamp", clamp)])
def test_real_input_map_objects_match_functional_helpers(kind: str, functional) -> None:
    x = torch.tensor([-3.0, -0.5, 0.0, 2.5], dtype=DTYPE)
    out_min = -0.25
    out_max = 1.5
    scale = 0.75

    factory = getattr(tc.maps.Real, kind)
    torch.testing.assert_close(
        factory(scale=scale)(x, out_min, out_max),
        functional(x, scale=scale, out_min=out_min, out_max=out_max),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("kind", ["rational", "arctan"])
def test_nonnegative_maps_use_the_full_target_interval(kind: str) -> None:
    x = torch.tensor([-3.0, 0.0, 2.0, 20.0], dtype=DTYPE)
    scale = 2.0
    out_min, out_max = -0.25, 1.5
    actual = getattr(tc.maps.Nonneg, kind)(scale=scale)(x, out_min, out_max)
    # Check nonnegative clamping and target-interval mapping here; the underlying
    # nonlinear maps have native precision coverage above.
    nonnegative = np.maximum(numpy64(x), 0)
    unit = (
        nonnegative / np.sqrt(scale**2 + nonnegative**2)
        if kind == "rational"
        else 2 * np.arctan(nonnegative / scale) / np.pi
    )
    expected = unit * (out_max - out_min) + out_min

    assert_reference(actual, expected, (DTYPE, 1e-12, 1e-12))
    assert actual[0].item() == out_min
    assert actual[1].item() == out_min
    assert out_min < actual[2].item() < actual[3].item() < out_max
