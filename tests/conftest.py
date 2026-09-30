import pytest
import torch

# Forward budgets for modest-degree curves with bounded, unit-scale inputs.
# This fixture is opt-in: validation, consistency and gradcheck tests stay small.
PRECISION_CASES = [
    pytest.param((torch.float64, 1e-10, 1e-12), id="fp64"),
    pytest.param((torch.float32, 1e-5, 1e-6), id="fp32"),
    pytest.param((torch.float16, 5e-3, 1e-3), id="fp16"),
    pytest.param((torch.bfloat16, 3e-2, 1e-2), id="bf16"),
]


@pytest.fixture(params=PRECISION_CASES)
def precision(request: pytest.FixtureRequest) -> tuple[torch.dtype, float, float]:
    return request.param
