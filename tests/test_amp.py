"""Test autocast values, gradients, and dtype behavior across supported execution modes."""

import pytest
import torch

from torchcurves import BSplineBasis
from torchcurves.functional import bspline_curves, uniform_augmented_knots

from ._references import assert_reference, bspline_reference


@pytest.fixture(
    params=[
        ("cpu", torch.bfloat16),
        ("cuda", torch.float16),
        ("cuda", torch.bfloat16),
        ("xpu", torch.float16),
        ("xpu", torch.bfloat16),
    ],
    ids=["cpu-bf16", "cuda-fp16", "cuda-bf16", "xpu-fp16", "xpu-bf16"],
)
def amp_backend(request):
    device, dtype = request.param
    backend = getattr(torch, device, None)
    if device != "cpu" and (backend is None or not backend.is_available()):
        pytest.skip(f"{device.upper()} hardware/runtime is unavailable")
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("This CUDA device does not support BF16")
    available = getattr(torch.amp.autocast_mode, "is_autocast_available", None)
    if available is not None and not available(device):
        pytest.skip(f"PyTorch {torch.__version__} does not support {device} autocast")
    return device, dtype


# D=1 and D=8 select different backward contractions. Each also covers all
# gradient modes, with both FP32 and upstream-autocast arguments represented.
@pytest.mark.parametrize(
    ("dim", "mixed_input", "gradients", "use_basis"),
    [
        (1, False, "arguments", False),
        (1, True, "coefficients", False),
        (1, True, "both", False),
        (8, True, "arguments", False),
        (8, False, "coefficients", False),
        (8, False, "both", False),
        (8, True, "both", True),
    ],
    ids=[
        "scalar-arguments",
        "scalar-coefficients",
        "scalar-both",
        "vector-arguments",
        "vector-coefficients",
        "vector-both",
        "basis",
    ],
)
def test_autocast_values_and_gradients(amp_backend, dim, mixed_input, gradients, use_basis):
    device, amp_dtype = amp_backend
    generator = torch.Generator().manual_seed(23)
    u = torch.tensor(
        [[-0.8125, -0.625], [-0.375, 0.1875], [0.125, 0.3125], [0.125, 0.3125], [0.625, 0.8125]], dtype=torch.float64
    ).to(device=device, dtype=amp_dtype if mixed_input else torch.float32)
    coefficients = (0.8 * torch.randn(2, 7, dim, dtype=torch.float64, generator=generator)).to(
        device=device, dtype=torch.float32
    )
    u.requires_grad_(gradients != "coefficients")
    coefficients.requires_grad_(gradients != "arguments")
    knots = uniform_augmented_knots(7, 3, dtype=torch.float32, device=device)
    if use_basis:
        basis = BSplineBasis(degree=3, knots_config=knots, input_map="real.clamp").to(device)

    with torch.autocast(device, dtype=amp_dtype):
        output = basis(u, coefficients) if use_basis else bspline_curves(u, coefficients, knots, 3)
    upstream = torch.randn(output.shape, dtype=torch.float64, generator=generator).to(output)
    output.backward(upstream)
    values, grad_u, grad_coefficients = bspline_reference(u, coefficients, knots, 3, upstream)

    # The matmul result must be low precision (also verifies autocast is active).
    # Leaf gradient storage retains its dtype; arithmetic still has an AMP budget.
    rtol, atol = (5e-3, 1e-3) if amp_dtype == torch.float16 else (3e-2, 1e-2)
    assert_reference(output, values, (amp_dtype, rtol, atol), device=u.device)
    for leaf, expected in [(u, grad_u), (coefficients, grad_coefficients)]:
        if leaf.requires_grad:
            assert leaf.grad is not None
            assert_reference(leaf.grad, expected, (leaf.dtype, rtol, atol), device=leaf.device)
        else:
            assert leaf.grad is None
