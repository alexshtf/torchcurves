"""Compare matched KAN training steps with efficient-kan."""

import copy
import time
from itertools import pairwise

from common import SEED

# isort: split
import torch
import torch.nn.functional as F  # noqa: N812
from efficient_kan import KANLinear

from torchcurves.functional import bspline_curves

WIDTHS = (32, 64, 64, 1)
REPEATS = 5
STEPS = 100
WARMUP = 50


class CurveLayer(torch.nn.Module):
    """An efficient-kan layer evaluated with TorchCurves' B-spline function."""

    def __init__(self, source: KANLinear):
        super().__init__()
        if source.spline_order != 3 or source.spline_weight.shape[-1] != source.grid_size + 3:
            raise ValueError("Expected cubic splines with grid_size + 3 coefficients")
        if not torch.equal(source.grid, source.grid[0].expand_as(source.grid)):
            raise ValueError("Input features must share one knot vector")

        self.base_weight = torch.nn.Parameter(source.base_weight.detach().clone())
        self.raw_coefficients = torch.nn.Parameter(
            source.spline_weight.detach().permute(1, 2, 0).clone(memory_format=torch.contiguous_format)
        )
        self.spline_scaler = torch.nn.Parameter(
            source.spline_scaler.detach().T.clone(memory_format=torch.contiguous_format)
        )
        self.register_buffer("knots", source.grid[0].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1, 1)
        coefficients = self.raw_coefficients * self.spline_scaler[:, None, :]
        base = F.linear(F.silu(x), self.base_weight)
        spline = bspline_curves(x, coefficients, self.knots, degree=3).sum(dim=1)
        return base + spline


class ClampedKANLinear(KANLinear):
    """Apply the same input clamp before every upstream layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.clamp(-1, 1))


def build_models(coefficients: int, device: str, dtype: torch.dtype, widths: tuple[int, ...] = WIDTHS):
    baseline = torch.nn.Sequential(
        *(
            ClampedKANLinear(input_width, output_width, grid_size=coefficients - 3, spline_order=3)
            for input_width, output_width in pairwise(widths)
        )
    ).to(device=device, dtype=dtype)
    candidate = torch.nn.Sequential(*(CurveLayer(layer) for layer in baseline))
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    if baseline_parameters != candidate_parameters:
        raise ValueError("The matched networks have different parameter counts")
    return baseline, candidate


def compare_parameters(baseline, candidate, *, gradients: bool, rtol: float, atol: float) -> None:
    """Compare corresponding tensors after accounting for their axis order."""
    for original, replacement in zip(baseline, candidate, strict=True):
        base = original.base_weight.grad if gradients else original.base_weight
        coefficients = original.spline_weight.grad if gradients else original.spline_weight
        scaler = original.spline_scaler.grad if gradients else original.spline_scaler
        expected = (base, coefficients.permute(1, 2, 0), scaler.T)
        actual = (
            replacement.base_weight.grad if gradients else replacement.base_weight,
            replacement.raw_coefficients.grad if gradients else replacement.raw_coefficients,
            replacement.spline_scaler.grad if gradients else replacement.spline_scaler,
        )
        for source, target in zip(expected, actual, strict=True):
            torch.testing.assert_close(source, target, rtol=rtol, atol=atol)


def adam(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0, foreach=False, fused=False)


def train_step(model, optimizer, inputs, targets) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return loss


def check_match(coefficients: int, device: str, dtype: torch.dtype) -> None:
    """Check a layer and a network, including gradients and one Adam update."""
    rtol, atol = (1e-8, 1e-10) if dtype == torch.float64 else (1e-4, 1e-6)
    for widths in (WIDTHS[:2], WIDTHS):
        torch.manual_seed(SEED)
        baseline, candidate = build_models(coefficients, device, dtype, widths)
        inputs = torch.empty(4, widths[0], device=device, dtype=dtype).uniform_(-0.8, 0.8)
        targets = torch.randn(4, widths[-1], device=device, dtype=dtype)
        baseline_inputs = inputs.clone().requires_grad_()
        candidate_inputs = inputs.clone().requires_grad_()

        baseline_output = baseline(baseline_inputs)
        candidate_output = candidate(candidate_inputs)
        torch.testing.assert_close(baseline_output, candidate_output, rtol=rtol, atol=atol)
        baseline_loss = F.mse_loss(baseline_output, targets)
        candidate_loss = F.mse_loss(candidate_output, targets)
        torch.testing.assert_close(baseline_loss, candidate_loss, rtol=rtol, atol=atol)

        baseline_loss.backward()
        candidate_loss.backward()
        torch.testing.assert_close(baseline_inputs.grad, candidate_inputs.grad, rtol=rtol, atol=atol)
        compare_parameters(baseline, candidate, gradients=True, rtol=rtol, atol=atol)

        train_step(baseline, adam(baseline), inputs, targets)
        train_step(candidate, adam(candidate), inputs, targets)
        torch.testing.assert_close(baseline(inputs), candidate(inputs), rtol=rtol, atol=atol)
        compare_parameters(baseline, candidate, gradients=False, rtol=rtol, atol=atol)


def check_finite(model, loss) -> None:
    if not torch.isfinite(loss):
        raise ValueError("Training produced a non-finite loss")
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all() or (
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ):
            raise ValueError("Training produced a non-finite parameter or gradient")


def measure_blocks(models, inputs, targets, *, warmup=WARMUP, repeats=REPEATS, steps=STEPS) -> dict[str, list[float]]:
    """Warm up, then time alternating blocks from independent copies of the warmed state."""
    runs = {name: (model, adam(model)) for name, model in models.items()}
    states = {}
    for name, (model, optimizer) in runs.items():
        for _ in range(warmup):
            loss = train_step(model, optimizer, inputs, targets)
        check_finite(model, loss)
        states[name] = copy.deepcopy((model.state_dict(), optimizer.state_dict()))

    times = {name: [] for name in runs}
    for repeat in range(repeats):
        order = runs.items() if repeat % 2 == 0 else reversed(runs.items())
        for name, (model, optimizer) in order:
            # Adam updates its state tensors in place; never hand it the saved snapshot.
            model_state, optimizer_state = copy.deepcopy(states[name])
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            if inputs.is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(steps):
                loss = train_step(model, optimizer, inputs, targets)
            if inputs.is_cuda:
                torch.cuda.synchronize()
            times[name].append((time.perf_counter() - start) * 1000 / steps)
            check_finite(model, loss)
    return times


def measure_configuration(coefficients: int, batch: int, device: str) -> list[tuple[dict, list[float]]]:
    seed = SEED + batch + coefficients
    torch.manual_seed(seed)
    baseline, candidate = build_models(coefficients, device, torch.float32)
    inputs = torch.empty(batch, WIDTHS[0], device=device).uniform_(-1, 1)
    targets = torch.randn(batch, 1, device=device)
    models = {"efficient-kan": baseline, "TorchCurves": candidate}
    times = measure_blocks(models, inputs, targets)
    measurements = []
    for name, values in times.items():
        config = {
            "coefficients": coefficients,
            "batch": batch,
            "implementation": name,
            "device": device,
            "dtype": "float32",
            "threads": 1 if device == "cpu" else None,
            "seed": seed,
            "warmup_steps": WARMUP,
            "steps_per_block": STEPS,
        }
        measurements.append((config, values))
    return measurements


def run() -> list[tuple[dict, list[float]]]:
    measurements = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for coefficients in (8, 32):
        check_match(coefficients, "cpu", torch.float64)
        for device in devices:
            check_match(coefficients, device, torch.float32)
            for batch in (64, 256, 1024):
                measurements.extend(measure_configuration(coefficients, batch, device))
    return measurements
