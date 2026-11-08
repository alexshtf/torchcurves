"""Compare matched KAN training steps with efficient-kan."""

# ruff: noqa: I001

import copy
import time
from itertools import pairwise

from common import SEED, summary

import torch
import torch.nn.functional as F  # noqa: N812
from efficient_kan import KANLinear
from torchcurves.functional import bspline_curves

WIDTHS = (32, 64, 64, 1)


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


class ClampedBaseline(torch.nn.Module):
    """Apply the same input clamp before every upstream layer."""

    def __init__(self, layers: list[KANLinear]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x.clamp(-1, 1))
        return x


def build_models(coefficients: int, device: str, dtype: torch.dtype, single_layer: bool = False):
    widths = WIDTHS[:2] if single_layer else WIDTHS
    layers = [
        KANLinear(input_width, output_width, grid_size=coefficients - 3, spline_order=3).to(device=device, dtype=dtype)
        for input_width, output_width in pairwise(widths)
    ]
    baseline = ClampedBaseline(layers)
    candidate = torch.nn.Sequential(*(CurveLayer(layer) for layer in layers))
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    if baseline_parameters != candidate_parameters:
        raise ValueError("The matched networks have different parameter counts")
    return baseline, candidate


def compare_parameters(baseline, candidate, *, gradients: bool, rtol: float, atol: float) -> None:
    """Compare corresponding tensors after accounting for their axis order."""
    for original, replacement in zip(baseline.layers, candidate, strict=True):
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


def train_step(model, optimizer, inputs, targets) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()


def check_match(coefficients: int, device: str, dtype: torch.dtype) -> None:
    """Check a layer and a network, including gradients and one Adam update."""
    rtol, atol = (1e-8, 1e-10) if dtype == torch.float64 else (1e-4, 1e-6)
    for single_layer in (True, False):
        torch.manual_seed(SEED)
        baseline, candidate = build_models(coefficients, device, dtype, single_layer)
        inputs = torch.empty(4, WIDTHS[0], device=device, dtype=dtype).uniform_(-0.8, 0.8)
        targets = torch.randn(4, 64 if single_layer else 1, device=device, dtype=dtype)
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


def check_finite(model) -> None:
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all() or (
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ):
            raise ValueError("Training produced a non-finite parameter or gradient")


def measure_blocks(
    models, optimizers, inputs, targets, device: str, repeats: int, steps: int
) -> dict[str, list[float]]:
    """Restore warmed states before each block; alternate the run order."""
    states = {
        name: (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizers[name].state_dict()))
        for name, model in models.items()
    }
    times = {name: [] for name in models}
    names = list(models)

    for repeat in range(repeats):
        for name in names if repeat % 2 == 0 else names[::-1]:
            model, optimizer = models[name], optimizers[name]
            model.load_state_dict(states[name][0])
            optimizer.load_state_dict(states[name][1])
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(steps):
                train_step(model, optimizer, inputs, targets)
            if device == "cuda":
                torch.cuda.synchronize()
            times[name].append((time.perf_counter() - start) * 1000 / steps)
            check_finite(model)
    return times


def measure_configuration(coefficients: int, batch: int, device: str) -> tuple[list[dict], list[dict]]:
    repeats, steps, warmup = 5, 100, 50
    seed = SEED + batch + coefficients
    torch.manual_seed(seed)
    baseline, candidate = build_models(coefficients, device, torch.float32)
    inputs = torch.empty(batch, WIDTHS[0], device=device).uniform_(-1, 1)
    targets = torch.randn(batch, 1, device=device)
    models = {"efficient-kan": baseline, "TorchCurves": candidate}
    optimizers = {name: adam(model) for name, model in models.items()}

    for name, model in models.items():
        for _ in range(warmup):
            train_step(model, optimizers[name], inputs, targets)
        check_finite(model)

    times = measure_blocks(models, optimizers, inputs, targets, device, repeats, steps)
    raw, aggregates = [], []
    for name, values in times.items():
        config = {
            "coefficients": coefficients,
            "batch": batch,
            "implementation": name,
            "device": device,
            "dtype": "float32",
            "threads": 1 if device == "cpu" else None,
            "seed": seed,
            "warmup_steps": warmup,
            "steps_per_block": steps,
        }
        raw.extend({**config, "repeat": repeat, "ms": value} for repeat, value in enumerate(values))
        aggregates.append({**config, **summary(values)})
    return raw, aggregates


def run() -> tuple[list[dict], list[dict]]:
    raw, aggregates = [], []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for coefficients in (8, 32):
        check_match(coefficients, "cpu", torch.float64)
        for device in devices:
            check_match(coefficients, device, torch.float32)
            for batch in (64, 256, 1024):
                samples, summaries = measure_configuration(coefficients, batch, device)
                raw.extend(samples)
                aggregates.extend(summaries)
    return raw, aggregates
