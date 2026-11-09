"""Compare idiomatic KAN training steps in TorchCurves and efficient-kan."""

import copy
import time
from itertools import pairwise

from common import SEED

# isort: split
import torch
import torch.nn.functional as F  # noqa: N812
from efficient_kan import KAN

import torchcurves as tc

WIDTHS = (32, 64, 64, 1)
REPEATS = 5
STEPS = 100
WARMUP = 50


def build_models(coefficients: int, device: str) -> dict[str, torch.nn.Module]:
    efficient_kan = KAN(list(WIDTHS), grid_size=coefficients - 3, spline_order=3).to(device)
    layers = []
    for input_width, output_width in pairwise(WIDTHS):
        layers.append(tc.BSplineCurve(input_width, output_width, knots_config=coefficients))
        layers.append(tc.Sum())
    return {"efficient-kan": efficient_kan, "TorchCurves": torch.nn.Sequential(*layers).to(device)}


def adam(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0, foreach=False, fused=False)


def train_step(model, optimizer, inputs, targets) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return loss


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
    models = build_models(coefficients, device)
    inputs = torch.empty(batch, WIDTHS[0], device=device).uniform_(-1, 1)
    targets = torch.randn(batch, 1, device=device)
    times = measure_blocks(models, inputs, targets)
    measurements = []
    for name, values in times.items():
        config = {
            "coefficients": coefficients,
            "batch": batch,
            "implementation": name,
            "parameters": sum(parameter.numel() for parameter in models[name].parameters()),
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
        for device in devices:
            for batch in (64, 256, 1024):
                measurements.extend(measure_configuration(coefficients, batch, device))
    return measurements
