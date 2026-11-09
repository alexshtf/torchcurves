"""Compare idiomatic KAN training steps in TorchCurves and efficient-kan."""

import copy
import time
from itertools import pairwise, product

import torch
import torch.nn.functional as F  # noqa: N812
from common import SEED, progress
from efficient_kan import KAN

import torchcurves as tc

INPUT_WIDTH = 32
OUTPUT_WIDTH = 1
HIDDEN_WIDTHS = (32, 64, 128)
BATCHES = (64, 256, 1024)
COEFFICIENTS = (8, 32, 64)
REPEATS = 5
STEPS = 100
WARMUP = 50


def build_models(coefficients: int, hidden_width: int, device: str) -> dict[str, torch.nn.Module]:
    widths = [INPUT_WIDTH, hidden_width, hidden_width, OUTPUT_WIDTH]
    efficient_kan = KAN(widths, grid_size=coefficients - 3, spline_order=3).to(device)
    layers = []
    for input_width, output_width in pairwise(widths):
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


def measure_blocks(
    models, inputs, targets, *, warmup=WARMUP, repeats=REPEATS, steps=STEPS, report=None
) -> dict[str, list[float]]:
    """Warm up, then time alternating blocks from independent copies of the warmed state."""
    runs = {name: (model, adam(model)) for name, model in models.items()}
    states = {}
    for name, (model, optimizer) in runs.items():
        if report is not None:
            report(f"{name} warm-up")
        for _ in range(warmup):
            loss = train_step(model, optimizer, inputs, targets)
        check_finite(model, loss)
        states[name] = copy.deepcopy((model.state_dict(), optimizer.state_dict()))

    times = {name: [] for name in runs}
    for repeat in range(repeats):
        order = runs.items() if repeat % 2 == 0 else reversed(runs.items())
        for name, (model, optimizer) in order:
            if report is not None:
                report(f"{name} block {repeat + 1}/{repeats}")
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


def measure_configuration(
    coefficients: int, hidden_width: int, batch: int, device: str, *, report=None
) -> list[tuple[dict, list[float]]]:
    seed = SEED + batch + hidden_width + coefficients
    torch.manual_seed(seed)
    models = build_models(coefficients, hidden_width, device)
    inputs = torch.empty(batch, INPUT_WIDTH, device=device).uniform_(-1, 1)
    targets = torch.randn(batch, OUTPUT_WIDTH, device=device)
    times = measure_blocks(models, inputs, targets, report=report)
    measurements = []
    for name, values in times.items():
        config = {
            "coefficients": coefficients,
            "hidden_width": hidden_width,
            "batch": batch,
            "implementation": name,
            "parameters": sum(parameter.numel() for parameter in models[name].parameters()),
            "device": device,
            "dtype": "float32",
            "threads": torch.get_num_threads(),
            "seed": seed,
            "warmup_steps": WARMUP,
            "steps_per_block": STEPS,
        }
        measurements.append((config, values))
    return measurements


def run() -> list[tuple[dict, list[float]]]:
    measurements = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    configurations = list(product(devices, COEFFICIENTS, HIDDEN_WIDTHS, BATCHES))
    with progress(len(configurations), "KAN") as bar:
        for device, coefficients, hidden_width, batch in configurations:
            bar.set_description_str(f"KAN {device.upper()} B={batch} W={hidden_width} K={coefficients}", refresh=False)
            bar.set_postfix_str("initializing")
            measurements.extend(
                measure_configuration(coefficients, hidden_width, batch, device, report=bar.set_postfix_str)
            )
            bar.update()
        bar.set_description_str("KAN", refresh=False)
        bar.set_postfix_str("done")
    return measurements
