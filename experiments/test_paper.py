"""Curve agreement, idiomatic KAN training, repeated blocks, and saved results."""

import csv
import json
from itertools import product

from common import SEED, prepare, save

# isort: split
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from curves import evaluators
from efficient_kan import KAN
from kan import INPUT_WIDTH, OUTPUT_WIDTH, adam, build_models, check_finite, measure_blocks, train_step
from render import curve_figure, kan_figure, plot_curves, plot_kan

import torchcurves as tc

prepare()


@pytest.mark.parametrize("family", ("Legendre", "B-spline"))
@pytest.mark.parametrize("batch,count,dimension", ((1, 1, 1), (5, 3, 7)))
@pytest.mark.parametrize("coefficients", (4, 16, 64))
def test_curve_references(family, batch, count, dimension, coefficients) -> None:
    with torch.inference_mode():
        candidate, reference, _ = evaluators(family, batch, count, dimension, coefficients, np.random.default_rng(SEED))
        actual = candidate()
        expected = torch.from_numpy(reference())
        assert actual.shape == expected.shape == (batch, count, dimension)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_curve_heatmaps(tmp_path) -> None:
    ratios = np.array([[0.01, 0.1, 1], [3, 10, 100]])
    results = []
    for family, reference in (("Legendre", "NumPy"), ("B-spline", "SciPy")):
        reference_ms = 2 if family == "Legendre" else 5
        for count, dimension, (row, batch), (column, coefficients) in product(
            (1, 3), (1, 7), enumerate((1, 5)), enumerate((4, 16, 64))
        ):
            config = {
                "family": family,
                "curves": count,
                "outputs": dimension,
                "batch": batch,
                "coefficients": coefficients,
            }
            candidate_ms = reference_ms * ratios[row, column] * count * dimension
            results.extend(
                [
                    {**config, "implementation": reference, "median_ms": reference_ms},
                    {**config, "implementation": "TorchCurves", "median_ms": candidate_ms},
                ]
            )
    results.reverse()  # Rendering must follow axis values, not measurement order.

    for family in ("Legendre", "B-spline"):
        figure = curve_figure(results, family)
        for axis, (count, dimension) in zip(figure.axes[:4], product((1, 3), (1, 7)), strict=True):
            expected = ratios * count * dimension
            np.testing.assert_allclose(axis.images[0].get_array(), expected)
            assert [text.get_text() for text in axis.texts] == [f"{value:.2g}" for value in expected.flat]
            assert [tick.get_text() for tick in axis.get_xticklabels()] == ["4", "16", "64"]
            assert (axis.images[0].norm.vmin, axis.images[0].norm.vmax) == (0.01, 100)
        plt.close(figure)

    plot_curves(results, tmp_path)
    for stem in ("legendre", "bspline"):
        for extension in ("pdf", "png"):
            assert (tmp_path / f"{stem}.{extension}").stat().st_size > 0
        assert "Panel rows show curve count C" in (tmp_path / f"{stem}_caption.md").read_text()


@pytest.mark.parametrize("hidden_width,coefficients", ((32, 8), (64, 32), (128, 64)))
@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ),
)
def test_kan_training(hidden_width, coefficients, device) -> None:
    torch.manual_seed(SEED)
    models = build_models(coefficients, hidden_width, device)
    assert type(models["efficient-kan"]) is KAN
    assert [type(layer) for layer in models["TorchCurves"]] == [tc.BSplineCurve, tc.Sum] * 3
    assert models["efficient-kan"].layers[0].spline_weight.shape[-1] == coefficients
    assert models["TorchCurves"][0].n_control_points_per_curve == coefficients
    connections = INPUT_WIDTH * hidden_width + hidden_width**2 + hidden_width * OUTPUT_WIDTH
    assert sum(p.numel() for p in models["TorchCurves"].parameters()) == connections * coefficients
    assert sum(p.numel() for p in models["efficient-kan"].parameters()) == connections * (coefficients + 2)

    inputs = torch.empty(4, INPUT_WIDTH, device=device).uniform_(-1, 1)
    targets = torch.randn(4, OUTPUT_WIDTH, device=device)
    for model in models.values():
        assert model(inputs).shape == targets.shape
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loss = train_step(model, adam(model), inputs, targets)
        check_finite(model, loss)
        assert all(parameter.grad is not None for parameter in model.parameters())
        assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))


class RecordingLinear(torch.nn.Linear):
    def __init__(self, name, history):
        super().__init__(1, 1, bias=False)
        self.name = name
        self.history = history

    def forward(self, inputs):
        self.history.append((self.name, self.weight.detach().clone()))
        return super().forward(inputs)


def test_training_blocks_restore_state_and_alternate() -> None:
    torch.manual_seed(SEED)
    history = []
    reports = []
    models = {name: RecordingLinear(name, history) for name in ("first", "second")}
    times = measure_blocks(
        models, torch.ones(2, 1), torch.zeros(2, 1), warmup=2, repeats=3, steps=2, report=reports.append
    )
    measured = history[4:]  # Two warm-up steps for each model.
    assert [name for name, _ in measured[::2]] == ["first", "second", "second", "first", "first", "second"]
    assert all(len(values) == 3 for values in times.values())
    assert reports == [
        "first warm-up",
        "second warm-up",
        "first block 1/3",
        "second block 1/3",
        "second block 2/3",
        "first block 2/3",
        "first block 3/3",
        "second block 3/3",
    ]
    for name in models:
        weights = torch.stack([weight for implementation, weight in measured if implementation == name]).reshape(3, 2)
        torch.testing.assert_close(weights, weights[0].expand_as(weights), rtol=0, atol=0)


def test_saved_results(tmp_path) -> None:
    save(tmp_path, "example", [({"seed": SEED}, [1, 2, 3, 4, 5])])
    raw = json.loads((tmp_path / "example_raw.json").read_text())
    assert [row["repeat"] for row in raw] == [0, 1, 2, 3, 4]
    assert [row["ms"] for row in raw] == [1, 2, 3, 4, 5]
    assert json.loads((tmp_path / "example_summary.json").read_text()) == [
        {"seed": SEED, "median_ms": 3, "q1_ms": 2, "q3_ms": 4},
    ]
    for kind in ("raw", "summary"):
        expected = json.loads((tmp_path / f"example_{kind}.json").read_text())
        with (tmp_path / f"example_{kind}.csv").open(newline="") as file:
            assert list(csv.DictReader(file)) == [{key: str(value) for key, value in row.items()} for row in expected]


@pytest.mark.parametrize("devices", (("cpu",), ("cpu", "cuda")))
def test_kan_heatmap(devices, tmp_path) -> None:
    ratios = np.array([[0.1, 1], [10, 100]])
    results = []
    for device, coefficients, (row, width), (column, batch) in product(
        devices, (8, 32, 64), enumerate((32, 128)), enumerate((64, 1024))
    ):
        config = {
            "device": device,
            "coefficients": coefficients,
            "hidden_width": width,
            "batch": batch,
            "warmup_steps": 50,
            "steps_per_block": 100,
        }
        reference_ms = 2 if device == "cpu" else 5
        ratio = ratios[row, column] * coefficients / 8 * (1 if device == "cpu" else 0.5)
        results.extend(
            [
                {**config, "implementation": "efficient-kan", "median_ms": reference_ms},
                {**config, "implementation": "TorchCurves", "median_ms": reference_ms * ratio},
            ]
        )
    results.reverse()

    figure = kan_figure(results)
    for axis, (device, coefficients) in zip(figure.axes[:6], product(("cpu", "cuda"), (8, 32, 64)), strict=True):
        if device not in devices:
            assert not axis.images
            assert [text.get_text() for text in axis.texts] == ["Not measured"]
            continue
        expected = ratios * coefficients / 8 * (1 if device == "cpu" else 0.5)
        np.testing.assert_allclose(axis.images[0].get_array(), expected)
        assert [text.get_text() for text in axis.texts] == [f"{value:.2g}" for value in expected.flat]
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["64", "1024"]
        assert (axis.images[0].norm.vmin, axis.images[0].norm.vmax) == (0.01, 100)
    plt.close(figure)

    plot_kan(results, {"cpu": "test CPU", "gpu": "test GPU" if "cuda" in devices else None}, tmp_path)
    for extension in ("pdf", "png"):
        assert (tmp_path / f"kan.{extension}").stat().st_size > 0
    assert "[32, W, W, 1]" in (tmp_path / "kan_caption.md").read_text()
