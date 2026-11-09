"""Curve agreement, idiomatic KAN training, repeated blocks, and saved results."""

import csv
import json

from common import SEED, prepare, save

# isort: split
import numpy as np
import pytest
import torch
from curves import check_outputs, evaluators
from efficient_kan import KAN
from kan import WIDTHS, adam, build_models, check_finite, measure_blocks, train_step
from render import table_rows, write_table

import torchcurves as tc

prepare()


@pytest.mark.parametrize("family", ("Legendre", "B-spline"))
@pytest.mark.parametrize("count", (1, 32))
@pytest.mark.parametrize("batch", (1, 32))
def test_curve_references(family, count, batch) -> None:
    with torch.inference_mode():
        candidate, reference, _ = evaluators(family, batch, count, np.random.default_rng(SEED))
        check_outputs(candidate, reference, batch, count)


@pytest.mark.parametrize("coefficients", (8, 32))
@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ),
)
def test_kan_training(coefficients, device) -> None:
    torch.manual_seed(SEED)
    models = build_models(coefficients, device)
    assert type(models["efficient-kan"]) is KAN
    assert [type(layer) for layer in models["TorchCurves"]] == [tc.BSplineCurve, tc.Sum] * 3
    assert models["efficient-kan"].layers[0].spline_weight.shape[-1] == coefficients
    assert models["TorchCurves"][0].n_control_points_per_curve == coefficients

    inputs = torch.empty(4, WIDTHS[0], device=device).uniform_(-1, 1)
    targets = torch.randn(4, WIDTHS[-1], device=device)
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
    models = {name: RecordingLinear(name, history) for name in ("first", "second")}
    times = measure_blocks(models, torch.ones(2, 1), torch.zeros(2, 1), warmup=2, repeats=3, steps=2)
    measured = history[4:]  # Two warm-up steps for each model.
    assert [name for name, _ in measured[::2]] == ["first", "second", "second", "first", "first", "second"]
    assert all(len(values) == 3 for values in times.values())
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


def test_table_marks_only_separated_intervals(tmp_path) -> None:
    config = {"coefficients": 8, "device": "cpu", "warmup_steps": 50, "steps_per_block": 100}
    results = [
        {**config, "batch": 64, "implementation": "TorchCurves", "q1_ms": 0.9, "median_ms": 1, "q3_ms": 1.1},
        {**config, "batch": 64, "implementation": "efficient-kan", "q1_ms": 2, "median_ms": 2.5, "q3_ms": 3},
        {**config, "batch": 256, "implementation": "TorchCurves", "q1_ms": 0.8, "median_ms": 1, "q3_ms": 1.6},
        {**config, "batch": 256, "implementation": "efficient-kan", "q1_ms": 1.1, "median_ms": 1.5, "q3_ms": 2},
    ]
    rows = table_rows(results)
    assert len(rows) == 6
    assert rows[0] == ("8", "64", "**1.000**", "2.500", "N/A", "N/A")
    assert rows[1] == ("8", "256", "1.000", "1.500", "N/A", "N/A")
    write_table(results, {"cpu": "test CPU", "gpu": None}, tmp_path)
    assert "**1.000**" in (tmp_path / "kan_table.md").read_text()
    assert r"\textbf{1.000}" in (tmp_path / "kan_table.tex").read_text()
