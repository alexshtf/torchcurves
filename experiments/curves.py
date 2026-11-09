"""CPU evaluation of identical vector-valued curves."""

from functools import partial
from itertools import product

from common import SEED

# isort: split
import numpy as np
import torch
from scipy.interpolate import BSpline
from torch.utils.benchmark import Timer

from torchcurves.functional import bspline_curves, legendre_curves, uniform_augmented_knots

BATCHES = (1, 256, 2048)
COUNTS = (1, 16, 64)
DIMENSIONS = (1, 16, 64)
COEFFICIENTS = (4, 16, 64)


def spline_reference(x, splines):
    return np.stack([spline(x[:, i]) for i, spline in enumerate(splines)], axis=1)


def evaluators(family: str, batch: int, count: int, dimension: int, coefficients: int, rng):
    x_np = rng.uniform(-1, 1, (batch, count))
    x = torch.from_numpy(x_np)
    if family == "Legendre":
        coefficients_np = rng.standard_normal((coefficients, count, dimension))
        return (
            partial(legendre_curves, x, torch.from_numpy(coefficients_np)),
            partial(np.polynomial.legendre.legval, x_np[..., None], coefficients_np, tensor=False),
            "NumPy",
        )

    coefficients_np = rng.standard_normal((count, coefficients, dimension))
    knots = uniform_augmented_knots(coefficients, 3, dtype=torch.float64)
    splines = [BSpline(knots.numpy(), curve_coefficients, 3) for curve_coefficients in coefficients_np]
    return (
        partial(bspline_curves, x, torch.from_numpy(coefficients_np), knots, 3),
        partial(spline_reference, x_np, splines),
        "SciPy",
    )


def run() -> list[tuple[dict, list[float]]]:
    measurements = []
    rng = np.random.default_rng(SEED)

    with torch.inference_mode():
        for family, count, dimension, batch, coefficients in product(
            ("Legendre", "B-spline"), COUNTS, DIMENSIONS, BATCHES, COEFFICIENTS
        ):
            candidate, reference, other = evaluators(family, batch, count, dimension, coefficients, rng)

            for label, evaluator in (("TorchCurves", candidate), (other, reference)):
                timer = Timer(stmt="evaluate()", globals={"evaluate": evaluator}, num_threads=1)
                times = [timer.blocked_autorange(min_run_time=0.5).median * 1000 for _ in range(5)]
                config = {
                    "family": family,
                    "batch": batch,
                    "curves": count,
                    "outputs": dimension,
                    "coefficients": coefficients,
                    "degree": coefficients - 1 if family == "Legendre" else 3,
                    "implementation": label,
                    "device": "cpu",
                    "dtype": "float64",
                    "threads": 1,
                    "seed": SEED,
                }
                measurements.append((config, times))
    return measurements
