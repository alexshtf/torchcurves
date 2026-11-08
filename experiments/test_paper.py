"""The two numerical contracts on which the timings depend."""

# ruff: noqa: I001

from common import prepare

import torch
import numpy as np
from curves import check_outputs, evaluators
from kan import check_match


def test_curve_references() -> None:
    prepare()
    rng = np.random.default_rng(2026)
    with torch.inference_mode():
        for family in ("Legendre", "B-spline"):
            for count in (1, 32):
                candidate, reference, _ = evaluators(family, 32, count, rng)
                check_outputs(candidate, reference, 32, count)


def test_matched_kan() -> None:
    prepare()
    for coefficients in (8, 32):
        for dtype in (torch.float32, torch.float64):
            check_match(coefficients, "cpu", dtype)
