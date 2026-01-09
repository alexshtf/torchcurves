import unittest

import numpy as np
import pytest
import torch
import torch.nn as nn

from torchcurves import LegendreCurve
from torchcurves.functional import legendre_curves


@pytest.mark.parametrize("num_curves", [1, 2, 5])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("n_samples", [1, 10, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_legendre_curves(num_curves, dim, degree, n_samples, dtype):
    torch.random.manual_seed(42)  # For reproducibility
    coefs = torch.randn(1 + degree, num_curves, dim, dtype=dtype)
    x = 2 * torch.rand(n_samples, num_curves, dtype=dtype)
    torch_eval = legendre_curves(x, coefs)
    for ci in range(num_curves):
        for mi in range(dim):
            coef_np = coefs[:, ci, mi].numpy()
            x_np = x[:, ci].numpy()
            np_vals = np.polynomial.legendre.legval(x_np, coef_np)
            torch_vals = torch_eval[:, ci, mi].numpy()
            np.testing.assert_allclose(
                np_vals,
                torch_vals,
                rtol=1e-3 if dtype == torch.float32 else 1e-10,
                err_msg=f"Mismatch for curve {ci}, dimension {mi} with degree {degree} and dtype {dtype}",
            )


def test_legendre_curves_checkpoint_segments():
    torch.random.manual_seed(0)
    coefs = torch.randn(5, 2, 3, dtype=torch.float64, requires_grad=True)
    x = (2 * torch.rand(4, 2, dtype=torch.float64) - 1).requires_grad_(True)

    out_base = legendre_curves(x, coefs).detach()
    out_ckpt = legendre_curves(x, coefs, checkpoint_segments=2)
    torch.testing.assert_close(out_ckpt.detach(), out_base, rtol=1e-10, atol=1e-12)

    out_ckpt.sum().backward()
    assert x.grad is not None
    assert coefs.grad is not None


def test_legendre_curves_checkpoint_segments_invalid():
    coefs = torch.randn(2, 1, 1, dtype=torch.float64)
    x = torch.rand(2, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        legendre_curves(x, coefs, checkpoint_segments=0)


class TestLegendreCurveModule(unittest.TestCase):
    def setUp(self):
        self.default_dtype = torch.float64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_init(self):
        num_curves = 2
        dim = 3
        degree = 4
        module = (
            LegendreCurve(num_curves, dim, degree)
            .to(self.device)
            .to(self.default_dtype)
        )

        self.assertEqual(module.num_curves, num_curves)
        self.assertEqual(module.dim, dim)
        self.assertEqual(module.degree, degree)
        self.assertEqual(module.n_coefficients, degree + 1)
        self.assertIsInstance(module.coefficients, nn.Parameter)
        self.assertTrue(module.coefficients.requires_grad)
        self.assertEqual(module.coefficients.shape, (degree + 1, num_curves, dim))

    def test_init_errors(self):
        with self.assertRaises(ValueError):
            LegendreCurve(num_curves=0, dim=1, degree=1)  # num_curves <= 0
        with self.assertRaises(ValueError):
            LegendreCurve(num_curves=1, dim=0, degree=1)  # dim <= 0
        with self.assertRaises(ValueError):
            LegendreCurve(num_curves=1, dim=1, degree=-1)  # degree < 0
        with self.assertRaises(ValueError):  # Unknown normalization
            LegendreCurve(num_curves=1, dim=1, degree=1, normalize_fn="unknown_norm")
        with self.assertRaises(ValueError):  # Scale <=0
            LegendreCurve(num_curves=1, dim=1, degree=1, normalization_scale=0)
        with self.assertRaises(ValueError):
            LegendreCurve(num_curves=1, dim=1, degree=1, checkpoint_segments=0)

    def test_forward_pass_shape_and_device(self):
        num_curves = 2
        dim = 3
        degree = 2
        n_samples = 10

        module = (
            LegendreCurve(num_curves, dim, degree)
            .to(self.device)
            .to(self.default_dtype)
        )

        # u: (N, M)
        u_input = (
            torch.rand(
                n_samples, num_curves, device=self.device, dtype=self.default_dtype
            )
            * 2
            - 1
        )  # in [-1,1]

        points = module(u_input)  # Output (N, M, D)

        self.assertEqual(points.shape, (n_samples, num_curves, dim))
        self.assertEqual(points.device, self.device)
        self.assertEqual(points.dtype, self.default_dtype)

    def test_forward_with_checkpoint_segments(self):
        num_curves = 2
        dim = 2
        degree = 3
        n_samples = 6

        module = (
            LegendreCurve(num_curves, dim, degree, checkpoint_segments=2)
            .to(self.device)
            .to(self.default_dtype)
        )
        u_input = torch.rand(
            n_samples, num_curves, device=self.device, dtype=self.default_dtype
        )
        points = module(u_input)

        self.assertEqual(points.shape, (n_samples, num_curves, dim))
        self.assertEqual(points.device, self.device)
        self.assertEqual(points.dtype, self.default_dtype)

    def test_backward_pass_module(self):
        num_curves = 2
        dim = 2
        degree = 3
        n_samples = 5
        module = (
            LegendreCurve(num_curves, dim, degree)
            .to(self.device)
            .to(self.default_dtype)
        )

        u_input = torch.rand(
            n_samples, num_curves, device=self.device, dtype=self.default_dtype
        ).requires_grad_(True)

        self.assertIsNone(module.coefficients.grad)

        points = module(u_input)  # (N,M,D)
        loss = points.sum()
        loss.backward()

        self.assertIsNotNone(module.coefficients.grad)
        self.assertEqual(module.coefficients.grad.shape, module.coefficients.shape)
        self.assertNotEqual(torch.sum(module.coefficients.grad**2).item(), 0.0)

        self.assertIsNotNone(u_input.grad)  # Check grad w.r.t. u as well
        self.assertEqual(u_input.grad.shape, u_input.shape)

    def test_backward_pass_module_checkpoint(self):
        num_curves = 2
        dim = 2
        degree = 3
        n_samples = 5
        module = (
            LegendreCurve(num_curves, dim, degree, checkpoint_segments=2)
            .to(self.device)
            .to(self.default_dtype)
        )

        u_input = torch.rand(
            n_samples, num_curves, device=self.device, dtype=self.default_dtype
        ).requires_grad_(True)

        points = module(u_input)
        loss = points.sum()
        loss.backward()

        self.assertIsNotNone(module.coefficients.grad)
        self.assertIsNotNone(u_input.grad)

    def test_device_movement_module(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device movement test.")

        num_curves = 2
        dim = 1
        degree = 2
        module_cpu = LegendreCurve(num_curves, dim, degree)

        self.assertEqual(module_cpu.coefficients.device.type, "cpu")

        module_cuda = module_cpu.to("cuda").to(self.default_dtype)
        self.assertEqual(module_cuda.coefficients.device.type, "cuda")

        u_cuda = (
            torch.rand(5, num_curves, device="cuda", dtype=self.default_dtype) * 2 - 1
        )
        points = module_cuda(u_cuda)

        self.assertEqual(points.device.type, "cuda")
        self.assertEqual(points.shape, (5, num_curves, dim))

        loss = points.sum()
        loss.backward()
        self.assertIsNotNone(module_cuda.coefficients.grad)
        self.assertEqual(module_cuda.coefficients.grad.device.type, "cuda")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
