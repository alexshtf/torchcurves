import unittest

import numpy as np
import numpy.polynomial.legendre as np_leg
import torch

from torchcurves import LegendreCurveFunction


class TestLegendreFunctionHelpers(unittest.TestCase):
    def setUp(self):
        self.default_dtype = torch.float64  # For gradcheck accuracy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_eval_legendre_polys_and_derivs_against_numpy(self):
        x_np = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0])
        x_torch = torch.tensor(x_np, dtype=self.default_dtype, device=self.device)
        max_degree = 5

        # Test _eval_legendre_polys
        torch_polys = LegendreCurveFunction._eval_legendre_polys(x_torch, max_degree)
        self.assertEqual(torch_polys.shape, (x_np.shape[0], max_degree + 1))

        for k in range(max_degree + 1):
            numpy_coefs = [0.0] * k + [1.0]
            expected_val = np_leg.legval(x_np, numpy_coefs)
            torch.testing.assert_close(
                torch_polys[:, k].cpu(),
                torch.tensor(expected_val, dtype=self.default_dtype),
                msg=f"Polynomial P_{k}(x) mismatch",
            )

        # Test _eval_legendre_derivs (which uses precomputed polys)
        torch_derivs = LegendreCurveFunction._eval_legendre_derivs(max_degree, torch_polys)
        self.assertEqual(torch_derivs.shape, (x_np.shape[0], max_degree + 1))

        for k in range(max_degree + 1):
            numpy_coefs = [0.0] * k + [1.0]
            # Derivative of P_k(x)
            numpy_coefs_deriv = np_leg.legder(numpy_coefs, m=1)
            expected_deriv = np_leg.legval(x_np, numpy_coefs_deriv)
            torch.testing.assert_close(
                torch_derivs[:, k].cpu(),
                torch.tensor(expected_deriv, dtype=self.default_dtype),
                msg=f"Derivative P'_{k}(x) mismatch",
            )


class TestLegendreCurveFunction(unittest.TestCase):
    def setUp(self):
        self.default_dtype = torch.float64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_constant_function_degree0(self):
        degree = 0
        # C(x) = CP0 * P0(x). If C(x) = 2.5, then CP0 = 2.5 since P0(x)=1.
        control_points = torch.tensor([[2.5]], dtype=self.default_dtype, device=self.device)
        x_inputs = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=self.default_dtype, device=self.device)

        for x_val_scalar in x_inputs:
            x = x_val_scalar.unsqueeze(0)  # Batch of 1
            points = LegendreCurveFunction.apply(x, control_points, degree)
            self.assertAlmostEqual(points.item(), control_points[0, 0].item(), places=5)

            x_gc = x.clone().requires_grad_(True)
            cp_gc = control_points.clone()
            self.assertTrue(
                torch.autograd.gradcheck(
                    lambda val_x: LegendreCurveFunction.apply(val_x, cp_gc, degree),  # noqa: B023
                    x_gc,
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-3,
                    nondet_tol=1e-7,
                )
            )
            points_gc = LegendreCurveFunction.apply(x_gc, cp_gc, degree)
            points_gc.sum().backward()
            self.assertAlmostEqual(x_gc.grad.item(), 0.0, places=5, msg=f"Grad_x non-zero for x={x.item()}")

    def test_constant_function(self):
        degree = 2  # P0, P1, P2
        const_val = 5.0
        # C(x) = CP0*P0 + CP1*P1 + CP2*P2. If C(x) = const_val,
        # then CP0 = const_val, CP1 = 0, CP2 = 0.
        control_points = torch.tensor([[const_val], [0.0], [0.0]], dtype=self.default_dtype, device=self.device)
        x_inputs = torch.tensor([-0.8, 0.0, 0.5, 0.9], dtype=self.default_dtype, device=self.device)

        points = LegendreCurveFunction.apply(x_inputs, control_points, degree)
        expected_points = torch.full((x_inputs.shape[0], 1), const_val, dtype=self.default_dtype, device=self.device)
        torch.testing.assert_close(points, expected_points, atol=1e-5, rtol=1e-5)

        x_gc = x_inputs.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)
        output = LegendreCurveFunction.apply(x_gc, cp_gc, degree)
        output.sum().backward()
        torch.testing.assert_close(x_gc.grad, torch.zeros_like(x_gc), atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(cp_gc.grad[0].item(), x_inputs.shape[0], places=5)  # Sum(1*P0(x))

    def test_linear_function(self):
        # C(x) = x.
        # C(x) = CP0*P0(x) + CP1*P1(x) = CP0 + CP1*x.
        # So, CP0 = 0, CP1 = 1.
        degree = 1
        control_points = torch.tensor([[0.0], [1.0]], dtype=self.default_dtype, device=self.device)
        x_inputs = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=self.default_dtype, device=self.device)
        expected_points = x_inputs.unsqueeze(1)  # C(x) = x

        points = LegendreCurveFunction.apply(x_inputs, control_points, degree)
        torch.testing.assert_close(points, expected_points, atol=1e-6, rtol=1e-5)

        x_gc = x_inputs.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda val_x: LegendreCurveFunction.apply(val_x, cp_gc.detach(), degree),
                x_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda val_cp: LegendreCurveFunction.apply(x_gc.detach(), val_cp, degree),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )

        output_an = LegendreCurveFunction.apply(x_gc, cp_gc.detach(), degree)
        output_an.sum().backward()  # grad_output is 1
        # dC/dx = CP1*P'1(x) = 1*1 = 1.
        expected_grad_x = torch.ones_like(x_gc)
        torch.testing.assert_close(x_gc.grad, expected_grad_x, atol=1e-6, rtol=1e-5)

    def test_quadratic_function(self):
        # C(x) = x^2.
        # C(x) = CP0*P0 + CP1*P1 + CP2*P2 = CP0 + CP1*x + CP2*0.5*(3x^2-1)
        #      = (1.5*CP2)x^2 + (CP1)x + (CP0 - 0.5*CP2)
        # Equating coefficients: 1.5*CP2 = 1 => CP2 = 2/3. CP1 = 0. CP0 - 0.5*CP2 = 0 => CP0 = 1/3.
        degree = 2
        control_points = torch.tensor([[1 / 3], [0.0], [2 / 3]], dtype=self.default_dtype, device=self.device)
        x_inputs = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=self.default_dtype, device=self.device)
        expected_points = x_inputs.pow(2).unsqueeze(1)  # C(x) = x^2

        points = LegendreCurveFunction.apply(x_inputs, control_points, degree)
        torch.testing.assert_close(points, expected_points, atol=1e-6, rtol=1e-5)

        x_gc = x_inputs.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda val_x: LegendreCurveFunction.apply(val_x, cp_gc.detach(), degree),
                x_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda val_cp: LegendreCurveFunction.apply(x_gc.detach(), val_cp, degree),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )

        output_an = LegendreCurveFunction.apply(x_gc, cp_gc.detach(), degree)
        output_an.sum().backward()  # grad_output is 1
        # dC/dx = CP1*P'1 + CP2*P'2 = 0*1 + (2/3)*(3x) = 2x.
        expected_grad_x = 2 * x_gc.detach()
        torch.testing.assert_close(x_gc.grad, expected_grad_x, atol=1e-6, rtol=1e-5)

    def test_boundary_values_known_function(self):
        # Using C(x) = x. CP0=0, CP1=1
        degree = 1
        control_points = torch.tensor([[0.0], [1.0]], dtype=self.default_dtype, device=self.device)
        x_start = torch.tensor([-1.0], dtype=self.default_dtype, device=self.device)
        x_end = torch.tensor([1.0], dtype=self.default_dtype, device=self.device)

        point_start = LegendreCurveFunction.apply(x_start, control_points, degree)
        point_end = LegendreCurveFunction.apply(x_end, control_points, degree)

        torch.testing.assert_close(point_start, torch.tensor([[-1.0]], dtype=self.default_dtype, device=self.device))
        torch.testing.assert_close(point_end, torch.tensor([[1.0]], dtype=self.default_dtype, device=self.device))


if __name__ == "__main__":
    unittest.main()
