import unittest

import numpy as np
import numpy.polynomial.legendre as np_leg
import pytest
import torch
import torch.nn as nn

from torchcurves.legendre import LegendreCurve, LegendreCurveFunction


class TestLegendreFunctionHelpers(unittest.TestCase):
    def setUp(self):
        self.default_dtype = torch.float64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_eval_legendre_polys_and_derivs_against_numpy(self):
        # x_np_scalar: (N_samples,)
        x_np_scalar = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0])
        num_curves_m = 2  # Test with M > 1

        # Create batched x: (N_samples, M_curves)
        # For simplicity, let each curve be evaluated at the same N scalar points
        x_np_batched = np.tile(x_np_scalar[:, np.newaxis], (1, num_curves_m))
        x_torch_batched = torch.tensor(x_np_batched, dtype=self.default_dtype, device=self.device)
        max_degree = 5

        # Test _eval_legendre_polys
        # Input x: (N, M), Output polys: (N, M, degree+1)
        torch_polys_batched = LegendreCurveFunction._eval_legendre_polys(x_torch_batched, max_degree)
        self.assertEqual(torch_polys_batched.shape, (x_np_scalar.shape[0], num_curves_m, max_degree + 1))

        for m_idx in range(num_curves_m):  # Check each curve in the batch
            for k in range(max_degree + 1):
                numpy_coefs = [0.0] * k + [1.0]  # P_k
                expected_val_scalar = np_leg.legval(x_np_scalar, numpy_coefs)
                torch.testing.assert_close(
                    torch_polys_batched[:, m_idx, k].cpu(),
                    torch.tensor(expected_val_scalar, dtype=self.default_dtype),
                    msg=f"Polynomial P_{k}(x) mismatch for curve {m_idx}",
                )

        # Test _eval_legendre_derivs
        # Input polys: (N,M,deg+1), Output derivs: (N,M,deg+1)
        torch_derivs_batched = LegendreCurveFunction._eval_legendre_derivs(max_degree, torch_polys_batched)
        self.assertEqual(torch_derivs_batched.shape, (x_np_scalar.shape[0], num_curves_m, max_degree + 1))

        for m_idx in range(num_curves_m):  # Check each curve
            for k in range(max_degree + 1):
                numpy_coefs = [0.0] * k + [1.0]
                numpy_coefs_deriv = np_leg.legder(numpy_coefs, m=1)  # P'_k
                expected_deriv_scalar = np_leg.legval(x_np_scalar, numpy_coefs_deriv)
                torch.testing.assert_close(
                    torch_derivs_batched[:, m_idx, k].cpu(),
                    torch.tensor(expected_deriv_scalar, dtype=self.default_dtype),
                    msg=f"Derivative P'_{k}(x) mismatch for curve {m_idx}",
                )


class TestLegendreCurveFunction(unittest.TestCase):
    def setUp(self):
        self.default_dtype = torch.float64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_constant_function_degree0(self):
        degree = 0
        num_curves_m = 1
        # control_points: (M, C, D) -> (1, 1, 1)
        control_points = torch.tensor([[[2.5]]], dtype=self.default_dtype, device=self.device)

        x_inputs_scalar = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=self.default_dtype, device=self.device)

        for x_val_scalar_item in x_inputs_scalar:
            # x: (N, M) -> (1, 1)
            x = x_val_scalar_item.view(1, num_curves_m)
            # points: (N, M, D) -> (1, 1, 1)
            points = LegendreCurveFunction.apply(x, control_points, degree)
            self.assertAlmostEqual(points.squeeze().item(), control_points.squeeze().item(), places=5)

            x_gc = x.clone().requires_grad_(True)
            cp_gc = control_points.clone()  # Not requiring grad for CP in this specific check

            self.assertTrue(
                torch.autograd.gradcheck(
                    lambda val_x: LegendreCurveFunction.apply(val_x, cp_gc, degree).sum(),  # noqa: B023
                    x_gc,
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-3,
                    nondet_tol=1e-7,
                )
            )
            points_gc = LegendreCurveFunction.apply(x_gc, cp_gc, degree)
            points_gc.sum().backward()
            self.assertAlmostEqual(x_gc.grad.squeeze().item(), 0.0, places=5, msg=f"Grad_x non-zero for x={x.item()}")

    def test_constant_function_all_cps_same_value(self):  # Renamed for clarity
        degree = 2  # P0, P1, P2
        num_curves_m = 1
        const_val = 5.0
        # C(x) = CP0*P0 + CP1*P1 + CP2*P2. If C(x) = const_val, then CP0=const_val, CP1=0, CP2=0.
        # control_points: (M,C,D) -> (1, 3, 1)
        control_points = torch.tensor([[[const_val], [0.0], [0.0]]], dtype=self.default_dtype, device=self.device)

        x_inputs_scalar = torch.tensor([-0.8, 0.0, 0.5, 0.9], dtype=self.default_dtype, device=self.device)
        x_inputs = x_inputs_scalar.unsqueeze(1)  # (N,1) for M=1 curve

        # points: (N,M,D) -> (N,1,1)
        points = LegendreCurveFunction.apply(x_inputs, control_points, degree)
        expected_points = torch.full(
            (x_inputs.shape[0], num_curves_m, 1), const_val, dtype=self.default_dtype, device=self.device
        )
        torch.testing.assert_close(points, expected_points, atol=1e-5, rtol=1e-5)

        x_gc = x_inputs.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)
        output = LegendreCurveFunction.apply(x_gc, cp_gc, degree)
        output.sum().backward()

        torch.testing.assert_close(x_gc.grad, torch.zeros_like(x_gc), atol=1e-5, rtol=1e-5)
        # grad_cp[m,c,d]. For CP0 (c=0), grad is sum(1*P0(x)*grad_out_points). grad_out_points=1. P0(x)=1.
        # So grad_cp[0,0,0] = N_samples.
        self.assertAlmostEqual(cp_gc.grad[0, 0, 0].item(), x_inputs.shape[0], places=5)
        self.assertAlmostEqual(
            cp_gc.grad[0, 1, 0].sum().item(),
            x_inputs_scalar.sum().item() * x_inputs.shape[0] / x_inputs.shape[0],
            places=4,
        )  # Approx sum(x_i)
        self.assertTrue(cp_gc.grad.sum() > 0)  # Ensure some gradients exist

    def test_linear_function(self):
        degree = 1  # C(x) = CP0*P0 + CP1*P1 = CP0 + CP1*x.
        # To get C(x)=x: CP0=0, CP1=1.
        # control_points: (M,C,D) -> (1,2,1)
        control_points = torch.tensor([[[0.0], [1.0]]], dtype=self.default_dtype, device=self.device)

        x_inputs_scalar = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=self.default_dtype, device=self.device)
        x_inputs = x_inputs_scalar.unsqueeze(1)  # (N,1)

        expected_points_scalar = x_inputs_scalar  # C(x) = x
        expected_points = expected_points_scalar.unsqueeze(1).unsqueeze(1)  # (N,1,1)

        points = LegendreCurveFunction.apply(x_inputs, control_points, degree)
        torch.testing.assert_close(points, expected_points, atol=1e-6, rtol=1e-5)

        x_gc = x_inputs.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda val_x: LegendreCurveFunction.apply(val_x, cp_gc.detach(), degree).sum(),
                x_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda val_cp: LegendreCurveFunction.apply(x_gc.detach(), val_cp, degree).sum(),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )

        output_an = LegendreCurveFunction.apply(x_gc, cp_gc.detach(), degree)
        output_an.sum().backward()  # grad_output is 1 for each point
        # dC/dx = CP1*P'_1(x) = 1*1 = 1.
        expected_grad_x = torch.ones_like(x_gc)
        torch.testing.assert_close(x_gc.grad, expected_grad_x, atol=1e-6, rtol=1e-5)

    def test_quadratic_function(self):
        degree = 2  # C(x) = CP0*P0 + CP1*P1 + CP2*P2 = CP0 + CP1*x + CP2*0.5*(3x^2-1)
        # To get C(x)=x^2: (1.5*CP2)x^2 + (CP1)x + (CP0 - 0.5*CP2) = x^2
        # CP2 = 2/3, CP1 = 0, CP0 = 1/3.
        # control_points: (M,C,D) -> (1,3,1)
        control_points = torch.tensor([[[1 / 3], [0.0], [2 / 3]]], dtype=self.default_dtype, device=self.device)

        x_inputs_scalar = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=self.default_dtype, device=self.device)
        x_inputs = x_inputs_scalar.unsqueeze(1)  # (N,1)

        expected_points_scalar = x_inputs_scalar.pow(2)  # C(x) = x^2
        expected_points = expected_points_scalar.unsqueeze(1).unsqueeze(1)  # (N,1,1)

        points = LegendreCurveFunction.apply(x_inputs, control_points, degree)
        torch.testing.assert_close(points, expected_points, atol=1e-6, rtol=1e-5)

        x_gc = x_inputs.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda val_x: LegendreCurveFunction.apply(val_x, cp_gc.detach(), degree).sum(),
                x_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,  # Increased atol for x^2
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda val_cp: LegendreCurveFunction.apply(x_gc.detach(), val_cp, degree).sum(),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )

        output_an = LegendreCurveFunction.apply(x_gc, cp_gc.detach(), degree)
        output_an.sum().backward()  # grad_output is 1
        # dC/dx = CP1*P'_1 + CP2*P'_2 = 0*1 + (2/3)*(3x) = 2x.
        expected_grad_x = 2 * x_gc.detach()
        torch.testing.assert_close(x_gc.grad, expected_grad_x, atol=1e-6, rtol=1e-5)

    def test_boundary_values_known_function(self):
        degree = 1  # C(x) = x. CP0=0, CP1=1.
        control_points = torch.tensor([[[0.0], [1.0]]], dtype=self.default_dtype, device=self.device)  # (1,2,1)

        x_start_scalar = torch.tensor([-1.0], dtype=self.default_dtype, device=self.device)
        x_end_scalar = torch.tensor([1.0], dtype=self.default_dtype, device=self.device)

        x_start = x_start_scalar.unsqueeze(1)  # (1,1)
        x_end = x_end_scalar.unsqueeze(1)  # (1,1)

        point_start = LegendreCurveFunction.apply(x_start, control_points, degree)  # (1,1,1)
        point_end = LegendreCurveFunction.apply(x_end, control_points, degree)  # (1,1,1)

        torch.testing.assert_close(
            point_start.squeeze(), torch.tensor(-1.0, dtype=self.default_dtype, device=self.device)
        )
        torch.testing.assert_close(point_end.squeeze(), torch.tensor(1.0, dtype=self.default_dtype, device=self.device))

    def test_multiple_curves_equivalence(self):
        num_curves_m = 3
        n_samples_n = 5
        dim_d = 2
        degree = 2  # C = 3 coefficients

        # control_points_batched: (M, C, D)
        control_points_batched = torch.randn(
            num_curves_m, degree + 1, dim_d, dtype=self.default_dtype, device=self.device
        )
        control_points_batched_clone_for_grad = control_points_batched.clone().requires_grad_(True)

        # x_inputs_batched: (N, M) in [-1, 1]
        x_inputs_batched = torch.rand(n_samples_n, num_curves_m, dtype=self.default_dtype, device=self.device) * 2 - 1
        x_inputs_batched_clone_for_grad = x_inputs_batched.clone().requires_grad_(True)

        # 1. Evaluate all curves together
        points_batched_eval = LegendreCurveFunction.apply(
            x_inputs_batched_clone_for_grad, control_points_batched_clone_for_grad, degree
        )  # (N, M, D)

        # 2. Evaluate each curve individually
        points_individual_list = []
        for i in range(num_curves_m):
            cp_single = control_points_batched[i : i + 1, :, :].clone()  # Shape (1, C, D)
            x_single = x_inputs_batched[:, i : i + 1].clone()  # Shape (N, 1)

            points_single = LegendreCurveFunction.apply(x_single, cp_single, degree)  # Output (N, 1, D)
            points_individual_list.append(points_single)

        points_stacked_eval = torch.cat(points_individual_list, dim=1)  # (N, M, D)
        torch.testing.assert_close(points_batched_eval.data, points_stacked_eval.data, atol=1e-6, rtol=1e-5)

        # Compare backward pass
        grad_output = torch.randn_like(points_batched_eval)

        points_batched_eval.backward(grad_output)
        grad_x_batched_actual = x_inputs_batched_clone_for_grad.grad.clone()
        grad_cp_batched_actual = control_points_batched_clone_for_grad.grad.clone()

        expected_grad_x_from_individuals = torch.zeros_like(x_inputs_batched)
        expected_grad_cp_from_individuals = torch.zeros_like(control_points_batched)

        for i in range(num_curves_m):
            cp_single_grad_target = control_points_batched[i : i + 1, :, :].detach().clone().requires_grad_(True)
            x_single_grad_target = x_inputs_batched[:, i : i + 1].detach().clone().requires_grad_(True)

            points_single_eval_for_grad = LegendreCurveFunction.apply(
                x_single_grad_target, cp_single_grad_target, degree
            )
            grad_output_single = grad_output[:, i : i + 1, :]  # (N, 1, D)
            points_single_eval_for_grad.backward(grad_output_single)

            expected_grad_x_from_individuals[:, i : i + 1] = x_single_grad_target.grad
            expected_grad_cp_from_individuals[i : i + 1, :, :] = cp_single_grad_target.grad

        torch.testing.assert_close(grad_x_batched_actual, expected_grad_x_from_individuals, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(grad_cp_batched_actual, expected_grad_cp_from_individuals, atol=1e-6, rtol=1e-5)


class TestLegendreCurveModule(unittest.TestCase):
    def setUp(self):
        self.default_dtype = torch.float64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_init(self):
        num_curves = 2
        dim = 3
        degree = 4
        module = LegendreCurve(num_curves, dim, degree).to(self.device).to(self.default_dtype)

        self.assertEqual(module.num_curves, num_curves)
        self.assertEqual(module.dim, dim)
        self.assertEqual(module.degree, degree)
        self.assertEqual(module.n_coefficients, degree + 1)
        self.assertIsInstance(module.coefficients, nn.Parameter)
        self.assertTrue(module.coefficients.requires_grad)
        self.assertEqual(module.coefficients.shape, (num_curves, degree + 1, dim))

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

    def test_forward_pass_shape_and_device(self):
        num_curves = 2
        dim = 3
        degree = 2
        n_samples = 10

        module = LegendreCurve(num_curves, dim, degree).to(self.device).to(self.default_dtype)

        # u: (N, M)
        u_input = torch.rand(n_samples, num_curves, device=self.device, dtype=self.default_dtype) * 2 - 1  # in [-1,1]

        points = module(u_input)  # Output (N, M, D)

        self.assertEqual(points.shape, (n_samples, num_curves, dim))
        self.assertEqual(points.device, self.device)
        self.assertEqual(points.dtype, self.default_dtype)

    def test_backward_pass_module(self):
        num_curves = 2
        dim = 2
        degree = 3
        n_samples = 5
        module = LegendreCurve(num_curves, dim, degree).to(self.device).to(self.default_dtype)

        u_input = torch.rand(n_samples, num_curves, device=self.device, dtype=self.default_dtype).requires_grad_(True)

        self.assertIsNone(module.coefficients.grad)

        points = module(u_input)  # (N,M,D)
        loss = points.sum()
        loss.backward()

        self.assertIsNotNone(module.coefficients.grad)
        self.assertEqual(module.coefficients.grad.shape, module.coefficients.shape)
        self.assertNotEqual(torch.sum(module.coefficients.grad**2).item(), 0.0)

        self.assertIsNotNone(u_input.grad)  # Check grad w.r.t. u as well
        self.assertEqual(u_input.grad.shape, u_input.shape)

    def test_gradcheck_module_full(self):  # More comprehensive gradcheck
        num_curves = 2
        dim = 1  # Simpler for gradcheck output interpretation
        degree = 2
        n_samples = 3

        module = LegendreCurve(num_curves, dim, degree).to(self.device).to(self.default_dtype)

        # u_gc: (N,M)
        u_gc = (torch.rand(n_samples, num_curves, device=self.device, dtype=self.default_dtype) * 2 - 1).requires_grad_(
            True
        )

        # We gradcheck the module's forward directly.
        # The inputs to the lambda for gradcheck must match the inputs to module.forward() that require grad.
        # If module parameters also need checking, they'd be separate inputs to lambda.
        # Here, we check grads w.r.t. u_gc and module.coefficients (implicitly through module call).

        # Check grad w.r.t u
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda u_in: module(u_in).sum(),  # Sum for scalar output
                u_gc,
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

        # Check grad w.r.t. coefficients (module parameters)
        # To do this, we need to make coefficients an input to a lambda
        # or use gradgradcheck for higher order.
        # A simpler way is to check if grads are populated, which test_backward_pass_module does.
        # For a full gradcheck on parameters, one might do:

        u_fixed = u_gc.detach().clone()

        # Create a temporary function that takes coeffs as input
        def temp_func_for_coeffs_grad(coeffs_param):
            # Temporarily assign new coeffs to the module (not ideal for nn.Module)
            # A better way is to use the functional form LegendreCurveFunction.apply directly
            # with detached u and the coeffs_param.
            original_coeffs = module.coefficients.data.clone()
            module.coefficients.data = coeffs_param.data  # Risky, but for local test
            output = module(u_fixed).sum()
            module.coefficients.data = original_coeffs  # Restore
            return output

        # This approach of modifying module state in lambda is not robust.
        # Better to test LegendreCurveFunction.apply directly for parameter grads:

        coeffs_for_apply_gc = module.coefficients.clone().requires_grad_(True)
        u_for_apply_fixed = u_gc.detach().clone()
        # Normalize u_for_apply_fixed as the module's forward would
        u_norm_for_apply = module.normalize_fn(u_for_apply_fixed, module.normalization_scale, out_min=-1.0, out_max=1.0)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda c_in: LegendreCurveFunction.apply(u_norm_for_apply, c_in, module.degree).sum(),
                coeffs_for_apply_gc,
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

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

        u_cuda = torch.rand(5, num_curves, device="cuda", dtype=self.default_dtype) * 2 - 1
        points = module_cuda(u_cuda)

        self.assertEqual(points.device.type, "cuda")
        self.assertEqual(points.shape, (5, num_curves, dim))

        loss = points.sum()
        loss.backward()
        self.assertIsNotNone(module_cuda.coefficients.grad)
        self.assertEqual(module_cuda.coefficients.grad.device.type, "cuda")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
