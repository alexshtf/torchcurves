import unittest

import pytest
import torch
import torch.nn as nn

from torchcurves import BSplineCurve
from torchcurves.functional import bspline_curves


class TestBSplineFunction(unittest.TestCase):
    def setUp(self):
        self.default_dtype = torch.float64  # For gradcheck
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu") # Force CPU for easier debugging if needed
        # print(f"Using device: {self.device}")

    @staticmethod
    def generate_clamped_knot_vector(
        n_control_points: int, degree: int, device="cpu", dtype=torch.float32
    ) -> torch.Tensor:
        """Generate a clamped knot vector in [-1, 1]."""
        if n_control_points <= degree:
            raise ValueError("Number of control points must be greater than degree.")

        # Total number of knots m = n_control_points + degree + 1.
        # Correct clamping: first p+1 knots are k_min, last p+1 knots are k_max
        k_min, k_max = -1.0, 1.0

        head_knots = torch.full((degree + 1,), k_min, dtype=dtype, device=device)
        tail_knots = torch.full((degree + 1,), k_max, dtype=dtype, device=device)

        num_internal_knots = n_control_points - degree - 1
        if num_internal_knots < 0:
            raise ValueError("Not enough control points for the given degree to form internal knots.")

        if num_internal_knots == 0:
            internal_knots = torch.empty(0, dtype=dtype, device=device)
        else:
            internal_knots = torch.linspace(k_min, k_max, num_internal_knots + 2, dtype=dtype, device=device)[1:-1]

        return torch.cat([head_knots, internal_knots, tail_knots])

    def test_constant_function_degree0(self):
        degree = 0
        # control_points: (M, C, D) -> (1 curve, 1 CP, 1 Dim)
        control_points = torch.tensor([[[2.5]]], dtype=self.default_dtype, device=self.device)
        n_cp_c = control_points.shape[1]
        knots = self.generate_clamped_knot_vector(n_cp_c, degree, device=self.device, dtype=self.default_dtype)
        self.assertEqual(knots.shape[0], n_cp_c + degree + 1)

        u_values_scalar = torch.tensor([0.0, 0.5, 0.99], dtype=self.default_dtype, device=self.device)

        for u_val_scalar_item in u_values_scalar:
            # u: (N, M) -> (1 sample, 1 curve)
            u = u_val_scalar_item.view(1, 1)
            # points: (N, M, D) -> (1, 1, 1)
            points = bspline_curves(u, control_points, knots, degree)
            self.assertAlmostEqual(
                points.squeeze().item(), control_points.squeeze().item(), places=5, msg=f"Failed for u={u.item()}"
            )

            u_gc = u.clone().requires_grad_(True)
            cp_gc = control_points.clone()

            # Output is (1,1,1), gradcheck handles this.
            self.assertTrue(
                torch.autograd.gradcheck(
                    lambda x: bspline_curves(x, cp_gc, knots, degree),  # noqa: B023
                    u_gc,
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-3,
                    nondet_tol=1e-7,
                )
            )

            points_gc = bspline_curves(u_gc, cp_gc, knots, degree)
            points_gc.sum().backward()  # .sum() for scalar loss
            self.assertAlmostEqual(u_gc.grad.squeeze().item(), 0.0, places=5, msg=f"Grad_u non-zero for u={u.item()}")

    def test_constant_function_all_cps_same(self):
        degree = 2
        n_cp_c = 4
        const_val = 5.0
        # control_points: (M,C,D) -> (1, 4, 1)
        control_points = torch.full((1, n_cp_c, 1), const_val, dtype=self.default_dtype, device=self.device)
        knots = self.generate_clamped_knot_vector(n_cp_c, degree, device=self.device, dtype=self.default_dtype)

        # u_scalar: (N,)
        u_scalar = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=self.default_dtype, device=self.device)
        # u: (N, M) -> (N, 1)
        u = u_scalar.unsqueeze(1)

        # points: (N, M, D) -> (N, 1, 1)
        points = bspline_curves(u, control_points, knots, degree)
        expected_points = torch.full((u.shape[0], 1, 1), const_val, dtype=self.default_dtype, device=self.device)
        torch.testing.assert_close(points, expected_points, atol=1e-5, rtol=1e-5)

        u_gc = u.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)

        output = bspline_curves(u_gc, cp_gc, knots, degree)
        output.sum().backward()

        # u_gc.grad: (N,1)
        torch.testing.assert_close(u_gc.grad, torch.zeros_like(u_gc), atol=1e-5, rtol=1e-5)
        # cp_gc.grad: (1, C, D). Sum of basis functions is 1.
        self.assertAlmostEqual(cp_gc.grad.sum().item(), u.shape[0], places=5)

    def test_linear_function_degree1(self):
        degree = 1
        # control_points: (M,C,D) -> (1, 2, 1)
        control_points = torch.tensor([[[0.0], [1.0]]], dtype=self.default_dtype, device=self.device)
        n_cp_c = control_points.shape[1]
        knots = self.generate_clamped_knot_vector(n_cp_c, degree, device=self.device, dtype=self.default_dtype)

        u_scalar = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=self.default_dtype, device=self.device)
        # u: (N,M) -> (N,1)
        u = u_scalar.unsqueeze(1)
        # For knots [-1,-1,1,1] and u in [-1,1], C(u) = ( (1-u)/2 * P0 + (1+u)/2 * P1 ) if knots are normalized to
        # [0,1] internally
        # If knots are [-1,-1,1,1] and u is directly used, then for u in [-1,1], it's linear interpolation.
        # The current BSplineFunction expects u to be in the knot range.
        # With knots [-1,-1,1,1], P0=[-1], P1=[1], then C(u)=u.
        # Here P0=[0], P1=[1]. Knots are [-1,-1,1,1].
        # N01(u) = (knots[1+1]-u)/(knots[1+1]-knots[1]) = (1-u)/(1-(-1)) = (1-u)/2 for u in [-1,1)
        # N11(u) = (u-knots[1])/(knots[1+1]-knots[1]) = (u-(-1))/(1-(-1)) = (u+1)/2 for u in [-1,1)
        # C(u) = (1-u)/2 * 0 + (u+1)/2 * 1 = (u+1)/2
        # To get C(u)=u, we need u_norm = (u+1)/2. If input u is already in [-1,1], then we expect (u+1)/2.
        # Let's adjust CPs or expected points. If CPs are [0],[1] and knots are [-1,-1,1,1]
        # C(-1) = P0 = 0. C(1) = P1 = 1. (u+1)/2.
        # If we want C(u) = u for u in [-1,1], then P0=-1, P1=1.
        # Let's keep P0=0, P1=1. Then expected is (u_scalar+1)/2
        expected_points_scalar = (u_scalar + 1.0) / 2.0
        expected_points = expected_points_scalar.unsqueeze(1).unsqueeze(1)  # (N,1,1)

        points = bspline_curves(u, control_points, knots, degree)
        torch.testing.assert_close(points, expected_points, atol=1e-6, rtol=1e-5)

        u_gc = u.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: bspline_curves(x, cp_gc.detach(), knots, degree).sum(),
                u_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: bspline_curves(u_gc.detach(), x, knots, degree).sum(),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

        output_an = bspline_curves(u_gc, cp_gc.detach(), knots, degree)
        output_an.sum().backward()
        # C'(u) = 0.5
        expected_grad_u = torch.full_like(u_gc, 0.5)
        torch.testing.assert_close(u_gc.grad, expected_grad_u, atol=1e-6, rtol=1e-5)

    def test_parabola_degree2(self):
        degree = 2
        # Knots [-1,-1,-1, 1,1,1]. u in [-1,1].
        # N02=(1-u_norm)^2, N12=2*u_norm(1-u_norm), N22=u_norm^2 where u_norm = (u+1)/2
        # C(u) = N02*P0 + N12*P1 + N22*P2.
        # To get C(u) = u_norm^2 = ((u+1)/2)^2: P0=0, P1=0, P2=1.
        # control_points: (M,C,D) -> (1,3,1)
        control_points = torch.tensor([[[0.0], [0.0], [1.0]]], dtype=self.default_dtype, device=self.device)
        n_cp_c = control_points.shape[1]
        knots = self.generate_clamped_knot_vector(n_cp_c, degree, device=self.device, dtype=self.default_dtype)

        u_scalar = torch.tensor([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0], dtype=self.default_dtype, device=self.device)
        u = u_scalar.unsqueeze(1)  # (N,1)

        u_norm_scalar = (u_scalar + 1.0) / 2.0
        expected_points_scalar = u_norm_scalar.pow(2)
        expected_points = expected_points_scalar.unsqueeze(1).unsqueeze(1)  # (N,1,1)

        points = bspline_curves(u, control_points, knots, degree)
        torch.testing.assert_close(points, expected_points, atol=1e-6, rtol=1e-5)

        u_gc = u.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_u: bspline_curves(x_u, cp_gc.detach(), knots, degree).sum(),
                u_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-4,  # Increased atol for parabola
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_cp: bspline_curves(u_gc.detach(), x_cp, knots, degree).sum(),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

        output_an = bspline_curves(u_gc, cp_gc.detach(), knots, degree)
        output_an.sum().backward()
        # C'(u) = d/du [((u+1)/2)^2] = 2 * ((u+1)/2) * (1/2) = (u+1)/2
        expected_grad_u_scalar = (u_gc.detach().squeeze(1) + 1.0) / 2.0
        expected_grad_u = expected_grad_u_scalar.unsqueeze(1)
        torch.testing.assert_close(u_gc.grad, expected_grad_u, atol=1e-6, rtol=1e-5)

    def test_boundary_values(self):
        degree = 3
        n_cp_c = 5
        # control_points: (M,C,D) -> (1,5,2)
        control_points = torch.randn(1, n_cp_c, 2, dtype=self.default_dtype, device=self.device)
        knots = self.generate_clamped_knot_vector(n_cp_c, degree, device=self.device, dtype=self.default_dtype)

        # u: (N,M) -> (1,1)
        u_start = torch.tensor([[-1.0]], dtype=self.default_dtype, device=self.device)  # Min knot value
        u_end = torch.tensor([[1.0]], dtype=self.default_dtype, device=self.device)  # Max knot value

        point_start = bspline_curves(u_start, control_points, knots, degree)  # (1,1,2)
        point_end = bspline_curves(u_end, control_points, knots, degree)  # (1,1,2)

        # control_points[:, 0, :] is (1,2). Need (1,1,2)
        torch.testing.assert_close(point_start, control_points[:, 0:1, :], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(point_end, control_points[:, -1:, :], atol=1e-6, rtol=1e-5)

    def test_multiple_dimensions(self):
        degree = 2
        # control_points: (M,C,D) -> (1,3,2)
        control_points_data = torch.tensor(
            [[[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]], dtype=self.default_dtype, device=self.device
        )
        n_cp_c = control_points_data.shape[1]
        knots = self.generate_clamped_knot_vector(n_cp_c, degree, device=self.device, dtype=self.default_dtype)

        u_scalar = torch.tensor([-1.0, 0.0, 1.0], dtype=self.default_dtype, device=self.device)
        u = u_scalar.unsqueeze(1)  # (N,1)

        # Expected points: (N,1,D)
        # u_norm = (u_scalar+1)/2 -> [0, 0.5, 1]
        # C(u_norm) = (1-u_norm)^2 P0 + 2u_norm(1-u_norm)P1 + u_norm^2 P2
        expected_points_calc = torch.empty((u_scalar.shape[0], 1, 2), dtype=self.default_dtype, device=self.device)
        P0, P1, P2 = control_points_data[0, 0], control_points_data[0, 1], control_points_data[0, 2]  # noqa: N806

        expected_points_calc[0, 0, :] = P0  # u_norm = 0
        expected_points_calc[1, 0, :] = 0.25 * P0 + 0.5 * P1 + 0.25 * P2  # u_norm = 0.5
        expected_points_calc[2, 0, :] = P2  # u_norm = 1

        points = bspline_curves(u, control_points_data, knots, degree)
        torch.testing.assert_close(points, expected_points_calc, atol=1e-6, rtol=1e-5)

        u_gc = u.clone().requires_grad_(True)
        cp_gc = control_points_data.clone().requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_u: bspline_curves(x_u, cp_gc.detach(), knots, degree).sum(),
                u_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_cp: bspline_curves(u_gc.detach(), x_cp, knots, degree).sum(),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

    def test_batch_processing_u_values_single_curve(self):  # Renamed for clarity
        degree = 1
        # control_points: (M,C,D) -> (1,2,2)
        control_points = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]], dtype=self.default_dtype, device=self.device)
        n_cp_c = control_points.shape[1]
        knots = self.generate_clamped_knot_vector(
            n_cp_c, degree, device=self.device, dtype=self.default_dtype
        )  # Knots [-1,-1,1,1]

        u_scalar_batch = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=self.default_dtype, device=self.device
        )  # Batch of N u-values
        u_batch = u_scalar_batch.unsqueeze(1)  # (N,1) for 1 curve

        # Expected points (N,1,D)
        # u_norm = (u_scalar_batch+1)/2
        # C(u_norm) = (1-u_norm)P0 + u_norm*P1
        expected_points_batch = torch.empty(
            (u_batch.shape[0], 1, control_points.shape[2]), dtype=self.default_dtype, device=self.device
        )
        P0, P1 = control_points[0, 0, :], control_points[0, 1, :]  # noqa: N806
        u_norm_vals = (u_scalar_batch + 1.0) / 2.0
        for i, u_n_val in enumerate(u_norm_vals):
            expected_points_batch[i, 0, :] = (1 - u_n_val) * P0 + u_n_val * P1

        points_batch = bspline_curves(u_batch, control_points, knots, degree)
        torch.testing.assert_close(points_batch, expected_points_batch, atol=1e-6, rtol=1e-5)
        self.assertEqual(points_batch.shape, (u_batch.shape[0], 1, control_points.shape[2]))

        u_gc_batch = u_batch.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_u: bspline_curves(x_u, cp_gc.detach(), knots, degree).sum(),
                u_gc_batch.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_cp: bspline_curves(u_gc_batch.detach(), x_cp, knots, degree).sum(),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

    def test_multiple_curves_equivalence(self):
        num_curves_m = 3
        n_samples_n = 5
        dim_d = 2
        degree = 2
        n_cp_c = 4  # Number of control points per curve

        knots = self.generate_clamped_knot_vector(
            n_cp_c, degree, device=self.device, dtype=self.default_dtype
        )  # Knots are in [-1,1]

        control_points_batched = torch.randn(num_curves_m, n_cp_c, dim_d, dtype=self.default_dtype, device=self.device)
        control_points_batched_clone_for_grad = control_points_batched.clone().requires_grad_(True)

        # u values for M curves: (N, M), in knot range [-1,1]
        u_batched_rand = torch.rand(n_samples_n, num_curves_m, dtype=self.default_dtype, device=self.device)
        # Scale u to be within the effective knot range [knots[degree], knots[n_cp_c]]
        # For default knots: knots[degree]=-1, knots[n_cp_c]=1
        knot_min_effective = knots[degree]
        knot_max_effective = knots[n_cp_c]  # This is the start of the last span's p+1 knots.
        # For u, it should be knots[n_cp_c] which is the end of the domain.

        u_batched = u_batched_rand * (knot_max_effective - knot_min_effective) + knot_min_effective
        u_batched_clone_for_grad = u_batched.clone().requires_grad_(True)

        # 1. Evaluate all curves together
        points_batched = bspline_curves(u_batched_clone_for_grad, control_points_batched_clone_for_grad, knots, degree)

        # 2. Evaluate each curve individually
        points_individual_list = []
        for i in range(num_curves_m):
            cp_single = control_points_batched[i : i + 1, :, :].clone()  # Shape (1, C, D)
            u_single = u_batched[:, i : i + 1].clone()  # Shape (N, 1)

            # For individual evaluation, BSplineFunction expects (M_cp=1, C, D) and (N, M_u=1)
            points_single = bspline_curves(u_single, cp_single, knots, degree)  # Output (N, 1, D)
            points_individual_list.append(points_single)

        points_stacked = torch.cat(points_individual_list, dim=1)  # (N,M,D)
        torch.testing.assert_close(points_batched.data, points_stacked.data, atol=1e-6, rtol=1e-5)

        # Compare backward pass
        grad_output = torch.randn_like(points_batched)

        points_batched.backward(grad_output)
        grad_u_batched_actual = u_batched_clone_for_grad.grad.clone()
        grad_cp_batched_actual = control_points_batched_clone_for_grad.grad.clone()

        # Zero grads for individual calculations
        # We need new tensors for individual grad accumulation if we want to compare to original batched grads

        expected_grad_u_from_individuals = torch.zeros_like(u_batched)
        expected_grad_cp_from_individuals = torch.zeros_like(control_points_batched)

        for i in range(num_curves_m):
            cp_single_grad_target = control_points_batched[i : i + 1, :, :].detach().clone().requires_grad_(True)
            u_single_grad_target = u_batched[:, i : i + 1].detach().clone().requires_grad_(True)

            points_single_eval = bspline_curves(u_single_grad_target, cp_single_grad_target, knots, degree)
            grad_output_single = grad_output[:, i : i + 1, :]
            points_single_eval.backward(grad_output_single)

            expected_grad_u_from_individuals[:, i : i + 1] = u_single_grad_target.grad
            expected_grad_cp_from_individuals[i : i + 1, :, :] = cp_single_grad_target.grad

        torch.testing.assert_close(grad_u_batched_actual, expected_grad_u_from_individuals, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(grad_cp_batched_actual, expected_grad_cp_from_individuals, atol=1e-6, rtol=1e-5)


class TestBSplineCurveModule(unittest.TestCase):
    def setUp(self):
        self.default_dtype = torch.float64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_init_with_int(self):
        num_curves = 1
        dim = 2
        degree = 3
        n_cps_per_curve = 5
        module = (
            BSplineCurve(num_curves=num_curves, dim=dim, degree=degree, knots_config=n_cps_per_curve)
            .to(self.device)
            .to(self.default_dtype)
        )

        self.assertEqual(module.num_curves, num_curves)
        self.assertEqual(module.n_control_points_per_curve, n_cps_per_curve)
        self.assertEqual(module.dim, dim)
        self.assertEqual(module.degree, degree)
        self.assertIsInstance(module.control_points, nn.Parameter)
        self.assertTrue(module.control_points.requires_grad)
        self.assertEqual(module.control_points.shape, (num_curves, n_cps_per_curve, dim))
        self.assertIsInstance(module.knots, torch.Tensor)
        self.assertEqual(module.knots.shape[0], n_cps_per_curve + degree + 1)
        # Check if knots are clamped to [-1,1]
        self.assertTrue(torch.all(module.knots[0 : degree + 1] == -1.0))
        self.assertTrue(torch.all(module.knots[n_cps_per_curve:] == 1.0))

    def test_init_with_tensor(self):
        num_curves = 1
        dim = 3
        degree = 2
        # n_cp=4, deg=2 -> knots=7. Example knots in [0,1]
        knots_tensor = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=self.default_dtype)
        expected_n_cps_per_curve = 4  # 7 - 2 - 1 = 4

        module = (
            BSplineCurve(num_curves=num_curves, dim=dim, degree=degree, knots_config=knots_tensor)
            .to(self.device)
            .to(self.default_dtype)
        )

        self.assertEqual(module.n_control_points_per_curve, expected_n_cps_per_curve)
        self.assertEqual(module.control_points.shape, (num_curves, expected_n_cps_per_curve, dim))
        torch.testing.assert_close(module.knots, knots_tensor.to(self.device).to(self.default_dtype))

    def test_init_errors(self):
        with self.assertRaisesRegex(ValueError, "must be greater than the degree"):
            BSplineCurve(num_curves=1, dim=2, degree=3, knots_config=3)  # n_cp <= degree

        knots_tensor_short = torch.tensor([0.0, 0.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "must be greater than the degree"):
            BSplineCurve(num_curves=1, dim=2, degree=3, knots_config=knots_tensor_short)

        with self.assertRaisesRegex(TypeError, "knots_config must be an int .*or.*Tensor.*"):
            BSplineCurve(num_curves=1, dim=2, degree=3, knots_config="wrong_type")  # type: ignore

        knots_tensor_2d = torch.tensor([[0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "Provided knots_config tensor must be 1D"):
            BSplineCurve(num_curves=1, dim=2, degree=1, knots_config=knots_tensor_2d)

    def test_forward_pass_shape_and_device(self):
        num_curves = 1
        dim = 3
        degree = 2
        n_cps_per_curve = 4
        batch_size = 10  # Number of u-samples per curve
        module = (
            BSplineCurve(num_curves=num_curves, dim=dim, degree=degree, knots_config=n_cps_per_curve)
            .to(self.device)
            .to(self.default_dtype)
        )

        # u: (N, M)
        u_scalar = torch.linspace(-1, 1, batch_size, device=self.device, dtype=self.default_dtype)
        u = u_scalar.unsqueeze(1)  # (N,1) for M=1 curve

        points = module(u)  # Output (N,M,D)

        self.assertEqual(points.shape, (batch_size, num_curves, dim))
        self.assertEqual(points.device, self.device)
        self.assertEqual(points.dtype, self.default_dtype)

    def test_boundary_interpolation_with_clamp_normalization(self):
        num_curves = 1
        dim = 2
        degree = 3
        n_cps_per_curve = 5
        module = (
            BSplineCurve(
                num_curves=num_curves, dim=dim, degree=degree, knots_config=n_cps_per_curve, normalize_fn="clamp"
            )
            .to(self.device)
            .to(self.default_dtype)
        )  # Knots are [-1,1]

        # u: (N,M) -> (1,1)
        u_start = torch.tensor([[-1.0]], device=self.device, dtype=self.default_dtype)
        u_end = torch.tensor([[1.0]], device=self.device, dtype=self.default_dtype)

        point_start = module(u_start)  # (1,1,D)
        point_end = module(u_end)  # (1,1,D)

        # module.control_points is (1,C,D)
        torch.testing.assert_close(point_start, module.control_points[:, 0:1, :])
        torch.testing.assert_close(point_end, module.control_points[:, -1:, :])

    def test_backward_pass(self):
        num_curves = 1
        dim = 2
        degree = 2
        n_cps_per_curve = 4
        module = (
            BSplineCurve(num_curves=num_curves, dim=dim, degree=degree, knots_config=n_cps_per_curve)
            .to(self.device)
            .to(self.default_dtype)
        )

        # u: (N,M)
        u = torch.tensor([[-0.7], [0.6]], device=self.device, dtype=self.default_dtype)  # N=2, M=1

        self.assertIsNone(module.control_points.grad)

        points = module(u)  # (2,1,D)
        loss = points.sum()
        loss.backward()

        self.assertIsNotNone(module.control_points.grad)
        self.assertEqual(module.control_points.grad.shape, module.control_points.shape)  # (1,C,D)
        self.assertNotEqual(torch.sum(module.control_points.grad**2).item(), 0.0)

    def test_gradcheck_module(self):
        num_curves = 1
        dim = 2
        degree = 2
        n_cps_per_curve = 3
        module = (
            BSplineCurve(num_curves=num_curves, dim=dim, degree=degree, knots_config=n_cps_per_curve)
            .to(self.device)
            .to(self.default_dtype)
        )

        # u_gc: (N,M)
        u_gc = torch.tensor([[-0.75], [0.25]], device=self.device, dtype=self.default_dtype).requires_grad_(True)

        # Check BSplineFunction.apply part
        cp_gc = module.control_points.clone().requires_grad_(True)  # (1,C,D)
        knots = module.knots
        current_degree = module.degree  # Use module's degree

        # Output of apply is (N,M,D), sum for gradcheck
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda u_in, cp_in: bspline_curves(u_in, cp_in, knots, current_degree).sum(),
                (u_gc, cp_gc),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

        # Check module call w.r.t 'u'
        module_clone = (
            BSplineCurve(num_curves=num_curves, dim=dim, degree=degree, knots_config=n_cps_per_curve)
            .to(self.device)
            .to(self.default_dtype)
        )
        module_clone.load_state_dict(module.state_dict())

        u_gc_mod = torch.tensor([[-0.6]], device=self.device, dtype=self.default_dtype).requires_grad_(True)  # (1,1)

        # Output of module is (N,M,D), sum for gradcheck
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda u_in: module_clone(u_in).sum(), u_gc_mod, eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-7
            )
        )

    def test_device_movement(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device movement test.")

        num_curves = 1
        dim = 2
        degree = 2
        n_cps_per_curve = 4
        module_cpu = BSplineCurve(num_curves, dim, degree, n_cps_per_curve)

        self.assertEqual(module_cpu.control_points.device.type, "cpu")
        self.assertEqual(module_cpu.knots.device.type, "cpu")

        module_cuda = module_cpu.to("cuda").to(self.default_dtype)

        self.assertEqual(module_cuda.control_points.device.type, "cuda")
        self.assertEqual(module_cuda.knots.device.type, "cuda")

        u_cuda = torch.tensor([[-0.7], [0.6]], device="cuda", dtype=self.default_dtype)  # (2,1)
        points = module_cuda(u_cuda)  # (2,1,D)

        self.assertEqual(points.device.type, "cuda")
        self.assertEqual(points.shape, (2, num_curves, dim))

        loss = points.sum()
        loss.backward()
        self.assertIsNotNone(module_cuda.control_points.grad)
        self.assertEqual(module_cuda.control_points.grad.device.type, "cuda")


def test_bspline_curves_default_knots_device_dtype():
    dtype = torch.float64
    device = torch.device("cpu")

    u = torch.tensor([[0.0]], dtype=dtype, device=device)
    control_points = torch.zeros((1, 4, 1), dtype=dtype, device=device)

    out = bspline_curves(u, control_points)

    assert out.dtype == control_points.dtype


def test_bspline_curves_default_knots_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping test.")

    dtype = torch.float64
    device = torch.device("cuda")

    control_points = torch.randn(1, 4, 1, dtype=dtype, device=device)
    u = torch.linspace(-1, 1, 5, dtype=dtype, device=device).unsqueeze(1)

    result = bspline_curves(u, control_points, knots=None, degree=3)

    assert result.device == device
    assert result.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
