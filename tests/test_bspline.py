import unittest

import pytest
import torch
import torch.nn as nn

from torchcurves.bspline import BSplineCurve, BSplineFunction


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
        """Generate a clamped knot vector.

        A clamped knot vector has the first (degree+1) knots equal to the start value (usually 0)
        and the last (degree+1) knots equal to the end value (usually 1).
        This makes the B-spline curve interpolate the first and last control points.
        Total number of knots m = n_control_points + degree + 1.
        """
        if n_control_points <= degree:
            raise ValueError("Number of control points must be greater than degree.")

        num_knots = n_control_points + degree + 1
        knots = torch.zeros(num_knots, device=device, dtype=dtype)

        # Number of internal knot segments is n_control_points - degree - 1
        # These internal knots divide the range [0,1] into n_control_points - degree segments.
        # So there are n_control_points - degree - 1 distinct internal knot values.

        # Example: n_cp=4, deg=2. Knots m = 4+2+1 = 7.
        # Knots: [0,0,0, k_internal, 1,1,1]
        # Internal knots start at index `degree+1` and end at `n_control_points-1`.
        # Number of distinct internal knots: (n_control_points-1) - (degree+1) + 1 = n_control_points - degree - 1

        # Fill first degree+1 knots with 0
        # knots[0 : degree+1] = 0.0 (already initialized)

        # Fill last degree+1 knots with 1
        knots[n_control_points:] = 1.0

        num_internal_knots_to_set = n_control_points - degree - 1
        if num_internal_knots_to_set > 0:
            internal_knot_values = torch.linspace(0, 1, n_control_points - degree + 1, device=device, dtype=dtype)
            knots[degree + 1 : n_control_points] = internal_knot_values[1:-1]

        return knots

    def test_constant_function_degree0(self):
        degree = 0
        control_points = torch.tensor([[2.5]], dtype=self.default_dtype, device=self.device)
        # For deg=0, n_cp=1, knots m = 1+0+1=2. Clamped: [0,1]
        knots = self.generate_clamped_knot_vector(
            control_points.shape[0], degree, device=self.device, dtype=self.default_dtype
        )
        self.assertEqual(knots.shape[0], control_points.shape[0] + degree + 1)

        u_values = torch.tensor([0.0, 0.5, 0.99], dtype=self.default_dtype, device=self.device)

        for u_val_scalar in u_values:
            u = u_val_scalar.unsqueeze(0)  # Batch of 1
            points = BSplineFunction.apply(u, control_points, knots, degree)
            self.assertAlmostEqual(points.item(), control_points[0, 0].item(), places=5, msg=f"Failed for u={u.item()}")

            # Gradcheck for u
            u_gc = u.clone().requires_grad_(True)
            cp_gc = control_points.clone()  # No grad w.r.t CP for deg 0 if CP is fixed

            # For degree 0, the output is piecewise constant. Derivative is 0 almost everywhere.
            # gradcheck might fail at knot points if not handled carefully by the function.
            # BSplineFunction.compute_basis_derivatives returns 0 for degree 0, so grad_u should be 0.
            self.assertTrue(
                torch.autograd.gradcheck(
                    lambda x: BSplineFunction.apply(x, cp_gc, knots, degree),  # noqa: B023
                    u_gc,
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-3,
                    nondet_tol=1e-7,
                )
            )

            points_gc = BSplineFunction.apply(u_gc, cp_gc, knots, degree)
            points_gc.sum().backward()
            self.assertAlmostEqual(u_gc.grad.item(), 0.0, places=5, msg=f"Grad_u non-zero for u={u.item()}")

    def test_constant_function_all_cps_same(self):
        degree = 2
        n_cp = 4
        const_val = 5.0
        control_points = torch.full((n_cp, 1), const_val, dtype=self.default_dtype, device=self.device)
        knots = self.generate_clamped_knot_vector(n_cp, degree, device=self.device, dtype=self.default_dtype)

        u = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=self.default_dtype, device=self.device)

        points = BSplineFunction.apply(u, control_points, knots, degree)
        expected_points = torch.full((u.shape[0], 1), const_val, dtype=self.default_dtype, device=self.device)
        torch.testing.assert_close(points, expected_points, atol=1e-5, rtol=1e-5)

        # Test gradients
        u_gc = u.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)

        output = BSplineFunction.apply(u_gc, cp_gc, knots, degree)
        output.sum().backward()

        # grad_u should be zero
        torch.testing.assert_close(u_gc.grad, torch.zeros_like(u_gc), atol=1e-5, rtol=1e-5)

        # sum of grad_cp for each dimension should be 1 for each u value (because sum of basis functions is 1)
        # and grad_output is implicitly 1 for each point from output.sum().
        # So, total sum of grad_cp should be batch_size * dim_of_cp_output (which is 1 here).
        self.assertAlmostEqual(cp_gc.grad.sum().item(), u.shape[0], places=5)

    def test_linear_function_degree1(self):
        degree = 1
        # P0 = [0], P1 = [1] -> C(u) = u
        control_points = torch.tensor([[0.0], [1.0]], dtype=self.default_dtype, device=self.device)
        n_cp = control_points.shape[0]
        knots = self.generate_clamped_knot_vector(n_cp, degree, device=self.device, dtype=self.default_dtype)
        # Expected knots for n_cp=2, deg=1: [0,0,1,1]

        u = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=self.default_dtype, device=self.device)
        expected_points = u.unsqueeze(1)  # C(u) = u

        points = BSplineFunction.apply(u, control_points, knots, degree)
        torch.testing.assert_close(points, expected_points, atol=1e-6, rtol=1e-5)

        # Test gradients
        u_gc = u.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)

        # Gradcheck
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: BSplineFunction.apply(x, cp_gc.detach(), knots, degree),
                u_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: BSplineFunction.apply(u_gc.detach(), x, knots, degree),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

        # Analytical gradient for u
        output_an = BSplineFunction.apply(u_gc, cp_gc.detach(), knots, degree)
        output_an.sum().backward()
        expected_grad_u = torch.ones_like(u_gc)
        torch.testing.assert_close(u_gc.grad, expected_grad_u, atol=1e-6, rtol=1e-5)

    def test_parabola_degree2(self):
        # C(u) = u^2. P0=[0], P1=[0.5*(1/2)^2 / (1/2*1/2) ] = [0.5], P2=[1] is not u^2.
        # For C(u) = u^2, with knots [0,0,0,1,1,1] (deg=2, n_cp=3)
        # P0=[0], P1=[0], P2=[1] does NOT give u^2.
        # C(u) = N02(u)P0 + N12(u)P1 + N22(u)P2
        # N02 = (1-u)^2, N12 = 2u(1-u), N22 = u^2 for this knot vector.
        # So, C(u) = (1-u)^2 * 0 + 2u(1-u) * P1_val + u^2 * 1
        # If P1_val = 0, C(u) = u^2.
        degree = 2
        control_points = torch.tensor([[0.0], [0.0], [1.0]], dtype=self.default_dtype, device=self.device)
        n_cp = control_points.shape[0]
        knots = self.generate_clamped_knot_vector(n_cp, degree, device=self.device, dtype=self.default_dtype)

        u = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=self.default_dtype, device=self.device)
        expected_points = u.pow(2).unsqueeze(1)  # C(u) = u^2

        points = BSplineFunction.apply(u, control_points, knots, degree)
        torch.testing.assert_close(points, expected_points, atol=1e-6, rtol=1e-5)

        # Test gradients: C'(u) = 2u
        u_gc = u.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)

        # Gradcheck
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_u: BSplineFunction.apply(x_u, cp_gc.detach(), knots, degree),
                u_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_cp: BSplineFunction.apply(u_gc.detach(), x_cp, knots, degree),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

        # Analytical gradient for u
        output_an = BSplineFunction.apply(u_gc, cp_gc.detach(), knots, degree)
        output_an.sum().backward()
        expected_grad_u = 2 * u_gc.detach()
        torch.testing.assert_close(u_gc.grad, expected_grad_u, atol=1e-6, rtol=1e-5)

    def test_boundary_values(self):
        degree = 3
        n_cp = 5  # Example: P0, P1, P2, P3, P4
        control_points = torch.randn(n_cp, 2, dtype=self.default_dtype, device=self.device)
        knots = self.generate_clamped_knot_vector(n_cp, degree, device=self.device, dtype=self.default_dtype)

        u_start = torch.tensor([0.0], dtype=self.default_dtype, device=self.device)
        u_end = torch.tensor([1.0], dtype=self.default_dtype, device=self.device)

        point_start = BSplineFunction.apply(u_start, control_points, knots, degree)
        point_end = BSplineFunction.apply(u_end, control_points, knots, degree)

        torch.testing.assert_close(point_start, control_points[0].unsqueeze(0), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(point_end, control_points[-1].unsqueeze(0), atol=1e-6, rtol=1e-5)

    def test_multiple_dimensions(self):
        degree = 2
        # P0=(0,0), P1=(0.5,1), P2=(1,0) for a parabola opening downwards if knots are [000, 0.5, 111]
        # With standard clamped knots [0,0,0,1,1,1] and C(u) = (1-u)^2 P0 + 2u(1-u)P1 + u^2 P2
        control_points = torch.tensor(
            [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=self.default_dtype, device=self.device
        )
        n_cp = control_points.shape[0]
        knots = self.generate_clamped_knot_vector(n_cp, degree, device=self.device, dtype=self.default_dtype)

        u = torch.tensor([0.0, 0.5, 1.0], dtype=self.default_dtype, device=self.device)

        expected_points = torch.empty_like(u.unsqueeze(1).expand(-1, 2))
        expected_points[0] = control_points[0]  # C(0) = P0
        expected_points[1] = 0.25 * control_points[0] + 0.5 * control_points[1] + 0.25 * control_points[2]  # C(0.5)
        expected_points[2] = control_points[2]  # C(1) = P2

        points = BSplineFunction.apply(u, control_points, knots, degree)
        torch.testing.assert_close(points, expected_points, atol=1e-6, rtol=1e-5)

        # Gradcheck
        u_gc = u.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_u: BSplineFunction.apply(x_u, cp_gc.detach(), knots, degree).sum(),  # Sum for multi-dim output
                u_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_cp: BSplineFunction.apply(
                    u_gc.detach(), x_cp, knots, degree
                ).sum(),  # Sum for multi-dim output
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

    def test_batch_processing(self):
        degree = 1
        control_points = torch.tensor(
            [[0.0, 1.0], [2.0, 3.0]], dtype=self.default_dtype, device=self.device
        )  # 2 CPs, 2 Dim
        n_cp = control_points.shape[0]
        knots = self.generate_clamped_knot_vector(
            n_cp, degree, device=self.device, dtype=self.default_dtype
        )  # [0,0,1,1]

        u_batch = torch.tensor([0.0, 0.5, 1.0], dtype=self.default_dtype, device=self.device)  # Batch of 3

        expected_points_batch = torch.empty(
            (u_batch.shape[0], control_points.shape[1]), dtype=self.default_dtype, device=self.device
        )
        for i, u_val in enumerate(u_batch):
            expected_points_batch[i] = (1 - u_val) * control_points[0] + u_val * control_points[1]

        points_batch = BSplineFunction.apply(u_batch, control_points, knots, degree)
        torch.testing.assert_close(points_batch, expected_points_batch, atol=1e-6, rtol=1e-5)
        self.assertEqual(points_batch.shape, (u_batch.shape[0], control_points.shape[1]))

        # Gradcheck with batch
        u_gc_batch = u_batch.clone().requires_grad_(True)
        cp_gc = control_points.clone().requires_grad_(True)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_u: BSplineFunction.apply(x_u, cp_gc.detach(), knots, degree).sum(),
                u_gc_batch.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x_cp: BSplineFunction.apply(u_gc_batch.detach(), x_cp, knots, degree).sum(),
                cp_gc.detach().requires_grad_(True),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )


class TestBSplineCurveModule(unittest.TestCase):
    def setUp(self):
        self.default_dtype = torch.float64  # For gradcheck
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")

    def test_init_with_int(self):
        dim = 2
        degree = 3
        n_cps = 5
        module = BSplineCurve(dim=dim, degree=degree, knots_config=n_cps).to(self.device).to(self.default_dtype)

        self.assertEqual(module.n_control_points, n_cps)
        self.assertEqual(module.dim, dim)
        self.assertEqual(module.degree, degree)
        self.assertIsInstance(module.control_points, nn.Parameter)
        self.assertTrue(module.control_points.requires_grad)
        self.assertEqual(module.control_points.shape, (n_cps, dim))
        self.assertIsInstance(module.knots, torch.Tensor)
        self.assertEqual(module.knots.shape[0], n_cps + degree + 1)
        self.assertEqual(module.knots.device, self.device)
        self.assertEqual(module.control_points.device, self.device)
        self.assertEqual(module.knots.dtype, self.default_dtype)
        self.assertEqual(module.control_points.dtype, self.default_dtype)
        # Check if knots are clamped
        self.assertTrue(torch.all(module.knots[: degree + 1] == -1.0))
        self.assertTrue(torch.all(module.knots[n_cps:] == 1.0))

    def test_init_with_tensor(self):
        dim = 3
        degree = 2
        # n_cp=4, deg=2 -> knots=7
        knots_tensor = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=self.default_dtype)
        expected_n_cps = 4

        module = BSplineCurve(dim=dim, degree=degree, knots_config=knots_tensor).to(self.device).to(self.default_dtype)

        self.assertEqual(module.n_control_points, expected_n_cps)
        self.assertEqual(module.dim, dim)
        self.assertEqual(module.degree, degree)
        self.assertEqual(module.control_points.shape, (expected_n_cps, dim))
        self.assertTrue(module.control_points.requires_grad)
        torch.testing.assert_close(module.knots, knots_tensor.to(self.device).to(self.default_dtype))
        self.assertEqual(module.knots.device, self.device)
        self.assertEqual(module.control_points.device, self.device)

    def test_init_errors(self):
        # n_cp <= degree
        with self.assertRaisesRegex(ValueError, "must be greater than the degree"):
            BSplineCurve(dim=2, degree=3, knots_config=3)

        knots_tensor_short = torch.tensor([0.0, 0.0, 1.0, 1.0])  # 4 knots, deg=3 -> n_cp = 4-3-1 = 0. 0 <= 3.
        with self.assertRaisesRegex(ValueError, "must be greater than the degree"):
            BSplineCurve(dim=2, degree=3, knots_config=knots_tensor_short)

        # Wrong knots_config type
        with self.assertRaisesRegex(TypeError, "knots_config must be an int .*or.*Tensor.*"):
            BSplineCurve(dim=2, degree=3, knots_config="wrong_type")  # type: ignore

        # Wrong tensor dim
        knots_tensor_2d = torch.tensor([[0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "Provided knots_config tensor must be 1D"):
            BSplineCurve(dim=2, degree=1, knots_config=knots_tensor_2d)

    def test_forward_pass_shape_and_device(self):
        dim = 3
        degree = 2
        n_cps = 4
        batch_size = 10
        module = BSplineCurve(dim=dim, degree=degree, knots_config=n_cps).to(self.device).to(self.default_dtype)
        u = torch.linspace(0, 1, batch_size, device=self.device, dtype=self.default_dtype)

        points = module(u)

        self.assertEqual(points.shape, (batch_size, dim))
        self.assertEqual(points.device, self.device)
        self.assertEqual(points.dtype, self.default_dtype)

    def test_boundary_interpolation(self):
        dim = 2
        degree = 3
        n_cps = 5
        module = BSplineCurve(dim=dim, degree=degree, knots_config=n_cps).to(self.device).to(self.default_dtype)

        u_start = torch.tensor([-1.0], device=self.device, dtype=self.default_dtype)
        u_end = torch.tensor([1.0], device=self.device, dtype=self.default_dtype)

        point_start = module(u_start)
        point_end = module(u_end)

        torch.testing.assert_close(point_start, module.control_points[0].unsqueeze(0))
        torch.testing.assert_close(point_end, module.control_points[-1].unsqueeze(0))

    def test_backward_pass(self):
        dim = 2
        degree = 2
        n_cps = 4
        module = BSplineCurve(dim=dim, degree=degree, knots_config=n_cps).to(self.device).to(self.default_dtype)
        u = torch.tensor([0.3, 0.6], device=self.device, dtype=self.default_dtype)

        self.assertIsNone(module.control_points.grad)

        points = module(u)
        loss = points.sum()  # Simple loss
        loss.backward()

        self.assertIsNotNone(module.control_points.grad)
        self.assertEqual(module.control_points.grad.shape, module.control_points.shape)
        self.assertNotEqual(torch.sum(module.control_points.grad**2).item(), 0.0)

    def test_gradcheck_module(self):
        dim = 2
        degree = 2
        n_cps = 3  # Smallest possible for deg 2 (e.g., u^2 test)
        module = BSplineCurve(dim=dim, degree=degree, knots_config=n_cps).to(self.device).to(self.default_dtype)

        u_gc = torch.tensor([0.25, 0.75], device=self.device, dtype=self.default_dtype).requires_grad_(True)

        # We gradcheck the BSplineFunction.apply, using parameters from the module
        # This is a robust way to check gradients w.r.t CPs and U through the core function
        cp_gc = module.control_points.clone().requires_grad_(True)
        knots = module.knots
        degree = module.degree

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda u_in, cp_in: BSplineFunction.apply(u_in, cp_in, knots, degree),
                (u_gc, cp_gc),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
                nondet_tol=1e-7,
            )
        )

        # We can also gradcheck the module call w.r.t 'u'
        module_clone = BSplineCurve(dim=dim, degree=degree, knots_config=n_cps).to(self.device).to(self.default_dtype)
        module_clone.load_state_dict(module.state_dict())  # Ensure same CPs

        u_gc_mod = torch.tensor([0.4], device=self.device, dtype=self.default_dtype).requires_grad_(True)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda u_in: module_clone(u_in), u_gc_mod, eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-7
            )
        )

    def test_device_movement(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device movement test.")

        dim = 2
        degree = 2
        n_cps = 4
        module = BSplineCurve(dim=dim, degree=degree, knots_config=n_cps)  # Init on CPU

        self.assertEqual(module.control_points.device.type, "cpu")
        self.assertEqual(module.knots.device.type, "cpu")

        module = module.to("cuda").to(self.default_dtype)

        self.assertEqual(module.control_points.device.type, "cuda")
        self.assertEqual(module.knots.device.type, "cuda")

        u_cuda = torch.tensor([0.3, 0.6], device="cuda", dtype=self.default_dtype)
        points = module(u_cuda)

        self.assertEqual(points.device.type, "cuda")
        self.assertEqual(points.shape, (2, dim))

        loss = points.sum()
        loss.backward()
        self.assertIsNotNone(module.control_points.grad)
        self.assertEqual(module.control_points.grad.device.type, "cuda")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
