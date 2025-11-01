import pytest
import torch

from torchcurves.modules._bspline import BSplineCurve


@pytest.mark.perf
@pytest.mark.parametrize("batch,curves,dim,degree,n_ctrl", [
    (256, 32, 64, 3, 16),
    (256, 64, 64, 3, 32),
    (256, 64, 64, 5, 32),
])
def test_bspline_forward(benchmark, device, sync, batch, curves, dim, degree, n_ctrl):
    torch.manual_seed(0)
    model = BSplineCurve(num_curves=curves, dim=dim, degree=degree, knots_config=n_ctrl).to(device)
    u = torch.rand(batch, curves, device=device)

    # Warmup
    _ = model(u)
    sync()

    def run():
        out = model(u)
        sync()
        return out

    benchmark(run)


@pytest.mark.perf
@pytest.mark.parametrize("batch,curves,dim,degree,n_ctrl", [
    (128, 32, 64, 3, 16),
    (128, 64, 64, 3, 32),
    (128, 64, 64, 5, 32),
])
def test_bspline_backward(benchmark, device, sync, batch, curves, dim, degree, n_ctrl):
    torch.manual_seed(0)
    model = BSplineCurve(num_curves=curves, dim=dim, degree=degree, knots_config=n_ctrl).to(device)
    u = torch.rand(batch, curves, device=device)

    # Warmup
    loss = model(u).square().mean()
    loss.backward()
    sync()

    def run():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        out = model(u)
        loss = out.square().mean()
        loss.backward()
        sync()
        return loss

    benchmark(run)



