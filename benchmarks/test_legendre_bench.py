import pytest
import torch

from torchcurves.modules._legendre import LegendreCurve


@pytest.mark.perf
@pytest.mark.parametrize(
    "batch,curves,dim,degree",
    [
        (256, 32, 64, 8),
        (256, 64, 64, 16),
    ],
)
def test_legendre_forward(benchmark, device, sync, batch, curves, dim, degree):
    """Benchmark forward pass only (no gradients required)."""
    torch.manual_seed(0)
    model = LegendreCurve(num_curves=curves, dim=dim, degree=degree).to(device)
    u = torch.rand(batch, curves, device=device)

    # Warmup (especially important for CUDA JIT and allocator)
    _ = model(u)
    sync()

    def run():
        out = model(u)
        sync()
        return out

    benchmark(run)


@pytest.mark.perf
@pytest.mark.parametrize(
    "batch,curves,dim,degree",
    [
        (128, 32, 64, 8),
        (128, 64, 64, 16),
    ],
)
def test_legendre_backward_params(benchmark, device, sync, batch, curves, dim, degree):
    """Benchmark backward pass through parameters only (inputs don't require grad)."""
    torch.manual_seed(0)
    model = LegendreCurve(num_curves=curves, dim=dim, degree=degree).to(device)
    u = torch.rand(batch, curves, device=device, requires_grad=False)

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


@pytest.mark.perf
@pytest.mark.parametrize(
    "batch,curves,dim,degree",
    [
        (128, 32, 64, 8),
        (128, 64, 64, 16),
    ],
)
def test_legendre_backward_inputs(benchmark, device, sync, batch, curves, dim, degree):
    """Benchmark backward pass through inputs only (parameters don't require grad)."""
    torch.manual_seed(0)
    model = LegendreCurve(num_curves=curves, dim=dim, degree=degree).to(device)
    # Disable gradients for parameters
    for p in model.parameters():
        p.requires_grad_(False)
    u = torch.rand(batch, curves, device=device, requires_grad=True)

    # Warmup
    loss = model(u).square().mean()
    loss.backward()
    sync()

    def run():
        if u.grad is not None:
            u.grad.zero_()
        out = model(u)
        loss = out.square().mean()
        loss.backward()
        sync()
        return loss

    benchmark(run)


@pytest.mark.perf
@pytest.mark.parametrize(
    "batch,curves,dim,degree",
    [
        (128, 32, 64, 8),
        (128, 64, 64, 16),
    ],
)
def test_legendre_backward_both(benchmark, device, sync, batch, curves, dim, degree):
    """Benchmark backward pass through both parameters and inputs."""
    torch.manual_seed(0)
    model = LegendreCurve(num_curves=curves, dim=dim, degree=degree).to(device)
    u = torch.rand(batch, curves, device=device, requires_grad=True)

    # Warmup
    loss = model(u).square().mean()
    loss.backward()
    sync()

    def run():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        if u.grad is not None:
            u.grad.zero_()
        out = model(u)
        loss = out.square().mean()
        loss.backward()
        sync()
        return loss

    benchmark(run)
