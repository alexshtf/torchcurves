import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "perf: performance benchmarks")


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return request.param


@pytest.fixture
def sync(device):
    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    return _sync
