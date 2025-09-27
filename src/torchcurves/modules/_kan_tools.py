import torch
from torch import nn


class Sum(nn.Module):
    """A pooling layer that sums along the given dimension.

    Args:
        dim: The dimension along which to sum.

    """

    def __init__(self, dim: int = -2):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return torch.sum(x, self.dim)
