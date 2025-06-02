import torch
from torch import nn


class Sum(nn.Module):
    def __init__(self, dim=-2):
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return torch.sum(x, self.dim)
