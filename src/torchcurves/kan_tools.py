import torch
from torch import nn


class Sum(nn.Module):
    def __init__(self, dim=1):
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return torch.sum(x, self.dim)


class Replicate(nn.Module):
    def __init__(self, num_replications, cls, *args, **kwargs):
        self.copies = nn.ModuleList([cls(*args, **kwargs) for _ in range(num_replications)])

    def forward(self, x: torch.Tensor):
        features = torch.unbind(x, 1)
        outputs = [copy(f) for f, copy in zip(features, self.copies, strict=True)]
        return torch.stack(outputs, 1)
