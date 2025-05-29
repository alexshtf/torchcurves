from typing import Callable, Sequence

import torch

Numeric = int | float
TensorLike = torch.Tensor | Sequence[Numeric]
NormalizationFn = Callable[[TensorLike, float], torch.Tensor]
