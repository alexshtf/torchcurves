from typing import Callable, Sequence, Union

import torch

Numeric = Union[int, float]
TensorLike = Union[torch.Tensor, Sequence[Numeric]]
NormalizationFn = Callable[[TensorLike, float], torch.Tensor]
