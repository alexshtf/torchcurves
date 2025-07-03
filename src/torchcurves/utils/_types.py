from typing import Protocol, Sequence, Union

import torch

Numeric = Union[int, float]
"""A number"""

TensorLike = Union[torch.Tensor, Sequence[Numeric]]
"""A PyTorch tensor or a sequence of numbers"""


class NormalizationFn(Protocol):
    """Protocol for normalization functions.

    A normalization function takes a tensor and normalizes it based on the provided parameters.

    Args:
        tensor: The input tensor to normalize.
        min_val: The minimum value for normalization.
        max_val: The maximum value for normalization.
        scale: Scale factor for normalization.

    Returns:
        The normalized tensor.

    """

    def __call__(self, x: TensorLike, scale: float, out_min: float, out_max: float) -> torch.Tensor: ...
