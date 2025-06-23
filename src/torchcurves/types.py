from typing import Protocol, Sequence, Union

import torch

Numeric = Union[int, float]
TensorLike = Union[torch.Tensor, Sequence[Numeric]]


class NormalizationFn(Protocol):
    """Protocol for normalization functions.

    A normalization function takes a tensor and normalizes it based on the provided parameters.

    Args:
        tensor (TensorLike): The input tensor to normalize.
        min_val (float): The minimum value for normalization.
        max_val (float): The maximum value for normalization.
        scale (float): Scale factor for normalization.

    Returns:
        torch.Tensor: The normalized tensor.

    """

    def __call__(self, x: TensorLike, scale: float, out_min: float, out_max: float) -> torch.Tensor: ...
