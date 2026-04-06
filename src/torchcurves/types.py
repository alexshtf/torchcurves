from typing import Protocol, Sequence, Union

import torch

Numeric = Union[int, float]
"""A number"""

TensorLike = Union[torch.Tensor, Sequence[Numeric]]
"""A PyTorch tensor or a sequence of numbers"""


class InputMap(Protocol):
    """Protocol for input maps.

    An input map takes raw feature values and maps them to a target interval
    chosen by the curve family.

    Args:
        tensor: The input tensor to map.
        out_min: The lower end of the target interval.
        out_max: The upper end of the target interval.

    Returns:
        The mapped tensor.

    """

    def __call__(self, x: TensorLike, out_min: float, out_max: float) -> torch.Tensor: ...
