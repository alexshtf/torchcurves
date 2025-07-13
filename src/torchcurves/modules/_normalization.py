from ..functional import clamp, rational
from ..types import NormalizationFn

_normalization_catalogue: dict[str, NormalizationFn] = {
    "rational": rational,
    "clamp": clamp,
}
