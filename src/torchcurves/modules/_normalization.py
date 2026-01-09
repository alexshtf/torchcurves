from ..functional import arctan, clamp, rational
from ..types import NormalizationFn

_normalization_catalogue: dict[str, NormalizationFn] = {
    "rational": rational,
    "clamp": clamp,
    "arctan": arctan,
}
