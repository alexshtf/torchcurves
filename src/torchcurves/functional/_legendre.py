from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint


def _checkpoint(fn, *args):
    try:
        return checkpoint.checkpoint(fn, *args, use_reentrant=False)
    except TypeError:
        return checkpoint.checkpoint(fn, *args)


def _clenshaw_segment(
    b1: torch.Tensor,
    b2: torch.Tensor,
    coeffs_chunk: torch.Tensor,
    x_expanded: torch.Tensor,
    k0: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    for i in range(coeffs_chunk.shape[0]):
        k = k0 - i
        alpha = (2 * k + 1) / (k + 1)
        beta = (k + 1) / (k + 2)
        curr_coef = coeffs_chunk[i].unsqueeze(0)
        b1_next = torch.add(torch.addcmul(curr_coef, x_expanded, b1, value=alpha), b2, alpha=-beta)
        b2, b1 = b1, b1_next
    return b1, b2


def _legendre_clenshaw(x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    n, c, m = coefficients.shape  # n - number of coefficients, c - number of curves, m - curve dimension
    x = x.unsqueeze(-1).expand(-1, -1, m)  # (b × c × m), b = batch size
    b2 = torch.zeros_like(x)  # (b × c × m)
    b1 = torch.zeros_like(x)  # (b × c × m)
    for k in reversed(range(n)):
        alpha = (2 * k + 1) / (k + 1)
        beta = (k + 1) / (k + 2)
        curr_coef = coefficients[k].unsqueeze(0)  # (1 x c x m)
        b1_next = torch.add(torch.addcmul(curr_coef, x, b1, value=alpha), b2, alpha=-beta)
        b2, b1 = b1, b1_next
    return b1


def legendre_curves(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    checkpoint_segments: Optional[int] = None,
) -> torch.Tensor:
    r"""Evaluate curves parametrized by Legendre polynomials.

    Args:
        coefficients: A tensor of size :math:`(N, C, D)` of curve coefficients, of a set of :math:`C` polynomial curves
            in :math:`\mathbb{R}^D` of degree :math:`N-1`, represented in the Legendre basis.
        x: Batch of size :math:`(B, C)`, where ``x[:, j]`` is the batch of inputs for the j-th curve in the batch.
        checkpoint_segments: Optional number of segments for gradient checkpointing. Larger values save memory but
            increase compute. Only used when gradients are enabled.

    Returns:
        Evaluated points on the curves, shape :math:`(B, C, D)`.

    Note:
        Uses the Clenshaw recursive algorithm, and thus requires :math:`O(N)` time. Implementation is vectorized along
        the :math:`B` and :math:`D` dimensions, but the algorithm requires a loop over the polynomial degree.

    """
    if checkpoint_segments is not None:
        if not isinstance(checkpoint_segments, int) or checkpoint_segments <= 0:
            raise ValueError("checkpoint_segments must be a positive integer or None.")
    if checkpoint_segments is None or not torch.is_grad_enabled():
        return _legendre_clenshaw(x, coefficients)
    if not (x.requires_grad or coefficients.requires_grad):
        return _legendre_clenshaw(x, coefficients)

    n, _, m = coefficients.shape
    num_segments = min(checkpoint_segments, n)
    chunk_size = (n + num_segments - 1) // num_segments

    x_expanded = x.unsqueeze(-1).expand(-1, -1, m)
    b2 = torch.zeros_like(x_expanded)
    b1 = torch.zeros_like(x_expanded)

    coeffs_rev = coefficients.flip(0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        coeffs_chunk = coeffs_rev[start:end]
        k0 = n - 1 - start

        def _segment(b1_t, b2_t, coeffs_chunk_t, x_expanded_t, k0=k0):
            return _clenshaw_segment(b1_t, b2_t, coeffs_chunk_t, x_expanded_t, k0)

        b1, b2 = _checkpoint(_segment, b1, b2, coeffs_chunk, x_expanded)

    return b1
