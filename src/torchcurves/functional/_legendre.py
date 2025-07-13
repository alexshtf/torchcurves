import torch


def legendre_curves(x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Evaluate curves parametrized by Legendre polynomials.

    Args:
        coefficients: A tensor of size (N, C, M) of curve coefficients, of a set of C polynomial curves in M dimensions
        of degree N-1, represented in the Legendre basis.
        x: Batch of size (B, C), where x[:, j] is the batch of inputs for the j-th curve in the batch.

    Returns:
        points: Evaluated points on the curves, shape (B, C, M).

    """
    n, c, m = coefficients.shape  # n - number of coefficients, c - number of curves, m - curve dimension
    x = x.unsqueeze(-1).expand(-1, -1, m)  # (b × c × m), b = batch size
    b2 = torch.zeros_like(x)  # (b × c × m)
    b1 = torch.zeros_like(x)  # (b × c × m)
    for k in reversed(range(n)):
        alpha = (2 * k + 1) / (k + 1)
        beta = (k + 1) / (k + 2)
        curr_coef = coefficients[k].unsqueeze(0)  # (1 x c x m)
        bnext = torch.add(torch.addcmul(curr_coef, x, b1, value=alpha), b2, alpha=-beta)
        b2, b1 = b1, bnext
    return b1
