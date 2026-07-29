"""Evaluation metrics."""

import torch


def compute_psnr(x1, x2):
    """Peak Signal-to-Noise Ratio for images in the [0, 1] range.

    Args:
        x1, x2: Tensors of the same shape with values in [0, 1].

    Returns:
        A scalar tensor with the PSNR in dB.
    """
    mse = torch.mean((x1 - x2) ** 2)
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr
