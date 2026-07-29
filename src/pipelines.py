"""Multi-stage reconstruction + region-sensing pipelines.

Each function composes a reconstruction (upscaler) model with a region-sensing
(importance mask) model. The mask selects, under the memory budget K, which
patches are fetched at high resolution and blended into the upscaled output:

    reconstructed = upscaled + (hr - upscaled) * mask

Here ``hr`` stands for the higher-resolution reference that the selected
patches are sensed from; the mask keeps only the top-K most important patches,
modelling the fixed memory budget of the sensing device.

These compositions are extracted verbatim (behaviour-preserving) from the
original test script.
"""


def reconstruct_single(upscaler, lr_tensor):
    """Plain upscaling with no region sensing (e.g. TransConv 128 -> 256)."""
    return upscaler(lr_tensor)


def reconstruct_with_mask(upscaler, mask_net, lr_tensor, hr_tensor):
    """One-stage: upscale, then blend top-K high-res patches via the mask.

    Args:
        upscaler: Reconstruction model (e.g. UDUCNN).
        mask_net: Region-sensing model (e.g. MRIMCNN) matching the upscaled size.
        lr_tensor: Low-resolution input (B, 3, h, w).
        hr_tensor: High-resolution reference at the upscaled size, sensed for
            the selected patches.

    Returns:
        The reconstructed tensor at the upscaled resolution.
    """
    up = upscaler(lr_tensor)
    mask = mask_net(up)
    return up + (hr_tensor - up) * mask


def reconstruct_two_stage_transconv(transconv1, mask_net, transconv2, lr_tensor, hr_mid_tensor):
    """Two-stage 128 -> 256 -> 512 using two TransConv upscalers and one mask.

    Stage 1 upscales 128 -> 256, senses top-K patches against ``hr_mid_tensor``
    (the 256 reference), then stage 2 upscales the blended result 256 -> 512.
    """
    up = transconv1(lr_tensor)
    mask = mask_net(up)
    reconstructed = up + (hr_mid_tensor - up) * mask
    return transconv2(reconstructed)


def reconstruct_two_stage_masked(upscaler, mask_net1, mask_net2,
                                 lr_tensor, hr_mid_tensor, hr_full_tensor):
    """Two-stage 128 -> 256 -> 512, applying region sensing at each stage.

    Stage 1: upscale 128 -> 256, blend top-K patches sensed from the 256
    reference (``hr_mid_tensor``).
    Stage 2: upscale the result 256 -> 512, blend top-K patches sensed from the
    512 reference (``hr_full_tensor``).

    The same ``upscaler`` instance is reused across both stages, matching the
    original script.
    """
    step1 = upscaler(lr_tensor)
    mask = mask_net1(step1)
    step2 = step1 + (hr_mid_tensor - step1) * mask
    step3 = upscaler(step2)
    mask2 = mask_net2(step3)
    step4 = step3 + (hr_full_tensor - step3) * mask2
    return step4
