"""Evaluation entry point.

Loads pretrained reconstruction and region-sensing models, then reports the
average PSNR over the test set for several reconstruction pipelines. Optionally
produces a qualitative side-by-side visualization.

Weights are loaded from paths in the config (weights/ is gitignored and NOT
included in the repo). Any pipeline whose weights are missing is skipped.

Usage:
    python test.py --config configs/default.yaml
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from src.metrics import compute_psnr
from src.models.reconstruction import TransConv, UDUCNN, UUDCNN
from src.models.regionsensing import IMCNN, MRIMCNN
from src.pipelines import (
    reconstruct_single,
    reconstruct_with_mask,
    reconstruct_two_stage_transconv,
    reconstruct_two_stage_masked,
)


def _load(model, path, device):
    """Load a state dict into ``model``; return None if the file is missing."""
    if not os.path.isfile(path):
        print(f"[skip] weights not found: {path}")
        return None
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _evaluate(image_paths, device, fn):
    """Run ``fn(hr_pil) -> (target_tensor, output_tensor)`` over all images and
    return the average PSNR."""
    total_psnr, total = 0.0, 0
    with torch.no_grad():
        for path in tqdm(image_paths):
            hr = Image.open(path).convert("RGB")
            target, output = fn(hr)
            total_psnr += compute_psnr(target, output).item()
            total += 1
    return total_psnr / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    w = cfg["weights"]
    em = cfg["eval_models"]

    # Reconstruction models
    transconv1 = _load(TransConv().to(device), w["transconv1"], device)
    transconv2 = _load(TransConv().to(device), w["transconv2"], device)
    uducnn = _load(UDUCNN().to(device), w["uducnn"], device)
    uudcnn = _load(UUDCNN().to(device), w["uudcnn"], device)

    # Region-sensing models (hard mask at eval time)
    def _rs(cls, key, wkey):
        m = cls(**em[key]).to(device)
        m = _load(m, w[wkey], device)
        if m is not None:
            m.use_hard_mask = True
        return m

    imcnn1 = _rs(IMCNN, "imcnn1", "imcnn1")
    imcnn2 = _rs(IMCNN, "imcnn2", "imcnn2")
    mrimcnn1 = _rs(MRIMCNN, "mrimcnn1", "mrimcnn1")
    mrimcnn2 = _rs(MRIMCNN, "mrimcnn2", "mrimcnn2")

    test_dir = cfg["data"]["test_dir"]
    image_paths = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    if not image_paths:
        raise RuntimeError(
            f"No *.png images found in {test_dir}. "
            "Set data.test_dir in the config to your COCO test folder."
        )

    # --- Pipeline 1: TransConv plain upscaling 128 -> 256 -------------------
    if transconv1 is not None:
        def fn(hr):
            hr256 = hr.resize((256, 256), Image.BICUBIC)
            w_, h_ = hr256.size
            lr = hr256.resize((w_ // 2, h_ // 2), Image.BICUBIC)
            lr_t = to_tensor(lr).unsqueeze(0).to(device)
            out = reconstruct_single(transconv1, lr_t).squeeze(0).cpu()
            return to_tensor(hr256), out

        print(f"[TransConv 128->256] Average PSNR: {_evaluate(image_paths, device, fn):.4f}")

    # --- Pipeline 2: UDUCNN + MRIMCNN mask blend at 256 ---------------------
    if uducnn is not None and mrimcnn1 is not None:
        def fn(hr):
            hr256 = hr.resize((256, 256), Image.BICUBIC)
            w_, h_ = hr256.size
            lr = hr256.resize((w_ // 2, h_ // 2), Image.BICUBIC)
            hr_t = to_tensor(hr256).unsqueeze(0).to(device)
            lr_t = to_tensor(lr).unsqueeze(0).to(device)
            out = reconstruct_with_mask(uducnn, mrimcnn1, lr_t, hr_t).squeeze(0).cpu()
            return to_tensor(hr256), out

        print(f"[UDUCNN + MRIMCNN @256] Average PSNR: {_evaluate(image_paths, device, fn):.4f}")

    # --- Pipeline 3: two-stage TransConv 128 -> 256 -> 512 ------------------
    if transconv1 is not None and transconv2 is not None and mrimcnn1 is not None:
        def fn(hr):
            hr512 = hr
            hr256 = hr512.resize((256, 256), Image.BICUBIC)
            hr128 = hr512.resize((128, 128), Image.BICUBIC)
            hr_mid = to_tensor(hr256).unsqueeze(0).to(device)
            lr_t = to_tensor(hr128).unsqueeze(0).to(device)
            out = reconstruct_two_stage_transconv(
                transconv1, mrimcnn1, transconv2, lr_t, hr_mid
            ).squeeze(0).cpu()
            return to_tensor(hr512), out

        print(f"[TransConv two-stage 128->256->512] Average PSNR: {_evaluate(image_paths, device, fn):.4f}")

    # --- Pipeline 4: two-stage UUDCNN + IMCNN 128 -> 256 -> 512 -------------
    if uudcnn is not None and imcnn1 is not None and imcnn2 is not None:
        def fn(hr):
            hr512 = hr
            hr256 = hr512.resize((256, 256), Image.BICUBIC)
            hr128 = hr512.resize((128, 128), Image.BICUBIC)
            hr_full = to_tensor(hr512).unsqueeze(0).to(device)
            hr_mid = to_tensor(hr256).unsqueeze(0).to(device)
            lr_t = to_tensor(hr128).unsqueeze(0).to(device)
            out = reconstruct_two_stage_masked(
                uudcnn, imcnn1, imcnn2, lr_t, hr_mid, hr_full
            ).squeeze(0).cpu()
            return to_tensor(hr512), out

        print(f"[UUDCNN + IMCNN two-stage 128->256->512] Average PSNR: {_evaluate(image_paths, device, fn):.4f}")

    # --- Qualitative visualization -----------------------------------------
    viz_image = cfg.get("viz_image") or ""
    if viz_image and os.path.isfile(viz_image) and uudcnn is not None:
        hr_512 = Image.open(viz_image).convert("RGB")
        hr_256 = hr_512.resize((256, 256), Image.BICUBIC)
        hr_128 = hr_512.resize((128, 128), Image.BICUBIC)
        lr_t = to_tensor(hr_128).unsqueeze(0).to(device)

        with torch.no_grad():
            step1 = uudcnn(lr_t)

        psnr_val = compute_psnr(to_tensor(hr_256), step1.squeeze(0).cpu())

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(hr_512)
        plt.title("Original", fontsize=15)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(step1.squeeze(0).cpu().permute(1, 2, 0))
        plt.title(f"Output (PSNR: {psnr_val:.2f} dB)", fontsize=15)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
