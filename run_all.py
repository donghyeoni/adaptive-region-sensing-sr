"""Regenerate the committed artifacts under ``results/`` in one command.

The full project trains on COCO images with a GPU; that data is not
redistributed. To give a **reproducible smoke run with no external data**, this
script synthesizes a tiny set of images, trains the ``TransConv`` model for a
few epochs on CPU at reduced resolution, and saves:

* ``results/train.log``       -- per-epoch MSE loss
* ``results/model_summary.txt`` -- torchinfo layer/param summary
* ``results/sample_sr.png``   -- LR input / model output / HR target triptych
* ``results/metrics.json``    -- final loss + parameter count

This is a pipeline sanity check (small synthetic data, few epochs), not a
super-resolution benchmark. Run ``train.py`` on COCO for real results.

Usage
-----
    python run_all.py
"""

import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torchinfo import summary

from src.data.dataset import LoadDataset
from src.models.reconstruction import TransConv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(OUT_DIR, "synthetic_data")
LR_SIZE = 64                     # HR target side = 128
CKPT = os.path.join(REPO_ROOT, "weights", "model.pt")


def make_image(path, size=256, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    r = (xx / size * 255)
    g = (yy / size * 255)
    b = (np.sin(xx / 20.0) + np.cos(yy / 20.0)) * 60 + 128
    img = np.stack([r, g, b], axis=2).astype(np.float64)
    for _ in range(6):
        cx, cy = rng.integers(0, size, 2)
        rad = int(rng.integers(20, 60))
        color = rng.integers(0, 256, 3)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
        img[mask] = color
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    os.makedirs(os.path.join(DATA_DIR, "Train"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "Test"), exist_ok=True)
    for i in range(12):
        make_image(os.path.join(DATA_DIR, "Train", f"img_{i:02d}.png"), seed=i)
    for i in range(4):
        make_image(os.path.join(DATA_DIR, "Test", f"img_{i:02d}.png"), seed=100 + i)

    # temp config for a fast CPU smoke run
    cfg = {
        "data": {"train_dir": os.path.join(DATA_DIR, "Train"),
                 "test_dir": os.path.join(DATA_DIR, "Test"),
                 "lr_size": LR_SIZE, "max_cache_size": 100},
        "train": {"model": "transconv", "batch_size": 4, "learning_rate": 0.001,
                  "num_epochs": 10, "num_workers": 0,
                  "scheduler_step_size": 8, "scheduler_gamma": 0.1,
                  "checkpoint_path": CKPT},
    }
    cfg_path = os.path.join(OUT_DIR, "train_config.yaml")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    print("Training TransConv on synthetic data (CPU smoke run) ...")
    proc = subprocess.run([sys.executable, "train.py", "--config", cfg_path,
                           "--seed", "0"], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    with open(os.path.join(OUT_DIR, "train.log"), "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n" + proc.stderr)

    # parse final loss
    final_loss = None
    for line in proc.stdout.splitlines():
        if "Loss:" in line:
            try:
                final_loss = float(line.rsplit("Loss:", 1)[1].strip().rstrip(".,"))
            except ValueError:
                pass

    # model summary + a qualitative triptych
    model = TransConv()
    with open(os.path.join(OUT_DIR, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(str(summary(model, input_size=(1, 3, LR_SIZE, LR_SIZE), verbose=0)))
    n_params = sum(p.numel() for p in model.parameters())

    if os.path.isfile(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    ds = LoadDataset(os.path.join(DATA_DIR, "Test"), lr_size=LR_SIZE)
    lr, hr = ds[0]
    with torch.no_grad():
        out = model(lr.unsqueeze(0)).squeeze(0).clamp(0, 1)

    def chw_to_img(t):
        return t.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, img, title in zip(
            axes,
            [chw_to_img(lr), chw_to_img(out), chw_to_img(hr)],
            [f"LR input ({LR_SIZE}px)", "TransConv output", f"HR target ({2*LR_SIZE}px)"]):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sample_sr.png"), dpi=150, bbox_inches="tight")

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"model": "transconv", "epochs": 10, "lr_size": LR_SIZE,
                   "hr_size": 2 * LR_SIZE, "final_train_loss": final_loss,
                   "num_parameters": n_params}, f, indent=2)

    print(f"Done. final_train_loss={final_loss}, params={n_params}. Artifacts under results/.")


if __name__ == "__main__":
    main()
