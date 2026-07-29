# Adaptive Region-Sensing Super-Resolution

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)

PyTorch image super-resolution / reconstruction under a memory budget.

## Overview

A learned **region-sensing** module predicts a per-patch importance map and,
under a fixed memory budget `K`, keeps only the **top-K most important patches**
at high resolution. The retained patches are blended into a CNN-upscaled base
image, modelling a memory-constrained sensing device (e.g. a UAV) that cannot
afford to fetch every patch at full resolution.

The project combines two families of models:

- **Reconstruction / upscalers** (`src/models/reconstruction.py`)
  - `TransConv` — single transposed-conv 2x upscaler
  - `UDUCNN` — up / down / up CNN (2x)
  - `UUDCNN` — up / up / down CNN (4x)
  - `ResidualBlock`, `ResidualBlock2` — residual building blocks
- **Region sensing / importance masks** (`src/models/regionsensing.py`)
  - `IMCNN` — single-resolution importance-mask CNN
  - `MRIMCNN` — multi-resolution importance-mask CNN (fuses full/½/¼ scales)

The memory budget is `total_memory = 128 * 128`, so `K = total_memory / patch_size²`
patches are kept. Models are trained with **MSE loss**, **Adam**, and a
**StepLR** schedule, and evaluated by **PSNR**, including a two-stage
`128 -> 256 -> 512` reconstruction.

See `docs/` for the accompanying presentation.

## Dataset

Training and evaluation use **COCO** images as high-resolution ground truth.
Each image is treated as an HR target (e.g. 512x512) and bicubic-downsampled to
produce the low-resolution input (256 or 128).

The dataset is **not included** in this repository. Download COCO images from
[cocodataset.org](http://cocodataset.org) and point the config at your local
folders (`data.train_dir` / `data.test_dir`). The loader globs `*.png` files in
those directories.

## Structure

```
adaptive-region-sensing-sr/
├── configs/
│   └── default.yaml            # data dirs, batch, lr, epochs, patch/image size, temperature, weight paths
├── docs/
│   └── Memory-constrained UAV multi-resolution sensing with adaptive region selection.pptx
├── src/
│   ├── data/
│   │   └── dataset.py          # LoadDataset: COCO *.png HR -> bicubic LR pairs
│   ├── models/
│   │   ├── reconstruction.py   # TransConv, UDUCNN, UUDCNN, ResidualBlock, ResidualBlock2
│   │   └── regionsensing.py    # IMCNN, MRIMCNN
│   ├── metrics.py              # compute_psnr
│   └── pipelines.py            # multi-stage reconstruct + mask compositions
├── train.py                    # training loop
├── test.py                     # PSNR evaluation + qualitative visualization
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Edit `configs/default.yaml` first — set `data.train_dir` / `data.test_dir` to
your local COCO folders and adjust hyper-parameters as needed.

**Train** a reconstruction model (`transconv` / `uducnn` / `uudcnn`, chosen via
`train.model` in the config):

```bash
python train.py --config configs/default.yaml
```

The trained checkpoint is written to `train.checkpoint_path`.

**Evaluate** PSNR across the reconstruction pipelines (and optionally render a
qualitative comparison by setting `viz_image`):

```bash
python test.py --config configs/default.yaml
```

`test.py` loads pretrained weights from the paths under `weights:` in the
config. Any pipeline whose weight files are missing is skipped.

## Notes

- **Weights are not included.** The `weights/` directory and all `*.pt` / `*.pth`
  files are gitignored. Provide your own trained checkpoints and update the
  `weights:` paths in the config.
- The original `loaddataandtrain.py` was a Colab export and **did not run
  as-is**. The following bugs were fixed while modularizing:
  - `LoadDataset1` (undefined) → `LoadDataset`
  - `model = "insert model"()` placeholder → a real model selected from the config
  - missing `import torch.nn as nn`
  - empty `torch.save('')` path → a real checkpoint path from the config
  - stripped the Colab header and the inline `!pip install torchinfo`
  - moved all hardcoded `/home/dh/...` paths and weight filenames into `configs/default.yaml`
  - de-duplicated `compute_psnr` (defined twice in `test.py`) into `src/metrics.py`
