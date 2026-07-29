"""Training entry point.

Trains a reconstruction model (TransConv / UDUCNN / UUDCNN) on COCO images
using MSE loss, Adam, and a StepLR schedule. All paths and hyper-parameters
come from a YAML config.

Usage:
    python train.py --config configs/default.yaml
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LoadDataset
from src.models.reconstruction import TransConv, UDUCNN, UUDCNN

MODELS = {
    "transconv": TransConv,
    "uducnn": UDUCNN,
    "uudcnn": UUDCNN,
}


def train(model, device, train_loader, optimizer, criterion, num_epochs, scheduler=None):
    """Standard supervised training loop. Returns an (epochs, 2) history array
    of [epoch, avg_loss]."""
    history = np.zeros((0, 2))
    model.train()

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history = np.vstack((history, np.array([epoch + 1, avg_loss])))

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset / dataloader
    trainset = LoadDataset(
        img_dir=data_cfg["train_dir"],
        lr_size=data_cfg["lr_size"],
        max_cache_size=data_cfg["max_cache_size"],
    )
    print(f"# of trainset = {len(trainset)}")
    if len(trainset) == 0:
        raise RuntimeError(
            f"No *.png images found in {data_cfg['train_dir']}. "
            "Set data.train_dir in the config to your COCO image folder."
        )

    trainloader = DataLoader(
        trainset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        drop_last=True,
    )

    # Model
    model_name = train_cfg["model"].lower()
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODELS)}.")
    model = MODELS[model_name]().to(device)

    # Optimizer / scheduler / loss
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg["scheduler_step_size"],
        gamma=train_cfg["scheduler_gamma"],
    )
    criterion = nn.MSELoss()

    # Train
    train(model, device, trainloader, optimizer, criterion, train_cfg["num_epochs"], scheduler)

    # Save
    ckpt_path = train_cfg["checkpoint_path"]
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
