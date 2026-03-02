"""
train.py - Main training entry point.

Usage:
    python train.py --config configs/resnet.yaml
    python train.py --config configs/vgg.yaml --epochs 50 --lr 0.01
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data import ImageFolderDataset, get_train_transforms, get_val_transforms
from models import get_model
from utils import AverageMeter, accuracy, save_checkpoint, build_checkpoint, TrainingLogger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict):
    data_cfg = cfg["data"]
    train_tf = get_train_transforms(data_cfg["image_size"], data_cfg["normalize"])
    val_tf = get_val_transforms(data_cfg["image_size"], data_cfg["normalize"])

    train_ds = ImageFolderDataset(data_cfg["train_dir"], transform=train_tf)
    val_ds = ImageFolderDataset(data_cfg["val_dir"], transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    train_cfg = cfg["training"]
    sched_name = train_cfg.get("lr_scheduler", "cosine")

    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"]
        )
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg.get("lr_step_size", 30),
            gamma=train_cfg.get("lr_gamma", 0.1),
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple:
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        top1 = accuracy(outputs, targets)[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1, images.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple:
    model.eval()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        top1 = accuracy(outputs, targets)[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1, images.size(0))

    return loss_meter.avg, acc_meter.avg


def main(args):
    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    set_seed(args.seed)
    device = get_device()

    logger = TrainingLogger(cfg["logging"].get("log_file"))
    logger.log_info(f"Device: {device}")
    logger.log_info(f"Config: {args.config}")

    train_loader, val_loader, class_names = build_dataloaders(cfg)
    logger.log_info(f"Classes: {class_names}")

    model_cfg = cfg["model"]
    model = get_model(
        model_cfg["name"],
        num_classes=model_cfg["num_classes"],
        **{k: v for k, v in model_cfg.items() if k not in ("name", "num_classes")},
    ).to(device)
    logger.log_info(f"Parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    best_acc = 0.0
    epochs = cfg["training"]["epochs"]
    ckpt_cfg = cfg["checkpoint"]

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.log_epoch(epoch, epochs, train_loss, train_acc, val_loss, val_acc)

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        if epoch % ckpt_cfg.get("save_every", 10) == 0 or is_best:
            state = build_checkpoint(model, optimizer, epoch, best_acc, cfg)
            save_checkpoint(state, ckpt_cfg["save_dir"], f"epoch_{epoch}.pth", is_best)

    logger.log_info(f"Training complete. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
