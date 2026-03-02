"""
eval.py - Evaluate a trained model on the test set.

Usage:
    python eval.py --config configs/resnet.yaml --checkpoint checkpoints/resnet/best_model.pth
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from data import ImageFolderDataset, get_test_transforms
from models import get_model
from utils import accuracy, load_checkpoint, get_logger, compute_class_accuracy, confusion_matrix, AverageMeter


def main(args):
    logger = get_logger("eval")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_cfg = cfg["data"]
    test_tf = get_test_transforms(data_cfg["image_size"], data_cfg["normalize"])
    test_ds = ImageFolderDataset(data_cfg["test_dir"], transform=test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    model_cfg = cfg["model"]
    model = get_model(
        model_cfg["name"],
        num_classes=model_cfg["num_classes"],
        **{k: v for k, v in model_cfg.items() if k not in ("name", "num_classes")},
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = AverageMeter("loss")
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1)
            loss_meter.update(loss.item(), images.size(0))
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    top1 = accuracy(
        torch.tensor([[1.0 if i == p else 0.0 for i in range(model_cfg["num_classes"])] for p in all_preds]),
        torch.tensor(all_targets),
    )[0]

    logger.info(f"Test Loss: {loss_meter.avg:.4f} | Top-1 Acc: {top1:.2f}%")

    class_acc = compute_class_accuracy(all_preds, all_targets, test_ds.classes)
    logger.info("Per-class accuracy:")
    for cls, acc in class_acc.items():
        logger.info(f"  {cls:<20} {acc:.2f}%")

    if args.confusion_matrix:
        cm = confusion_matrix(all_preds, all_targets, model_cfg["num_classes"])
        logger.info("Confusion Matrix:")
        header = " " * 20 + "  ".join(f"{c[:8]:>8}" for c in test_ds.classes)
        logger.info(header)
        for i, row in enumerate(cm):
            row_str = f"{test_ds.classes[i]:<20}" + "  ".join(f"{v:>8}" for v in row)
            logger.info(row_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--confusion_matrix", action="store_true")
    args = parser.parse_args()
    main(args)
