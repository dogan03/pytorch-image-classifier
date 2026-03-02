from typing import Dict, List, Optional
import torch
import numpy as np


class AverageMeter:
    """Tracks a running average of a scalar metric."""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """Compute top-k accuracy for given outputs and targets."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(correct_k.mul(100.0 / batch_size).item())
        return results


def compute_class_accuracy(
    all_preds: List[int],
    all_targets: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute per-class accuracy."""
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    classes = np.unique(all_targets)

    result = {}
    for cls in classes:
        mask = all_targets == cls
        cls_acc = (all_preds[mask] == all_targets[mask]).mean() * 100
        label = class_names[cls] if class_names else str(cls)
        result[label] = round(float(cls_acc), 2)

    return result


def confusion_matrix(
    all_preds: List[int],
    all_targets: List[int],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        matrix[t][p] += 1
    return matrix
