import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    """Save a training checkpoint."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / filename
    torch.save(state, filepath)

    if is_best:
        best_path = path / "best_model.pth"
        torch.save(state, best_path)
        print(f"  [Checkpoint] New best model saved to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint into a model and optionally an optimizer."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"  [Checkpoint] Loaded from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


def build_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    config: dict,
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "config": config,
    }
