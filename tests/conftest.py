"""
Shared fixtures used across all test modules.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image


# ---------------------------------------------------------------------------
# Image / dataset helpers
# ---------------------------------------------------------------------------

def make_image(path: Path, size=(32, 32), color=(100, 150, 200)):
    img = Image.new("RGB", size, color=color)
    img.save(path)


def make_class_dir(root: Path, class_name: str, n: int = 5, size=(32, 32)):
    cls_dir = root / class_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        make_image(cls_dir / f"img_{i:03d}.jpg", size=size, color=(i * 30 % 255, i * 50 % 255, i * 70 % 255))
    return cls_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def dummy_dataset_dir(tmp_path):
    """Dataset with 3 classes, 6 images each."""
    for cls in ["cat", "dog", "bird"]:
        make_class_dir(tmp_path, cls, n=6)
    return tmp_path


@pytest.fixture
def dummy_dataset_dir_small(tmp_path):
    """Dataset with 2 classes, 4 images each."""
    for cls in ["class_a", "class_b"]:
        make_class_dir(tmp_path, cls, n=4)
    return tmp_path


@pytest.fixture
def single_image(tmp_path):
    path = tmp_path / "test_image.jpg"
    make_image(path)
    return path


@pytest.fixture
def batch_input_32():
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def batch_input_224():
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def resnet_config(tmp_path):
    cfg = {
        "model": {"name": "simple_resnet", "num_classes": 3, "num_blocks": [1, 1, 1, 1]},
        "data": {
            "train_dir": str(tmp_path / "train"),
            "val_dir": str(tmp_path / "val"),
            "test_dir": str(tmp_path / "test"),
            "image_size": 32,
            "normalize": "cifar",
            "num_workers": 0,
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "lr_scheduler": "cosine",
        },
        "checkpoint": {"save_dir": str(tmp_path / "checkpoints"), "save_every": 1},
        "logging": {"log_file": None},
    }
    config_path = tmp_path / "resnet_test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return cfg, str(config_path)


@pytest.fixture
def vgg_config(tmp_path):
    cfg = {
        "model": {"name": "vgg", "num_classes": 3, "variant": "vgg11", "batch_norm": True, "dropout": 0.0},
        "data": {
            "train_dir": str(tmp_path / "train"),
            "val_dir": str(tmp_path / "val"),
            "test_dir": str(tmp_path / "test"),
            "image_size": 224,
            "normalize": "imagenet",
            "num_workers": 0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "lr_scheduler": "step",
            "lr_step_size": 1,
            "lr_gamma": 0.1,
        },
        "checkpoint": {"save_dir": str(tmp_path / "checkpoints"), "save_every": 1},
        "logging": {"log_file": None},
    }
    config_path = tmp_path / "vgg_test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return cfg, str(config_path)
