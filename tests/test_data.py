import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from data.dataset import ImageFolderDataset
from data.transforms import get_train_transforms, get_val_transforms, _get_norm_stats
from data.split import split_dataset, get_class_names


def create_dummy_dataset(root: Path, classes: list, n_per_class: int = 5):
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(n_per_class):
            img = Image.new("RGB", (32, 32), color=(i * 10, i * 20, i * 30))
            img.save(cls_dir / f"img_{i}.jpg")


class TestImageFolderDataset:
    def test_loads_images(self, tmp_path):
        create_dummy_dataset(tmp_path, ["cat", "dog"], n_per_class=3)
        ds = ImageFolderDataset(str(tmp_path))
        assert len(ds) == 6

    def test_class_discovery(self, tmp_path):
        create_dummy_dataset(tmp_path, ["a", "b", "c"])
        ds = ImageFolderDataset(str(tmp_path))
        assert ds.classes == ["a", "b", "c"]
        assert ds.class_to_idx == {"a": 0, "b": 1, "c": 2}

    def test_with_transforms(self, tmp_path):
        import torch
        create_dummy_dataset(tmp_path, ["cat", "dog"])
        tf = get_val_transforms(image_size=32, normalize="cifar")
        ds = ImageFolderDataset(str(tmp_path), transform=tf)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)

    def test_class_counts(self, tmp_path):
        create_dummy_dataset(tmp_path, ["x", "y"], n_per_class=4)
        ds = ImageFolderDataset(str(tmp_path))
        counts = ds.class_counts()
        assert counts == {"x": 4, "y": 4}

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ImageFolderDataset(str(tmp_path))


class TestTransforms:
    def test_train_transform_runs(self):
        tf = get_train_transforms(image_size=64)
        img = Image.new("RGB", (128, 128))
        result = tf(img)
        assert result.shape == (3, 64, 64)

    def test_val_transform_runs(self):
        tf = get_val_transforms(image_size=64)
        img = Image.new("RGB", (128, 128))
        result = tf(img)
        assert result.shape == (3, 64, 64)

    def test_invalid_normalize_preset(self):
        with pytest.raises(ValueError):
            _get_norm_stats("unknown_preset")


class TestSplit:
    def test_split_creates_dirs(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        create_dummy_dataset(src, ["cat", "dog"], n_per_class=10)
        split_dataset(str(src), str(dst), train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert (dst / "train").exists()
        assert (dst / "val").exists()
        assert (dst / "test").exists()

    def test_split_counts_sum(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        create_dummy_dataset(src, ["a", "b"], n_per_class=10)
        train_n, val_n, test_n = split_dataset(str(src), str(dst))
        assert train_n + val_n + test_n == 20
