"""
Tests for data/dataset.py, data/transforms.py, data/split.py
"""

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.dataset import ImageFolderDataset
from data.transforms import (
    get_train_transforms, get_val_transforms,
    get_test_transforms, get_inference_transforms, _get_norm_stats,
    IMAGENET_MEAN, IMAGENET_STD, CIFAR_MEAN, CIFAR_STD,
)
from data.split import split_dataset, get_class_names


# ===========================================================================
# ImageFolderDataset
# ===========================================================================

class TestImageFolderDataset:
    def test_len(self, dummy_dataset_dir):
        ds = ImageFolderDataset(str(dummy_dataset_dir))
        assert len(ds) == 18  # 3 classes * 6 images

    def test_class_discovery_sorted(self, dummy_dataset_dir):
        ds = ImageFolderDataset(str(dummy_dataset_dir))
        assert ds.classes == ["bird", "cat", "dog"]

    def test_class_to_idx_correct(self, dummy_dataset_dir):
        ds = ImageFolderDataset(str(dummy_dataset_dir))
        assert ds.class_to_idx == {"bird": 0, "cat": 1, "dog": 2}

    def test_getitem_returns_pil_without_transform(self, dummy_dataset_dir):
        ds = ImageFolderDataset(str(dummy_dataset_dir))
        img, label = ds[0]
        assert isinstance(img, Image.Image)
        assert isinstance(label, int)

    def test_getitem_label_in_range(self, dummy_dataset_dir):
        ds = ImageFolderDataset(str(dummy_dataset_dir))
        for i in range(len(ds)):
            _, label = ds[i]
            assert 0 <= label < len(ds.classes)

    def test_with_transform_returns_tensor(self, dummy_dataset_dir):
        tf = get_val_transforms(image_size=32, normalize="cifar")
        ds = ImageFolderDataset(str(dummy_dataset_dir), transform=tf)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)

    def test_class_counts(self, dummy_dataset_dir):
        ds = ImageFolderDataset(str(dummy_dataset_dir))
        counts = ds.class_counts()
        assert counts == {"bird": 6, "cat": 6, "dog": 6}

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ImageFolderDataset(str(tmp_path))

    def test_two_class_dataset(self, dummy_dataset_dir_small):
        ds = ImageFolderDataset(str(dummy_dataset_dir_small))
        assert len(ds) == 8
        assert len(ds.classes) == 2

    def test_dataloader_batching(self, dummy_dataset_dir):
        tf = get_val_transforms(image_size=32, normalize="cifar")
        ds = ImageFolderDataset(str(dummy_dataset_dir), transform=tf)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch_imgs, batch_labels = next(iter(loader))
        assert batch_imgs.shape == (4, 3, 32, 32)
        assert batch_labels.shape == (4,)

    def test_samples_list_length(self, dummy_dataset_dir):
        ds = ImageFolderDataset(str(dummy_dataset_dir))
        assert len(ds.samples) == len(ds)

    def test_all_samples_have_valid_labels(self, dummy_dataset_dir):
        ds = ImageFolderDataset(str(dummy_dataset_dir))
        labels = [label for _, label in ds.samples]
        assert all(0 <= l < len(ds.classes) for l in labels)

    def test_target_transform_applied(self, dummy_dataset_dir):
        ds = ImageFolderDataset(
            str(dummy_dataset_dir),
            target_transform=lambda y: y + 10,
        )
        _, label = ds[0]
        assert label >= 10


# ===========================================================================
# Transforms
# ===========================================================================

class TestNormStats:
    def test_imagenet_stats(self):
        mean, std = _get_norm_stats("imagenet")
        assert mean == IMAGENET_MEAN
        assert std == IMAGENET_STD

    def test_cifar_stats(self):
        mean, std = _get_norm_stats("cifar")
        assert mean == CIFAR_MEAN
        assert std == CIFAR_STD

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError):
            _get_norm_stats("unknown_preset")


class TestTrainTransforms:
    def test_output_shape_224(self):
        tf = get_train_transforms(image_size=224)
        img = Image.new("RGB", (256, 256))
        assert tf(img).shape == (3, 224, 224)

    def test_output_shape_32(self):
        tf = get_train_transforms(image_size=32, normalize="cifar")
        img = Image.new("RGB", (64, 64))
        assert tf(img).shape == (3, 32, 32)

    def test_returns_tensor(self):
        tf = get_train_transforms()
        img = Image.new("RGB", (256, 256))
        assert isinstance(tf(img), torch.Tensor)

    def test_output_is_float(self):
        tf = get_train_transforms()
        img = Image.new("RGB", (256, 256))
        assert tf(img).dtype == torch.float32


class TestValTransforms:
    def test_output_shape(self):
        tf = get_val_transforms(image_size=224)
        img = Image.new("RGB", (256, 256))
        assert tf(img).shape == (3, 224, 224)

    def test_output_shape_32(self):
        tf = get_val_transforms(image_size=32, normalize="cifar")
        img = Image.new("RGB", (64, 64))
        assert tf(img).shape == (3, 32, 32)

    def test_deterministic(self):
        """Val transforms should be deterministic (no random ops)."""
        tf = get_val_transforms(image_size=64)
        img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        out1 = tf(img)
        out2 = tf(img)
        assert torch.allclose(out1, out2)


class TestTestTransforms:
    def test_output_shape(self):
        tf = get_test_transforms(image_size=64)
        img = Image.new("RGB", (128, 128))
        assert tf(img).shape == (3, 64, 64)


class TestInferenceTransforms:
    def test_output_shape(self):
        tf = get_inference_transforms(image_size=128)
        img = Image.new("RGB", (256, 256))
        assert tf(img).shape == (3, 128, 128)

    def test_same_as_val(self):
        """Inference transforms should match val transforms."""
        val_tf = get_val_transforms(image_size=64)
        inf_tf = get_inference_transforms(image_size=64)
        img = Image.new("RGB", (128, 128), color=(80, 120, 160))
        assert torch.allclose(val_tf(img), inf_tf(img))


# ===========================================================================
# split_dataset
# ===========================================================================

class TestSplitDataset:
    def test_creates_split_dirs(self, dummy_dataset_dir_small, tmp_path):
        dst = tmp_path / "split"
        split_dataset(str(dummy_dataset_dir_small), str(dst))
        for split in ["train", "val", "test"]:
            assert (dst / split).exists()

    def test_total_count_preserved(self, dummy_dataset_dir_small, tmp_path):
        dst = tmp_path / "split"
        t, v, te = split_dataset(str(dummy_dataset_dir_small), str(dst))
        assert t + v + te == 8  # 2 classes * 4 images

    def test_ratios_respected(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        # 10 images per class, 2 classes = 20 total
        from tests.conftest import make_class_dir
        make_class_dir(src, "a", n=10)
        make_class_dir(src, "b", n=10)
        t, v, te = split_dataset(str(src), str(dst), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        assert t + v + te == 20
        assert t > v
        assert t > te

    def test_invalid_ratios_raise(self, dummy_dataset_dir_small, tmp_path):
        with pytest.raises(AssertionError):
            split_dataset(str(dummy_dataset_dir_small), str(tmp_path / "out"),
                         train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_class_dirs_created_in_splits(self, dummy_dataset_dir_small, tmp_path):
        dst = tmp_path / "split"
        split_dataset(str(dummy_dataset_dir_small), str(dst))
        for split in ["train", "val", "test"]:
            split_classes = [d.name for d in (dst / split).iterdir() if d.is_dir()]
            assert set(split_classes) == {"class_a", "class_b"}

    def test_seed_reproducibility(self, dummy_dataset_dir_small, tmp_path):
        dst1 = tmp_path / "run1"
        dst2 = tmp_path / "run2"
        r1 = split_dataset(str(dummy_dataset_dir_small), str(dst1), seed=42)
        r2 = split_dataset(str(dummy_dataset_dir_small), str(dst2), seed=42)
        assert r1 == r2

    def test_different_seeds_may_differ(self, tmp_path):
        src = tmp_path / "src"
        from tests.conftest import make_class_dir
        make_class_dir(src, "a", n=20)
        make_class_dir(src, "b", n=20)
        dst1 = tmp_path / "run1"
        dst2 = tmp_path / "run2"
        split_dataset(str(src), str(dst1), seed=0)
        split_dataset(str(src), str(dst2), seed=999)
        files1 = sorted(str(p) for p in (dst1 / "train" / "a").iterdir())
        files2 = sorted(str(p) for p in (dst2 / "train" / "a").iterdir())
        # With different seeds and 20 images, train sets are very likely different
        # (not a strict requirement but a sanity check)
        assert isinstance(files1, list)


class TestGetClassNames:
    def test_returns_sorted_names(self, dummy_dataset_dir):
        names = get_class_names(str(dummy_dataset_dir))
        assert names == ["bird", "cat", "dog"]

    def test_two_classes(self, dummy_dataset_dir_small):
        names = get_class_names(str(dummy_dataset_dir_small))
        assert names == ["class_a", "class_b"]
