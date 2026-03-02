import random
from pathlib import Path
from typing import Tuple, List
import shutil


def split_dataset(
    src_dir: str,
    dst_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """
    Split an image folder dataset into train/val/test directories.

    Args:
        src_dir: Source directory with class subdirectories.
        dst_dir: Destination root directory.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_count, val_count, test_count).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    src = Path(src_dir)
    dst = Path(dst_dir)
    random.seed(seed)

    splits = ["train", "val", "test"]
    for split in splits:
        (dst / split).mkdir(parents=True, exist_ok=True)

    counts = [0, 0, 0]

    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue

        images = [f for f in cls_dir.iterdir() if f.is_file()]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        buckets = [
            images[:n_train],
            images[n_train:n_train + n_val],
            images[n_train + n_val:],
        ]

        for split, bucket, i in zip(splits, buckets, range(3)):
            split_cls_dir = dst / split / cls_dir.name
            split_cls_dir.mkdir(parents=True, exist_ok=True)
            for img in bucket:
                shutil.copy2(img, split_cls_dir / img.name)
            counts[i] += len(bucket)

    return tuple(counts)


def get_class_names(data_dir: str) -> List[str]:
    return sorted(d.name for d in Path(data_dir).iterdir() if d.is_dir())
