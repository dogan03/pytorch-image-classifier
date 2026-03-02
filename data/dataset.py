import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """
    Custom image dataset that reads from a directory structured as:
        root/
            class_a/img1.jpg
            class_b/img2.jpg
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self) -> Tuple[List[str], dict]:
        classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        if not classes:
            raise FileNotFoundError(f"No class directories found in {self.root}")
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self) -> List[Tuple[Path, int]]:
        samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            label = self.class_to_idx[cls]
            for fpath in cls_dir.iterdir():
                if fpath.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    samples.append((fpath, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def class_counts(self) -> dict:
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return {self.classes[k]: v for k, v in sorted(counts.items())}
