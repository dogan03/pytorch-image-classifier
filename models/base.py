import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
        return self
