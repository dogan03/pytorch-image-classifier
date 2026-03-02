import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """Structured logger for training metrics."""

    def __init__(self, log_file: Optional[str] = None):
        self.logger = get_logger("training", log_file)
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def log_epoch(self, epoch: int, total_epochs: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        self.logger.info(
            f"Epoch [{epoch:>3}/{total_epochs}] "
            f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%"
        )

    def log_test(self, test_loss: float, test_acc: float):
        self.logger.info(f"Test Results | Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")

    def log_info(self, msg: str):
        self.logger.info(msg)
