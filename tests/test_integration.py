"""
Integration tests — run a minimal end-to-end training and validation cycle.
No real data required; uses synthetic TensorDataset.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.resnet import SimpleResNet
from utils.checkpoint import save_checkpoint, load_checkpoint, build_checkpoint
from utils.logger import TrainingLogger
from utils.metrics import AverageMeter, accuracy
from train import train_one_epoch, validate, build_scheduler


def make_synthetic_loader(n=64, num_classes=3, img_size=32, batch_size=16):
    images = torch.randn(n, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)


@pytest.fixture
def mini_setup():
    model = SimpleResNet(num_classes=3, num_blocks=[1, 1, 1, 1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cpu")
    train_loader = make_synthetic_loader()
    val_loader = make_synthetic_loader(n=32)
    return model, criterion, optimizer, device, train_loader, val_loader


class TestEndToEndTraining:
    def test_full_training_loop(self, mini_setup):
        model, criterion, optimizer, device, train_loader, val_loader = mini_setup
        cfg = {"training": {"epochs": 3, "lr_scheduler": "cosine"}}
        scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

        logger = TrainingLogger()
        best_acc = 0.0

        for epoch in range(1, 4):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()
            logger.log_epoch(epoch, 3, train_loss, train_acc, val_loss, val_acc)
            best_acc = max(best_acc, val_acc)

        assert len(logger.history["train_loss"]) == 3
        assert 0.0 <= best_acc <= 100.0

    def test_checkpoint_save_and_resume(self, mini_setup, tmp_path):
        model, criterion, optimizer, device, train_loader, val_loader = mini_setup

        # Train for 1 epoch
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Save
        state = build_checkpoint(model, optimizer, epoch=1, best_acc=val_acc, config={})
        save_checkpoint(state, str(tmp_path), "ckpt.pth", is_best=True)

        # Load into fresh model
        model2 = SimpleResNet(num_classes=3, num_blocks=[1, 1, 1, 1])
        result = load_checkpoint(str(tmp_path / "ckpt.pth"), model2, device="cpu")

        # Verify same val accuracy
        model2.eval()
        val_loss2, val_acc2 = validate(model2, val_loader, criterion, device)
        assert val_acc2 == pytest.approx(val_acc, rel=1e-4)

    def test_metrics_tracked_correctly(self, mini_setup):
        model, criterion, optimizer, device, train_loader, val_loader = mini_setup
        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("acc")

        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                top1 = accuracy(outputs, targets)[0]
                loss_meter.update(loss.item(), images.size(0))
                acc_meter.update(top1, images.size(0))

        assert loss_meter.count == 32
        assert loss_meter.avg > 0
        assert 0.0 <= acc_meter.avg <= 100.0

    def test_loss_can_decrease_over_epochs(self):
        """With a tiny overfit-prone dataset, loss should decrease with enough epochs."""
        torch.manual_seed(0)
        model = SimpleResNet(num_classes=2, num_blocks=[1, 1, 1, 1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        device = torch.device("cpu")

        # Tiny dataset: 8 samples, easy to overfit
        images = torch.randn(8, 3, 32, 32)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=8)

        losses = []
        for _ in range(10):
            loss, _ = train_one_epoch(model, loader, criterion, optimizer, device)
            losses.append(loss)

        # Loss at end should be lower than at start
        assert losses[-1] < losses[0]
