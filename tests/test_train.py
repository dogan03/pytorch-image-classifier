"""
Tests for the training/validation loops extracted from train.py
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train import train_one_epoch, validate, set_seed, get_device, build_scheduler
from models.resnet import SimpleResNet


# ===========================================================================
# Helpers
# ===========================================================================

def make_loader(n_samples=32, n_classes=3, batch_size=8, img_size=32):
    images = torch.randn(n_samples, 3, img_size, img_size)
    labels = torch.randint(0, n_classes, (n_samples,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@pytest.fixture
def simple_model():
    return SimpleResNet(num_classes=3, num_blocks=[1, 1, 1, 1])


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(simple_model):
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def train_loader():
    return make_loader(n_samples=32, n_classes=3)


@pytest.fixture
def val_loader():
    return make_loader(n_samples=16, n_classes=3)


# ===========================================================================
# set_seed
# ===========================================================================

class TestSetSeed:
    def test_seed_produces_reproducible_tensors(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_different_seeds_produce_different_tensors(self):
        set_seed(0)
        a = torch.randn(10)
        set_seed(1)
        b = torch.randn(10)
        assert not torch.allclose(a, b)


# ===========================================================================
# get_device
# ===========================================================================

class TestGetDevice:
    def test_returns_torch_device(self):
        d = get_device()
        assert isinstance(d, torch.device)

    def test_returns_cpu_if_no_gpu(self):
        # On most CI systems, this will be cpu
        d = get_device()
        assert d.type in ["cpu", "cuda", "mps"]


# ===========================================================================
# build_scheduler
# ===========================================================================

class TestBuildScheduler:
    def test_cosine_scheduler(self, simple_model, optimizer):
        cfg = {"training": {"epochs": 10, "lr_scheduler": "cosine"}}
        sched = build_scheduler(optimizer, cfg, steps_per_epoch=10)
        assert sched is not None

    def test_step_scheduler(self, simple_model, optimizer):
        cfg = {"training": {"epochs": 10, "lr_scheduler": "step", "lr_step_size": 3, "lr_gamma": 0.1}}
        sched = build_scheduler(optimizer, cfg, steps_per_epoch=10)
        assert sched is not None

    def test_invalid_scheduler_raises(self, optimizer):
        cfg = {"training": {"epochs": 10, "lr_scheduler": "nonexistent"}}
        with pytest.raises(ValueError):
            build_scheduler(optimizer, cfg, steps_per_epoch=10)

    def test_cosine_lr_decreases(self, simple_model, optimizer):
        cfg = {"training": {"epochs": 5, "lr_scheduler": "cosine"}}
        sched = build_scheduler(optimizer, cfg, steps_per_epoch=10)
        lr_before = optimizer.param_groups[0]["lr"]
        for _ in range(3):
            sched.step()
        lr_after = optimizer.param_groups[0]["lr"]
        assert lr_after < lr_before


# ===========================================================================
# train_one_epoch
# ===========================================================================

class TestTrainOneEpoch:
    def test_returns_two_floats(self, simple_model, train_loader, criterion, optimizer):
        device = torch.device("cpu")
        result = train_one_epoch(simple_model, train_loader, criterion, optimizer, device)
        assert len(result) == 2
        loss, acc = result
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_loss_is_positive(self, simple_model, train_loader, criterion, optimizer):
        device = torch.device("cpu")
        loss, _ = train_one_epoch(simple_model, train_loader, criterion, optimizer, device)
        assert loss > 0

    def test_accuracy_in_valid_range(self, simple_model, train_loader, criterion, optimizer):
        device = torch.device("cpu")
        _, acc = train_one_epoch(simple_model, train_loader, criterion, optimizer, device)
        assert 0.0 <= acc <= 100.0

    def test_weights_change_after_epoch(self, simple_model, train_loader, criterion, optimizer):
        device = torch.device("cpu")
        before = simple_model.fc.weight.data.clone()
        train_one_epoch(simple_model, train_loader, criterion, optimizer, device)
        after = simple_model.fc.weight.data
        assert not torch.allclose(before, after)

    def test_model_in_train_mode_after(self, simple_model, train_loader, criterion, optimizer):
        device = torch.device("cpu")
        simple_model.eval()
        train_one_epoch(simple_model, train_loader, criterion, optimizer, device)
        assert simple_model.training


# ===========================================================================
# validate
# ===========================================================================

class TestValidate:
    def test_returns_two_floats(self, simple_model, val_loader, criterion):
        device = torch.device("cpu")
        result = validate(simple_model, val_loader, criterion, device)
        assert len(result) == 2
        loss, acc = result
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_loss_positive(self, simple_model, val_loader, criterion):
        device = torch.device("cpu")
        loss, _ = validate(simple_model, val_loader, criterion, device)
        assert loss > 0

    def test_accuracy_in_valid_range(self, simple_model, val_loader, criterion):
        device = torch.device("cpu")
        _, acc = validate(simple_model, val_loader, criterion, device)
        assert 0.0 <= acc <= 100.0

    def test_weights_unchanged_during_validate(self, simple_model, val_loader, criterion):
        device = torch.device("cpu")
        before = simple_model.fc.weight.data.clone()
        validate(simple_model, val_loader, criterion, device)
        after = simple_model.fc.weight.data
        assert torch.allclose(before, after)

    def test_model_eval_mode_used(self, simple_model, val_loader, criterion):
        """Model should produce deterministic results when called twice."""
        device = torch.device("cpu")
        loss1, acc1 = validate(simple_model, val_loader, criterion, device)
        loss2, acc2 = validate(simple_model, val_loader, criterion, device)
        assert loss1 == pytest.approx(loss2, rel=1e-5)
        assert acc1 == pytest.approx(acc2, rel=1e-5)

    def test_perfect_model_has_high_accuracy(self):
        """A model that always predicts class 0 on all-class-0 data should get 100%."""
        device = torch.device("cpu")
        n = 16
        images = torch.randn(n, 3, 32, 32)
        labels = torch.zeros(n, dtype=torch.long)
        ds = TensorDataset(images, labels)
        loader = DataLoader(ds, batch_size=8)

        # Build a model whose output always favors class 0
        model = SimpleResNet(num_classes=3, num_blocks=[1, 1, 1, 1])
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.bias[0] = 100.0  # always predict class 0

        criterion = nn.CrossEntropyLoss()
        _, acc = validate(model, loader, criterion, device)
        assert acc == pytest.approx(100.0)
