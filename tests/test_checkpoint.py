"""
Tests for utils/checkpoint.py
"""

import os
import pytest
import torch
import torch.nn as nn

from utils.checkpoint import save_checkpoint, load_checkpoint, build_checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def tiny_model():
    return TinyModel()


@pytest.fixture
def tiny_optimizer(tiny_model):
    return torch.optim.SGD(tiny_model.parameters(), lr=0.01)


@pytest.fixture
def sample_state(tiny_model, tiny_optimizer):
    return build_checkpoint(tiny_model, tiny_optimizer, epoch=5, best_acc=92.3, config={"lr": 0.01})


# ===========================================================================
# build_checkpoint
# ===========================================================================

class TestBuildCheckpoint:
    def test_contains_required_keys(self, sample_state):
        for key in ["epoch", "model_state_dict", "optimizer_state_dict", "best_acc", "config"]:
            assert key in sample_state

    def test_epoch_stored(self, sample_state):
        assert sample_state["epoch"] == 5

    def test_best_acc_stored(self, sample_state):
        assert sample_state["best_acc"] == pytest.approx(92.3)

    def test_config_stored(self, sample_state):
        assert sample_state["config"] == {"lr": 0.01}

    def test_model_state_is_dict(self, sample_state):
        assert isinstance(sample_state["model_state_dict"], dict)


# ===========================================================================
# save_checkpoint
# ===========================================================================

class TestSaveCheckpoint:
    def test_file_created(self, sample_state, tmp_path):
        save_checkpoint(sample_state, str(tmp_path), "ckpt.pth")
        assert (tmp_path / "ckpt.pth").exists()

    def test_creates_dir_if_missing(self, sample_state, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        save_checkpoint(sample_state, str(nested), "ckpt.pth")
        assert (nested / "ckpt.pth").exists()

    def test_best_model_saved_when_is_best(self, sample_state, tmp_path):
        save_checkpoint(sample_state, str(tmp_path), "ckpt.pth", is_best=True)
        assert (tmp_path / "best_model.pth").exists()

    def test_best_model_not_saved_when_not_best(self, sample_state, tmp_path):
        save_checkpoint(sample_state, str(tmp_path), "ckpt.pth", is_best=False)
        assert not (tmp_path / "best_model.pth").exists()

    def test_file_is_loadable(self, sample_state, tmp_path):
        save_checkpoint(sample_state, str(tmp_path), "ckpt.pth")
        loaded = torch.load(str(tmp_path / "ckpt.pth"), map_location="cpu")
        assert loaded["epoch"] == sample_state["epoch"]


# ===========================================================================
# load_checkpoint
# ===========================================================================

class TestLoadCheckpoint:
    def test_loads_model_weights(self, tiny_model, tiny_optimizer, sample_state, tmp_path):
        save_checkpoint(sample_state, str(tmp_path), "ckpt.pth")
        model2 = TinyModel()
        load_checkpoint(str(tmp_path / "ckpt.pth"), model2, device="cpu")
        for p1, p2 in zip(tiny_model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_loads_optimizer_state(self, tiny_model, tiny_optimizer, sample_state, tmp_path):
        save_checkpoint(sample_state, str(tmp_path), "ckpt.pth")
        model2 = TinyModel()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        result = load_checkpoint(str(tmp_path / "ckpt.pth"), model2, optimizer=opt2, device="cpu")
        assert "optimizer_state_dict" in result

    def test_file_not_found_raises(self, tiny_model):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path/model.pth", tiny_model)

    def test_returns_full_checkpoint(self, tiny_model, sample_state, tmp_path):
        save_checkpoint(sample_state, str(tmp_path), "ckpt.pth")
        result = load_checkpoint(str(tmp_path / "ckpt.pth"), tiny_model)
        assert result["epoch"] == 5
        assert result["best_acc"] == pytest.approx(92.3)

    def test_save_load_roundtrip_preserves_output(self, tiny_model, tiny_optimizer, sample_state, tmp_path):
        tiny_model.eval()
        x = torch.randn(2, 4)
        with torch.no_grad():
            out_before = tiny_model(x)
        save_checkpoint(sample_state, str(tmp_path), "ckpt.pth")

        model2 = TinyModel()
        load_checkpoint(str(tmp_path / "ckpt.pth"), model2)
        model2.eval()
        with torch.no_grad():
            out_after = model2(x)
        assert torch.allclose(out_before, out_after)
