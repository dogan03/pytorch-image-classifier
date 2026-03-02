"""
Tests for config loading in train.py and structure of config files
"""

import pytest
import yaml

from train import load_config


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        cfg_data = {"model": {"name": "simple_resnet"}, "training": {"epochs": 10}}
        path = tmp_path / "cfg.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg_data, f)
        cfg = load_config(str(path))
        assert cfg["model"]["name"] == "simple_resnet"
        assert cfg["training"]["epochs"] == 10

    def test_returns_dict(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        with open(path, "w") as f:
            yaml.dump({"key": "value"}, f)
        assert isinstance(load_config(str(path)), dict)

    def test_nested_keys(self, tmp_path):
        cfg_data = {"a": {"b": {"c": 42}}}
        path = tmp_path / "nested.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg_data, f)
        cfg = load_config(str(path))
        assert cfg["a"]["b"]["c"] == 42

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestResnetConfigFile:
    def test_resnet_config_loads(self):
        cfg = load_config("configs/resnet.yaml")
        assert "model" in cfg
        assert "training" in cfg
        assert "data" in cfg

    def test_resnet_model_name(self):
        cfg = load_config("configs/resnet.yaml")
        assert cfg["model"]["name"] == "simple_resnet"

    def test_resnet_has_num_classes(self):
        cfg = load_config("configs/resnet.yaml")
        assert "num_classes" in cfg["model"]
        assert cfg["model"]["num_classes"] > 0

    def test_resnet_training_has_required_keys(self):
        cfg = load_config("configs/resnet.yaml")
        for key in ["epochs", "batch_size", "learning_rate", "momentum", "weight_decay"]:
            assert key in cfg["training"]

    def test_resnet_data_has_required_keys(self):
        cfg = load_config("configs/resnet.yaml")
        for key in ["train_dir", "val_dir", "test_dir", "image_size", "normalize"]:
            assert key in cfg["data"]

    def test_resnet_learning_rate_positive(self):
        cfg = load_config("configs/resnet.yaml")
        assert cfg["training"]["learning_rate"] > 0

    def test_resnet_batch_size_positive(self):
        cfg = load_config("configs/resnet.yaml")
        assert cfg["training"]["batch_size"] > 0


class TestVGGConfigFile:
    def test_vgg_config_loads(self):
        cfg = load_config("configs/vgg.yaml")
        assert "model" in cfg

    def test_vgg_model_name(self):
        cfg = load_config("configs/vgg.yaml")
        assert cfg["model"]["name"] == "vgg"

    def test_vgg_has_variant(self):
        cfg = load_config("configs/vgg.yaml")
        assert "variant" in cfg["model"]
        assert cfg["model"]["variant"].startswith("vgg")

    def test_vgg_training_keys(self):
        cfg = load_config("configs/vgg.yaml")
        for key in ["epochs", "batch_size", "learning_rate"]:
            assert key in cfg["training"]

    def test_vgg_normalize_is_valid(self):
        cfg = load_config("configs/vgg.yaml")
        assert cfg["data"]["normalize"] in ["imagenet", "cifar"]
