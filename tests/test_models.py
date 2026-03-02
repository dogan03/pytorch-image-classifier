"""
Tests for models/base.py, models/resnet.py, models/vgg.py, models/__init__.py
"""

import os
import pytest
import torch
import torch.nn as nn

from models.base import BaseModel
from models.resnet import SimpleResNet, PretrainedResNet, ResidualBlock
from models.vgg import VGG, VGG_CONFIGS, _make_layers
from models import get_model, MODEL_REGISTRY


# ===========================================================================
# BaseModel
# ===========================================================================

class ConcreteModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(x)


class TestBaseModel:
    def test_num_classes_stored(self):
        m = ConcreteModel(7)
        assert m.num_classes == 7

    def test_count_parameters(self):
        m = ConcreteModel(5)
        expected = 4 * 5 + 5
        assert m.count_parameters() == expected

    def test_count_parameters_zero_for_frozen(self):
        m = ConcreteModel(3)
        for p in m.parameters():
            p.requires_grad = False
        assert m.count_parameters() == 0

    def test_save_and_load(self, tmp_path):
        m = ConcreteModel(4)
        path = str(tmp_path / "model.pth")
        m.save(path)
        assert os.path.exists(path)
        m2 = ConcreteModel(4)
        m2.load(path)
        for p1, p2 in zip(m.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_forward_output_shape(self):
        m = ConcreteModel(6)
        x = torch.randn(3, 4)
        out = m(x)
        assert out.shape == (3, 6)

    def test_is_nn_module(self):
        m = ConcreteModel(3)
        assert isinstance(m, nn.Module)

    def test_abstract_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseModel(10)


# ===========================================================================
# ResidualBlock
# ===========================================================================

class TestResidualBlock:
    def test_same_channels_output_shape(self):
        block = ResidualBlock(64, 64, stride=1)
        x = torch.randn(2, 64, 8, 8)
        assert block(x).shape == (2, 64, 8, 8)

    def test_different_channels_output_shape(self):
        block = ResidualBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 8, 8)
        assert block(x).shape == (2, 128, 4, 4)

    def test_shortcut_identity_when_same_dims(self):
        block = ResidualBlock(32, 32, stride=1)
        assert len(block.shortcut) == 0

    def test_shortcut_has_layers_when_dims_differ(self):
        block = ResidualBlock(32, 64, stride=2)
        assert len(block.shortcut) > 0

    def test_output_is_finite(self):
        block = ResidualBlock(16, 16)
        x = torch.randn(1, 16, 4, 4)
        assert torch.isfinite(block(x)).all()


# ===========================================================================
# SimpleResNet
# ===========================================================================

class TestSimpleResNet:
    def test_default_output_shape(self, batch_input_32):
        model = SimpleResNet(num_classes=10)
        assert model(batch_input_32).shape == (4, 10)

    def test_custom_num_classes(self, batch_input_32):
        for n in [2, 5, 100]:
            model = SimpleResNet(num_classes=n)
            assert model(batch_input_32).shape == (4, n)

    def test_custom_num_blocks(self, batch_input_32):
        model = SimpleResNet(num_classes=10, num_blocks=[1, 1, 1, 1])
        assert model(batch_input_32).shape == (4, 10)

    def test_deeper_blocks(self, batch_input_32):
        model = SimpleResNet(num_classes=10, num_blocks=[3, 4, 6, 3])
        assert model(batch_input_32).shape == (4, 10)

    def test_parameter_count_positive(self):
        assert SimpleResNet(num_classes=10).count_parameters() > 0

    def test_output_is_finite(self, batch_input_32):
        model = SimpleResNet(num_classes=10)
        assert torch.isfinite(model(batch_input_32)).all()

    def test_gradients_flow(self, batch_input_32):
        model = SimpleResNet(num_classes=10)
        model(batch_input_32).sum().backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_train_eval_modes(self, batch_input_32):
        model = SimpleResNet(num_classes=10)
        model.train()
        assert model.training
        model.eval()
        assert not model.training

    def test_batch_size_one(self):
        model = SimpleResNet(num_classes=5)
        model.eval()
        with torch.no_grad():
            assert model(torch.randn(1, 3, 32, 32)).shape == (1, 5)

    def test_save_load_preserves_output(self, tmp_path, batch_input_32):
        model = SimpleResNet(num_classes=10)
        model.eval()
        with torch.no_grad():
            out_before = model(batch_input_32)
        path = str(tmp_path / "resnet.pth")
        model.save(path)
        model2 = SimpleResNet(num_classes=10)
        model2.load(path)
        model2.eval()
        with torch.no_grad():
            out_after = model2(batch_input_32)
        assert torch.allclose(out_before, out_after)

    def test_different_input_sizes(self):
        model = SimpleResNet(num_classes=10)
        model.eval()
        for size in [16, 32, 64]:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 10)


# ===========================================================================
# VGG
# ===========================================================================

class TestVGGConfigs:
    def test_all_variants_defined(self):
        for v in ["vgg11", "vgg13", "vgg16", "vgg19"]:
            assert v in VGG_CONFIGS

    def test_configs_contain_maxpool(self):
        for cfg in VGG_CONFIGS.values():
            assert "M" in cfg

    def test_make_layers_output_is_sequential(self):
        assert isinstance(_make_layers(VGG_CONFIGS["vgg11"], batch_norm=True), nn.Sequential)

    def test_make_layers_no_batchnorm(self):
        layers = _make_layers(VGG_CONFIGS["vgg11"], batch_norm=False)
        assert not any(isinstance(m, nn.BatchNorm2d) for m in layers)

    def test_make_layers_with_batchnorm(self):
        layers = _make_layers(VGG_CONFIGS["vgg11"], batch_norm=True)
        assert any(isinstance(m, nn.BatchNorm2d) for m in layers)


class TestVGG:
    def test_vgg11_output_shape(self, batch_input_224):
        model = VGG(num_classes=10, variant="vgg11")
        model.eval()
        with torch.no_grad():
            assert model(batch_input_224).shape == (2, 10)

    def test_vgg16_output_shape(self, batch_input_224):
        model = VGG(num_classes=10, variant="vgg16")
        model.eval()
        with torch.no_grad():
            assert model(batch_input_224).shape == (2, 10)

    def test_custom_num_classes(self, batch_input_224):
        model = VGG(num_classes=3, variant="vgg11")
        model.eval()
        with torch.no_grad():
            assert model(batch_input_224).shape == (2, 3)

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown VGG variant"):
            VGG(num_classes=10, variant="vgg99")

    def test_parameter_count_positive(self):
        assert VGG(num_classes=10, variant="vgg11").count_parameters() > 0

    def test_no_batchnorm_option(self, batch_input_224):
        model = VGG(num_classes=5, variant="vgg11", batch_norm=False)
        model.eval()
        with torch.no_grad():
            assert model(batch_input_224).shape == (2, 5)

    def test_output_finite(self, batch_input_224):
        model = VGG(num_classes=10, variant="vgg11")
        model.eval()
        with torch.no_grad():
            assert torch.isfinite(model(batch_input_224)).all()

    def test_gradients_flow(self, batch_input_224):
        model = VGG(num_classes=10, variant="vgg11")
        model(batch_input_224).sum().backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_save_load_preserves_output(self, tmp_path, batch_input_224):
        model = VGG(num_classes=5, variant="vgg11")
        model.eval()
        with torch.no_grad():
            out_before = model(batch_input_224)
        path = str(tmp_path / "vgg.pth")
        model.save(path)
        model2 = VGG(num_classes=5, variant="vgg11")
        model2.load(path)
        model2.eval()
        with torch.no_grad():
            out_after = model2(batch_input_224)
        assert torch.allclose(out_before, out_after)


# ===========================================================================
# Model Registry
# ===========================================================================

class TestModelRegistry:
    def test_registry_has_expected_keys(self):
        for key in ["simple_resnet", "pretrained_resnet", "vgg"]:
            assert key in MODEL_REGISTRY

    def test_get_simple_resnet(self):
        assert isinstance(get_model("simple_resnet", num_classes=10), SimpleResNet)

    def test_get_vgg(self):
        assert isinstance(get_model("vgg", num_classes=5, variant="vgg11"), VGG)

    def test_get_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent", num_classes=10)

    def test_get_model_passes_kwargs(self):
        model = get_model("simple_resnet", num_classes=7, num_blocks=[1, 1, 1, 1])
        assert model.num_classes == 7

    def test_get_vgg_with_classes(self):
        assert get_model("vgg", num_classes=20, variant="vgg11").num_classes == 20
