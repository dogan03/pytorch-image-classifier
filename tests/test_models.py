import pytest
import torch

from models import get_model
from models.resnet import SimpleResNet, PretrainedResNet
from models.vgg import VGG


@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def dummy_input_224():
    return torch.randn(2, 3, 224, 224)


class TestSimpleResNet:
    def test_output_shape(self, dummy_input):
        model = SimpleResNet(num_classes=10)
        out = model(dummy_input)
        assert out.shape == (2, 10)

    def test_count_parameters(self):
        model = SimpleResNet(num_classes=10)
        assert model.count_parameters() > 0

    def test_custom_num_blocks(self, dummy_input):
        model = SimpleResNet(num_classes=5, num_blocks=[1, 1, 1, 1])
        out = model(dummy_input)
        assert out.shape == (2, 5)


class TestVGG:
    def test_vgg16_output_shape(self, dummy_input_224):
        model = VGG(num_classes=10, variant="vgg16")
        out = model(dummy_input_224)
        assert out.shape == (2, 10)

    def test_invalid_variant(self):
        with pytest.raises(ValueError, match="Unknown VGG variant"):
            VGG(num_classes=10, variant="vgg99")

    def test_parameter_count_positive(self):
        model = VGG(num_classes=10, variant="vgg11")
        assert model.count_parameters() > 0


class TestModelRegistry:
    def test_get_valid_model(self):
        model = get_model("simple_resnet", num_classes=10)
        assert isinstance(model, SimpleResNet)

    def test_get_invalid_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent_model", num_classes=10)
