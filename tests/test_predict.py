"""
Tests for predict.py inference logic
"""

import json
import pytest
import torch
from PIL import Image
from pathlib import Path

from data.transforms import get_inference_transforms
from predict import predict_single
from models.resnet import SimpleResNet


@pytest.fixture
def small_model():
    model = SimpleResNet(num_classes=3, num_blocks=[1, 1, 1, 1])
    model.eval()
    return model


@pytest.fixture
def transform():
    return get_inference_transforms(image_size=32, normalize="cifar")


@pytest.fixture
def class_names():
    return ["cat", "dog", "bird"]


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(120, 80, 200)).save(path)
    return path


# ===========================================================================
# predict_single
# ===========================================================================

class TestPredictSingle:
    def test_returns_dict(self, small_model, sample_image, transform, class_names):
        result = predict_single(small_model, sample_image, transform, class_names, device=torch.device("cpu"))
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, small_model, sample_image, transform, class_names):
        result = predict_single(small_model, sample_image, transform, class_names, device=torch.device("cpu"))
        for key in ["file", "prediction", "confidence", "top5"]:
            assert key in result

    def test_prediction_is_valid_class(self, small_model, sample_image, transform, class_names):
        result = predict_single(small_model, sample_image, transform, class_names, device=torch.device("cpu"))
        assert result["prediction"] in class_names

    def test_confidence_in_range(self, small_model, sample_image, transform, class_names):
        result = predict_single(small_model, sample_image, transform, class_names, device=torch.device("cpu"))
        assert 0.0 <= result["confidence"] <= 100.0

    def test_top5_length(self, small_model, sample_image, transform, class_names):
        result = predict_single(small_model, sample_image, transform, class_names, device=torch.device("cpu"))
        # With 3 classes, top5 is min(5, 3) = 3
        assert len(result["top5"]) == 3

    def test_top5_probabilities_sum_to_100(self, small_model, sample_image, transform, class_names):
        result = predict_single(small_model, sample_image, transform, class_names, device=torch.device("cpu"))
        total = sum(item["prob"] for item in result["top5"])
        assert total == pytest.approx(100.0, abs=0.1)

    def test_top5_classes_are_valid(self, small_model, sample_image, transform, class_names):
        result = predict_single(small_model, sample_image, transform, class_names, device=torch.device("cpu"))
        for item in result["top5"]:
            assert item["class"] in class_names

    def test_file_field_matches_input(self, small_model, sample_image, transform, class_names):
        result = predict_single(small_model, sample_image, transform, class_names, device=torch.device("cpu"))
        assert str(sample_image) in result["file"]

    def test_top_prediction_has_highest_prob(self, small_model, sample_image, transform, class_names):
        result = predict_single(small_model, sample_image, transform, class_names, device=torch.device("cpu"))
        top_prob = result["confidence"]
        for item in result["top5"]:
            assert top_prob >= item["prob"] - 0.01  # top should be >= all others

    def test_deterministic_for_same_image(self, small_model, sample_image, transform, class_names):
        device = torch.device("cpu")
        r1 = predict_single(small_model, sample_image, transform, class_names, device)
        r2 = predict_single(small_model, sample_image, transform, class_names, device)
        assert r1["prediction"] == r2["prediction"]
        assert r1["confidence"] == pytest.approx(r2["confidence"])

    def test_different_images_can_differ(self, small_model, tmp_path, transform, class_names):
        """Two very different images might produce different predictions."""
        img1 = tmp_path / "a.jpg"
        img2 = tmp_path / "b.jpg"
        Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img1)
        Image.new("RGB", (64, 64), color=(255, 255, 255)).save(img2)
        device = torch.device("cpu")
        r1 = predict_single(small_model, img1, transform, class_names, device)
        r2 = predict_single(small_model, img2, transform, class_names, device)
        # Just ensure both return valid results
        assert r1["prediction"] in class_names
        assert r2["prediction"] in class_names
