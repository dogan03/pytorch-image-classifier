"""
Tests for utils/metrics.py
"""

import pytest
import torch
import numpy as np

from utils.metrics import AverageMeter, accuracy, compute_class_accuracy, confusion_matrix


# ===========================================================================
# AverageMeter
# ===========================================================================

class TestAverageMeter:
    def test_initial_state(self):
        m = AverageMeter("loss")
        assert m.val == 0.0
        assert m.avg == 0.0
        assert m.sum == 0.0
        assert m.count == 0

    def test_single_update(self):
        m = AverageMeter("loss")
        m.update(2.0, n=1)
        assert m.val == 2.0
        assert m.avg == 2.0
        assert m.count == 1

    def test_multiple_updates_correct_avg(self):
        m = AverageMeter("acc")
        m.update(80.0, n=10)
        m.update(60.0, n=10)
        assert m.avg == pytest.approx(70.0)

    def test_weighted_average(self):
        m = AverageMeter("acc")
        m.update(100.0, n=1)
        m.update(0.0, n=9)
        assert m.avg == pytest.approx(10.0)

    def test_reset(self):
        m = AverageMeter("loss")
        m.update(5.0)
        m.reset()
        assert m.avg == 0.0
        assert m.count == 0
        assert m.sum == 0.0

    def test_name_stored(self):
        m = AverageMeter("my_metric")
        assert m.name == "my_metric"

    def test_repr(self):
        m = AverageMeter("loss")
        m.update(1.5)
        assert "loss" in repr(m)

    def test_update_default_n_is_one(self):
        m = AverageMeter("x")
        m.update(4.0)
        assert m.count == 1

    def test_large_number_of_updates(self):
        m = AverageMeter("x")
        for i in range(1000):
            m.update(float(i), n=1)
        expected = sum(range(1000)) / 1000
        assert m.avg == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# accuracy
# ===========================================================================

class TestAccuracy:
    def test_perfect_top1(self):
        outputs = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = torch.tensor([0, 1])
        top1 = accuracy(outputs, targets, topk=(1,))[0]
        assert top1 == pytest.approx(100.0)

    def test_zero_top1(self):
        outputs = torch.tensor([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
        targets = torch.tensor([0, 1])
        top1 = accuracy(outputs, targets, topk=(1,))[0]
        assert top1 == pytest.approx(0.0)

    def test_partial_top1(self):
        outputs = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 0.0]])
        targets = torch.tensor([0, 1, 1, 1])  # 2 correct out of 4
        top1 = accuracy(outputs, targets, topk=(1,))[0]
        assert top1 == pytest.approx(50.0)

    def test_top5_all_correct(self):
        outputs = torch.randn(4, 10)
        targets = torch.tensor([0, 1, 2, 3])
        # Make targets the highest score
        for i, t in enumerate(targets):
            outputs[i, t] += 100.0
        top1, top5 = accuracy(outputs, targets, topk=(1, 5))
        assert top1 == pytest.approx(100.0)
        assert top5 == pytest.approx(100.0)

    def test_returns_list(self):
        outputs = torch.randn(4, 5)
        targets = torch.zeros(4, dtype=torch.long)
        result = accuracy(outputs, targets, topk=(1,))
        assert isinstance(result, list)
        assert len(result) == 1

    def test_single_sample(self):
        outputs = torch.tensor([[0.0, 0.0, 10.0]])
        targets = torch.tensor([2])
        top1 = accuracy(outputs, targets)[0]
        assert top1 == pytest.approx(100.0)


# ===========================================================================
# compute_class_accuracy
# ===========================================================================

class TestComputeClassAccuracy:
    def test_perfect_accuracy(self):
        preds = [0, 1, 2, 0, 1, 2]
        targets = [0, 1, 2, 0, 1, 2]
        result = compute_class_accuracy(preds, targets)
        for v in result.values():
            assert v == pytest.approx(100.0)

    def test_zero_accuracy(self):
        preds = [1, 2, 0]
        targets = [0, 1, 2]
        result = compute_class_accuracy(preds, targets)
        for v in result.values():
            assert v == pytest.approx(0.0)

    def test_with_class_names(self):
        preds = [0, 0, 1, 1]
        targets = [0, 0, 1, 1]
        result = compute_class_accuracy(preds, targets, class_names=["cat", "dog"])
        assert "cat" in result
        assert "dog" in result

    def test_without_class_names(self):
        preds = [0, 1]
        targets = [0, 1]
        result = compute_class_accuracy(preds, targets)
        assert "0" in result or 0 in result

    def test_partial_accuracy(self):
        preds   = [0, 0, 1, 1]
        targets = [0, 1, 1, 0]  # class 0: 1/2=50%, class 1: 1/2=50%
        result = compute_class_accuracy(preds, targets)
        for v in result.values():
            assert v == pytest.approx(50.0)


# ===========================================================================
# confusion_matrix
# ===========================================================================

class TestConfusionMatrix:
    def test_perfect_predictions(self):
        preds   = [0, 1, 2]
        targets = [0, 1, 2]
        cm = confusion_matrix(preds, targets, num_classes=3)
        assert np.array_equal(cm, np.eye(3, dtype=int))

    def test_all_wrong(self):
        preds   = [1, 2, 0]
        targets = [0, 1, 2]
        cm = confusion_matrix(preds, targets, num_classes=3)
        assert cm.diagonal().sum() == 0

    def test_shape(self):
        preds   = [0, 1, 0, 1]
        targets = [0, 0, 1, 1]
        cm = confusion_matrix(preds, targets, num_classes=2)
        assert cm.shape == (2, 2)

    def test_sum_equals_total(self):
        preds   = [0, 0, 1, 2, 1]
        targets = [0, 1, 1, 2, 0]
        cm = confusion_matrix(preds, targets, num_classes=3)
        assert cm.sum() == 5

    def test_dtype_is_int(self):
        cm = confusion_matrix([0, 1], [0, 1], num_classes=2)
        assert cm.dtype == int
