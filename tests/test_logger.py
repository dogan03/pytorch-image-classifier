"""
Tests for utils/logger.py
"""

import logging
import os

import pytest

from utils.logger import TrainingLogger, get_logger

# ===========================================================================
# get_logger
# ===========================================================================


class TestGetLogger:
    def test_returns_logger_instance(self):
        logger = get_logger("test_basic")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = get_logger("my_module")
        assert logger.name == "my_module"

    def test_default_level_is_info(self):
        logger = get_logger("test_level")
        assert logger.level == logging.INFO

    def test_custom_level(self):
        logger = get_logger("test_debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_log_file_created(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = get_logger("test_file_logger", log_file=log_file)
        logger.info("test message")
        assert os.path.exists(log_file)

    def test_log_file_contains_message(self, tmp_path):
        log_file = str(tmp_path / "out.log")
        logger = get_logger("test_content", log_file=log_file)
        logger.info("hello world")
        content = open(log_file).read()
        assert "hello world" in content

    def test_no_duplicate_handlers_on_second_call(self):
        logger = get_logger("test_dedup_x")
        count1 = len(logger.handlers)
        logger2 = get_logger("test_dedup_x")
        count2 = len(logger2.handlers)
        assert count1 == count2

    def test_creates_log_dir_if_missing(self, tmp_path):
        log_file = str(tmp_path / "nested" / "dir" / "app.log")
        logger = get_logger("test_mkdirs", log_file=log_file)
        logger.info("test")
        assert os.path.exists(log_file)


# ===========================================================================
# TrainingLogger
# ===========================================================================


class TestTrainingLogger:
    def test_instantiates(self):
        tl = TrainingLogger()
        assert tl is not None

    def test_log_epoch_updates_history(self):
        tl = TrainingLogger()
        tl.log_epoch(1, 10, train_loss=0.5, train_acc=80.0, val_loss=0.6, val_acc=78.0)
        assert len(tl.history["train_loss"]) == 1
        assert tl.history["train_loss"][0] == pytest.approx(0.5)
        assert tl.history["val_acc"][0] == pytest.approx(78.0)

    def test_log_epoch_accumulates(self):
        tl = TrainingLogger()
        for i in range(5):
            tl.log_epoch(i + 1, 10, 0.1 * i, 90.0, 0.2 * i, 85.0)
        assert len(tl.history["train_loss"]) == 5

    def test_history_keys_exist(self):
        tl = TrainingLogger()
        for key in ["train_loss", "train_acc", "val_loss", "val_acc"]:
            assert key in tl.history

    def test_history_initially_empty(self):
        tl = TrainingLogger()
        for v in tl.history.values():
            assert v == []

    def test_log_test_does_not_raise(self):
        tl = TrainingLogger()
        tl.log_test(test_loss=0.3, test_acc=92.0)

    def test_log_info_does_not_raise(self):
        tl = TrainingLogger()
        tl.log_info("Training started")

    def test_multiple_epochs_stored_in_order(self):
        tl = TrainingLogger()
        losses = [0.9, 0.7, 0.5, 0.3]
        for i, loss in enumerate(losses):
            tl.log_epoch(i + 1, 4, loss, 80.0, loss + 0.1, 78.0)
        assert tl.history["train_loss"] == pytest.approx(losses)
