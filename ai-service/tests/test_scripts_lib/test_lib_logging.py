"""Tests for scripts.lib.logging_config module."""

import pytest
import logging
import json
import time
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock

from scripts.lib.logging_config import (
    JsonFormatter,
    ColoredFormatter,
    MetricsLogger,
    TimerContext,
    setup_logging,
    get_logger,
    get_metrics_logger,
    setup_script_logging,
)


class TestJsonFormatter:
    """Tests for JsonFormatter class."""

    def test_format_basic(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_format_with_extra_fields(self):
        formatter = JsonFormatter(extra_fields={"app": "ringrift"})
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["app"] == "ringrift"

    def test_format_without_timestamp(self):
        formatter = JsonFormatter(include_timestamp=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" not in data

    def test_format_with_pathname(self):
        formatter = JsonFormatter(include_pathname=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["pathname"] == "/path/to/test.py"
        assert data["lineno"] == 42


class TestColoredFormatter:
    """Tests for ColoredFormatter class."""

    def test_format_without_colors(self):
        formatter = ColoredFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "INFO" in result
        assert "Test message" in result

    def test_colors_disabled_for_non_tty(self):
        # ColoredFormatter should disable colors when stdout is not a TTY
        with patch('sys.stdout.isatty', return_value=False):
            formatter = ColoredFormatter(use_colors=True)
            assert formatter.use_colors is False


class TestMetricsLogger:
    """Tests for MetricsLogger class."""

    def test_set_metric(self):
        metrics = MetricsLogger("test", log_interval=3600)
        metrics.set("games_processed", 100)
        assert metrics.metrics["games_processed"] == 100

    def test_increment_counter(self):
        metrics = MetricsLogger("test", log_interval=3600)
        metrics.increment("errors")
        metrics.increment("errors")
        metrics.increment("errors", 3)
        assert metrics.counters["errors"] == 5

    def test_record_time(self):
        metrics = MetricsLogger("test", log_interval=3600)
        metrics.record_time("processing", 1.5)
        assert metrics.timers["processing"] == 1.5

    def test_timer_context(self):
        metrics = MetricsLogger("test", log_interval=3600)

        with metrics.time("operation"):
            time.sleep(0.01)

        assert "operation" in metrics.timers
        assert metrics.timers["operation"] >= 0.01

    def test_get_all(self):
        metrics = MetricsLogger("test", log_interval=3600)
        metrics.set("value", 42)
        metrics.increment("count", 10)
        metrics.record_time("time", 1.0)

        result = metrics.get_all()
        assert result["metrics"]["value"] == 42
        assert result["counters"]["count"] == 10
        assert result["timers"]["time"] == 1.0

    def test_flush(self):
        metrics = MetricsLogger("test", log_interval=3600)
        metrics.set("value", 42)

        # Flush should log without error
        metrics.flush()


class TestTimerContext:
    """Tests for TimerContext class."""

    def test_timer_context_records_time(self):
        metrics = MetricsLogger("test", log_interval=3600)
        ctx = TimerContext(metrics, "test_timer")

        with ctx:
            time.sleep(0.01)

        assert "test_timer" in metrics.timers
        assert metrics.timers["test_timer"] >= 0.01


class TestSetupFunctions:
    """Tests for setup functions."""

    def test_setup_logging_returns_logger(self):
        logger = setup_logging(level="INFO", log_to_console=False)
        assert isinstance(logger, logging.Logger)

    def test_get_logger(self):
        logger = get_logger("test.module")
        assert logger.name == "test.module"

    def test_get_metrics_logger(self):
        metrics = get_metrics_logger("test", log_interval=60)
        assert isinstance(metrics, MetricsLogger)
        assert metrics.name == "test"
        assert metrics.log_interval == 60

    def test_setup_script_logging(self, tmp_path):
        # Create the log directory
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_script_logging(
            script_name="test_script",
            log_dir=str(log_dir),
            level="DEBUG",
        )
        assert isinstance(logger, logging.Logger)


class TestLoggingIntegration:
    """Integration tests for logging setup."""

    def test_json_logging_output(self):
        formatter = JsonFormatter()
        handler = logging.StreamHandler(StringIO())
        handler.setFormatter(formatter)

        logger = logging.getLogger("integration_test")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("Test message", extra={"custom_field": "value"})

        output = handler.stream.getvalue()
        data = json.loads(output.strip())

        assert data["message"] == "Test message"
        assert data["custom_field"] == "value"

        # Cleanup
        logger.removeHandler(handler)
