"""Tests for logging utilities."""

import logging

import pytest

from app.utils.logging_utils import (
    PrefixedLogger,
    get_logger,
    LogContext,
    log_duration,
    log_duration_async,
    configure_logging,
    silence_logger,
)


class TestPrefixedLogger:
    """Tests for PrefixedLogger class."""

    def test_format_adds_prefix(self):
        base_logger = logging.getLogger("test_prefixed")
        logger = PrefixedLogger(base_logger, "MyPrefix")
        assert logger._format("message") == "[MyPrefix] message"

    def test_level_property(self):
        base_logger = logging.getLogger("test_level")
        base_logger.setLevel(logging.DEBUG)
        logger = PrefixedLogger(base_logger, "Test")
        assert logger.level == logging.DEBUG

    def test_set_level(self):
        base_logger = logging.getLogger("test_set_level")
        logger = PrefixedLogger(base_logger, "Test")
        logger.setLevel(logging.WARNING)
        assert logger.level == logging.WARNING

    def test_is_enabled_for(self):
        base_logger = logging.getLogger("test_enabled")
        base_logger.setLevel(logging.INFO)
        logger = PrefixedLogger(base_logger, "Test")
        assert logger.isEnabledFor(logging.INFO) is True
        assert logger.isEnabledFor(logging.DEBUG) is False


class TestGetLogger:
    """Tests for get_logger function."""

    def test_simple_name(self):
        logger = get_logger("MyComponent")
        assert isinstance(logger, PrefixedLogger)
        # Mixed case names are preserved as-is
        assert logger._prefix == "MyComponent"

    def test_module_path(self):
        logger = get_logger("app.utils.json_utils")
        assert isinstance(logger, PrefixedLogger)
        assert logger._prefix == "JsonUtils"

    def test_snake_case_converted(self):
        logger = get_logger("my_component_name")
        assert logger._prefix == "MyComponentName"

    def test_unprefixed(self):
        logger = get_logger("test_raw", prefixed=False)
        assert isinstance(logger, logging.Logger)
        assert not isinstance(logger, PrefixedLogger)


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_modifies_prefix_in_context(self):
        base_logger = logging.getLogger("test_context")
        logger = PrefixedLogger(base_logger, "Original")

        with LogContext(logger, key="value"):
            assert logger._prefix == "Original key=value"

        # Prefix should be restored
        assert logger._prefix == "Original"

    def test_multiple_context_values(self):
        base_logger = logging.getLogger("test_context_multi")
        logger = PrefixedLogger(base_logger, "Test")

        with LogContext(logger, a=1, b="two"):
            # Both values should be in prefix
            assert "a=1" in logger._prefix
            assert "b=two" in logger._prefix

    def test_works_with_regular_logger(self):
        logger = logging.getLogger("test_regular")
        # Should not raise
        with LogContext(logger, key="value"):
            pass


class TestLogDuration:
    """Tests for log_duration decorator."""

    def test_returns_function_result(self, caplog):
        base_logger = logging.getLogger("test_duration")
        logger = PrefixedLogger(base_logger, "Test")

        @log_duration(logger, "test operation")
        def my_func(x):
            return x * 2

        result = my_func(21)
        assert result == 42

    def test_logs_duration(self, caplog):
        base_logger = logging.getLogger("test_duration_log")
        base_logger.setLevel(logging.INFO)
        logger = PrefixedLogger(base_logger, "Test")

        @log_duration(logger, "test op", threshold_ms=0)
        def my_func():
            return 42

        with caplog.at_level(logging.INFO):
            my_func()

        # Should have logged something with duration
        assert any("test op" in record.message for record in caplog.records)


class TestLogDurationAsync:
    """Tests for log_duration_async decorator."""

    @pytest.mark.asyncio
    async def test_returns_function_result(self):
        base_logger = logging.getLogger("test_duration_async")
        logger = PrefixedLogger(base_logger, "Test")

        @log_duration_async(logger, "async operation")
        async def my_async_func(x):
            return x * 2

        result = await my_async_func(21)
        assert result == 42


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_sets_root_level(self):
        # Note: this modifies global state
        configure_logging(level=logging.WARNING)
        root = logging.getLogger()
        assert root.level == logging.WARNING


class TestSilenceLogger:
    """Tests for silence_logger function."""

    def test_raises_level(self):
        test_logger = logging.getLogger("noisy_package")
        test_logger.setLevel(logging.DEBUG)

        silence_logger("noisy_package", level=logging.ERROR)

        assert test_logger.level == logging.ERROR
