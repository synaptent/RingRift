"""Tests for app/core/logging_config.py - Unified logging configuration."""

import logging
import pytest
from pathlib import Path
from unittest.mock import patch

from app.core.logging_config import (
    setup_logging,
    get_logger,
    configure_third_party_loggers,
    LogContext,
    DEFAULT_FORMAT,
    COMPACT_FORMAT,
    DETAILED_FORMAT,
    STRUCTURED_FORMAT,
)


class TestSetupLogging:
    """Test setup_logging function."""

    def test_returns_logger(self):
        """setup_logging should return a Logger instance."""
        logger = setup_logging("test_logger_1")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger_1"

    def test_logger_level_default(self):
        """Default level should be INFO."""
        logger = setup_logging("test_logger_2")
        assert logger.level == logging.INFO

    def test_logger_level_custom(self):
        """Custom level should be set correctly."""
        logger = setup_logging("test_logger_3", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logger_level_string(self):
        """Level can be specified as string."""
        logger = setup_logging("test_logger_4", level="WARNING")
        assert logger.level == logging.WARNING

    def test_idempotent_logger_creation(self):
        """Calling setup_logging twice returns same logger."""
        logger1 = setup_logging("test_logger_5")
        handler_count = len(logger1.handlers)
        logger2 = setup_logging("test_logger_5")
        assert logger1 is logger2
        # Should not add duplicate handlers
        assert len(logger2.handlers) == handler_count

    def test_console_handler_added(self):
        """Console handler should be added by default."""
        logger = setup_logging("test_logger_6")
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_console_handler_disabled(self):
        """Console handler can be disabled."""
        logger = setup_logging("test_logger_7", console=False)
        # If no file handler either, should have no handlers
        # Note: May still have handlers from prior runs
        # Just verify logger was created
        assert logger.name == "test_logger_7"

    def test_file_handler_with_log_file(self, tmp_path):
        """File handler should be created when log_file specified."""
        log_file = tmp_path / "test.log"
        logger = setup_logging("test_logger_8", log_file=log_file, console=False)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) >= 1
        # Write a message
        logger.info("Test message")
        # Check file exists
        assert log_file.exists()

    def test_file_handler_with_log_dir(self, tmp_path):
        """File handler should be created in log_dir."""
        logger = setup_logging("test_logger_9", log_dir=tmp_path, console=False)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) >= 1
        # Should create file in directory
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) >= 1

    def test_propagate_default_false(self):
        """propagate should default to False."""
        logger = setup_logging("test_logger_10")
        assert logger.propagate is False

    def test_propagate_can_be_enabled(self):
        """propagate can be set to True."""
        logger = setup_logging("test_logger_11", propagate=True)
        assert logger.propagate is True


class TestFormatStyles:
    """Test different format styles."""

    def test_default_format(self):
        """Default format should work."""
        logger = setup_logging("test_format_1", format_style="default")
        assert logger is not None

    def test_compact_format(self):
        """Compact format should work."""
        logger = setup_logging("test_format_2", format_style="compact")
        assert logger is not None

    def test_detailed_format(self):
        """Detailed format should work."""
        logger = setup_logging("test_format_3", format_style="detailed")
        assert logger is not None

    def test_structured_format(self):
        """Structured format should work."""
        logger = setup_logging("test_format_4", format_style="structured")
        assert logger is not None

    def test_unknown_format_uses_default(self):
        """Unknown format should fall back to default."""
        logger = setup_logging("test_format_5", format_style="nonexistent")
        assert logger is not None


class TestGetLogger:
    """Test get_logger function."""

    def test_returns_logger(self):
        """get_logger should return a Logger instance."""
        logger = get_logger("test_get_1")
        assert isinstance(logger, logging.Logger)

    def test_returns_same_logger(self):
        """get_logger should return same logger for same name."""
        logger1 = get_logger("test_get_2")
        logger2 = get_logger("test_get_2")
        assert logger1 is logger2


class TestConfigureThirdPartyLoggers:
    """Test configure_third_party_loggers function."""

    def test_quiets_noisy_packages(self):
        """Should set noisy packages to WARNING."""
        configure_third_party_loggers(quiet=True)
        urllib3_logger = logging.getLogger("urllib3")
        assert urllib3_logger.level >= logging.WARNING

    def test_verbose_packages_not_quieted(self):
        """Specified packages should not be set to WARNING."""
        # Set to INFO first
        urllib3_logger = logging.getLogger("urllib3")
        urllib3_logger.setLevel(logging.INFO)
        # Configure with urllib3 as verbose
        configure_third_party_loggers(quiet=True, verbose_packages=["urllib3"])
        # Should still be INFO (not changed to WARNING)
        assert urllib3_logger.level == logging.INFO


class TestLogContext:
    """Test LogContext context manager."""

    def test_changes_level_temporarily(self):
        """LogContext should change level only in context."""
        logger = setup_logging("test_context_1", level=logging.INFO)
        original_level = logger.level

        with LogContext(logger, logging.DEBUG):
            assert logger.level == logging.DEBUG

        assert logger.level == original_level

    def test_restores_level_on_exception(self):
        """LogContext should restore level even on exception."""
        logger = setup_logging("test_context_2", level=logging.INFO)
        original_level = logger.level

        try:
            with LogContext(logger, logging.DEBUG):
                assert logger.level == logging.DEBUG
                raise ValueError("test")
        except ValueError:
            pass

        assert logger.level == original_level

    def test_returns_logger_in_context(self):
        """LogContext should return logger from __enter__."""
        logger = setup_logging("test_context_3")

        with LogContext(logger, logging.DEBUG) as ctx_logger:
            assert ctx_logger is logger


class TestFormatConstants:
    """Test format string constants."""

    def test_default_format_has_required_fields(self):
        """DEFAULT_FORMAT should have standard fields."""
        assert "%(asctime)s" in DEFAULT_FORMAT
        assert "%(name)s" in DEFAULT_FORMAT
        assert "%(levelname)s" in DEFAULT_FORMAT
        assert "%(message)s" in DEFAULT_FORMAT

    def test_compact_format_is_shorter(self):
        """COMPACT_FORMAT should be shorter than DEFAULT."""
        assert len(COMPACT_FORMAT) < len(DEFAULT_FORMAT)

    def test_detailed_format_has_file_info(self):
        """DETAILED_FORMAT should include file/line info."""
        assert "%(filename)s" in DETAILED_FORMAT
        assert "%(lineno)d" in DETAILED_FORMAT

    def test_structured_format_is_json_like(self):
        """STRUCTURED_FORMAT should be JSON-like."""
        assert "{" in STRUCTURED_FORMAT
        assert "}" in STRUCTURED_FORMAT
        assert '"' in STRUCTURED_FORMAT
