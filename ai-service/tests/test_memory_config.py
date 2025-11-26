"""Unit tests for MemoryConfig."""

import os
from unittest.mock import patch

import pytest

from app.utils.memory_config import MemoryConfig


class TestMemoryConfigDefaults:
    """Tests for default configuration values."""

    def test_default_max_memory_gb(self) -> None:
        """Default max memory should be 16.0 GB."""
        config = MemoryConfig()
        assert config.max_memory_gb == 16.0

    def test_default_training_allocation(self) -> None:
        """Default training allocation should be 60%."""
        config = MemoryConfig()
        assert config.training_allocation == 0.60

    def test_default_inference_allocation(self) -> None:
        """Default inference allocation should be 30%."""
        config = MemoryConfig()
        assert config.inference_allocation == 0.30

    def test_default_system_reserve(self) -> None:
        """Default system reserve should be 10%."""
        config = MemoryConfig()
        assert config.system_reserve == 0.10

    def test_allocations_sum_to_one(self) -> None:
        """Allocations should sum to 100%."""
        config = MemoryConfig()
        total = (
            config.training_allocation
            + config.inference_allocation
            + config.system_reserve
        )
        assert total == pytest.approx(1.0)


class TestMemoryConfigFromEnv:
    """Tests for environment variable loading."""

    def test_from_env_uses_default_when_not_set(self) -> None:
        """Should use default 16.0 GB when env var not set."""
        # Ensure env var is not set
        env = os.environ.copy()
        env.pop("RINGRIFT_MAX_MEMORY_GB", None)
        with patch.dict(os.environ, env, clear=True):
            config = MemoryConfig.from_env()
            assert config.max_memory_gb == 16.0

    def test_from_env_reads_custom_value(self) -> None:
        """Should read custom value from environment variable."""
        with patch.dict(os.environ, {"RINGRIFT_MAX_MEMORY_GB": "32.0"}):
            config = MemoryConfig.from_env()
            assert config.max_memory_gb == 32.0

    def test_from_env_handles_fractional_values(self) -> None:
        """Should handle fractional GB values."""
        with patch.dict(os.environ, {"RINGRIFT_MAX_MEMORY_GB": "8.5"}):
            config = MemoryConfig.from_env()
            assert config.max_memory_gb == 8.5


class TestMemoryLimitCalculations:
    """Tests for memory limit calculations."""

    def test_training_limit_bytes_default(self) -> None:
        """Training limit should be 60% of 16GB."""
        config = MemoryConfig()
        expected = int(16.0 * 0.60 * 1024**3)
        assert config.get_training_limit_bytes() == expected

    def test_inference_limit_bytes_default(self) -> None:
        """Inference limit should be 30% of 16GB."""
        config = MemoryConfig()
        expected = int(16.0 * 0.30 * 1024**3)
        assert config.get_inference_limit_bytes() == expected

    def test_transposition_table_limit_bytes_default(self) -> None:
        """Transposition table limit should be half of inference limit."""
        config = MemoryConfig()
        inference_limit = config.get_inference_limit_bytes()
        expected = inference_limit // 2
        assert config.get_transposition_table_limit_bytes() == expected

    def test_training_limit_with_custom_memory(self) -> None:
        """Training limit should scale with custom memory."""
        config = MemoryConfig(max_memory_gb=32.0)
        expected = int(32.0 * 0.60 * 1024**3)
        assert config.get_training_limit_bytes() == expected

    def test_inference_limit_with_custom_allocation(self) -> None:
        """Inference limit should scale with custom allocation."""
        config = MemoryConfig(inference_allocation=0.50)
        expected = int(16.0 * 0.50 * 1024**3)
        assert config.get_inference_limit_bytes() == expected

    def test_limits_are_integers(self) -> None:
        """All limits should return integers."""
        config = MemoryConfig()
        assert isinstance(config.get_training_limit_bytes(), int)
        assert isinstance(config.get_inference_limit_bytes(), int)
        assert isinstance(config.get_transposition_table_limit_bytes(), int)

    def test_training_limit_bytes_calculation(self) -> None:
        """Verify exact training limit calculation."""
        config = MemoryConfig(max_memory_gb=10.0, training_allocation=0.50)
        # 10 * 0.50 * 1024^3 = 5GB in bytes
        expected = int(10.0 * 0.50 * 1024**3)
        assert config.get_training_limit_bytes() == expected


class TestCheckAvailableMemory:
    """Tests for memory availability checking."""

    def test_check_available_memory_returns_tuple(self) -> None:
        """Should return (ok, available_gb) tuple."""
        config = MemoryConfig()
        result = config.check_available_memory()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_check_available_memory_positive_available(self) -> None:
        """Available memory should be positive."""
        config = MemoryConfig()
        _, available_gb = config.check_available_memory()
        assert available_gb > 0

    @patch("app.utils.memory_config.psutil")
    def test_check_available_memory_sufficient(self, mock_psutil) -> None:
        """Should return True when memory is sufficient."""
        # Simulate 10GB available
        mock_memory = type(
            "Memory", (), {"available": 10 * 1024**3}
        )()
        mock_psutil.virtual_memory.return_value = mock_memory

        # With 16GB max and 10% reserve, need 1.6GB minimum
        config = MemoryConfig(max_memory_gb=16.0, system_reserve=0.10)
        ok, available = config.check_available_memory()

        assert ok is True
        assert available == pytest.approx(10.0)

    @patch("app.utils.memory_config.psutil")
    def test_check_available_memory_insufficient(self, mock_psutil) -> None:
        """Should return False when memory is insufficient."""
        # Simulate 1GB available
        mock_memory = type(
            "Memory", (), {"available": 1 * 1024**3}
        )()
        mock_psutil.virtual_memory.return_value = mock_memory

        # With 16GB max and 10% reserve, need 1.6GB minimum
        config = MemoryConfig(max_memory_gb=16.0, system_reserve=0.10)
        ok, available = config.check_available_memory()

        assert ok is False
        assert available == pytest.approx(1.0)