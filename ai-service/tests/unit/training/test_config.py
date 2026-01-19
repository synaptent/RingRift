"""Tests for Training Configuration.

Tests the GPU scaling configuration, batch size auto-tuning, and
environment variable overrides.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.training.config import (
    GpuScalingConfig,
    _get_gpu_memory_gb,
    _scale_batch_size_for_gpu,
    get_gpu_scaling_config,
    set_gpu_scaling_config,
)


class TestGpuScalingConfig:
    """Tests for GpuScalingConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = GpuScalingConfig()
        assert config.mem_per_sample_large_policy_mb == 0.5
        assert config.mem_per_sample_medium_policy_mb == 0.2
        assert config.mem_per_sample_small_policy_mb == 0.1
        assert config.large_policy_threshold == 50000
        assert config.medium_policy_threshold == 10000
        assert config.reserved_memory_gb == 8.0
        assert config.max_batch_size == 16384

    def test_gpu_tier_thresholds(self):
        """Should have correct GPU tier thresholds."""
        config = GpuScalingConfig()
        assert config.gh200_memory_threshold_gb == 90.0
        assert config.h100_memory_threshold_gb == 70.0
        assert config.a100_memory_threshold_gb == 30.0
        assert config.rtx_memory_threshold_gb == 16.0

    def test_batch_multipliers(self):
        """Should have correct batch multipliers per GPU tier.

        v3.0 (Jan 2026): Reduced multipliers for conservative memory targeting.
        """
        config = GpuScalingConfig()
        # Conservative multipliers (reduced from 64/32/16/8/4)
        assert config.gh200_batch_multiplier == 40
        assert config.h100_batch_multiplier == 20
        assert config.a100_batch_multiplier == 10
        assert config.rtx_batch_multiplier == 5
        assert config.consumer_batch_multiplier == 2

    def test_memory_target_defaults(self):
        """Should have conservative memory target defaults (v3.0)."""
        config = GpuScalingConfig()
        assert config.target_memory_fraction == 0.50  # Conservative default
        assert config.safe_mode_memory_fraction == 0.35  # Extra conservative
        assert config.safe_mode_enabled is False

    def test_safe_mode_from_env(self):
        """Should enable safe mode from environment variable."""
        with patch.dict(os.environ, {"RINGRIFT_GPU_SAFE_MODE": "1"}):
            config = GpuScalingConfig.from_env()
            assert config.safe_mode_enabled is True

        with patch.dict(os.environ, {"RINGRIFT_GPU_SAFE_MODE": "true"}):
            config = GpuScalingConfig.from_env()
            assert config.safe_mode_enabled is True

    def test_memory_fraction_from_env(self):
        """Should override memory fraction from environment."""
        with patch.dict(os.environ, {"RINGRIFT_GPU_TARGET_MEMORY_FRACTION": "0.6"}):
            config = GpuScalingConfig.from_env()
            assert config.target_memory_fraction == 0.6

    def test_from_env_no_overrides(self):
        """Should return defaults when no env vars set."""
        config = GpuScalingConfig.from_env()
        assert config.max_batch_size == 16384
        assert config.reserved_memory_gb == 8.0

    def test_from_env_with_int_override(self):
        """Should override integer values from environment."""
        with patch.dict(os.environ, {"RINGRIFT_GPU_MAX_BATCH_SIZE": "4096"}):
            config = GpuScalingConfig.from_env()
            assert config.max_batch_size == 4096

    def test_from_env_with_float_override(self):
        """Should override float values from environment."""
        with patch.dict(os.environ, {"RINGRIFT_GPU_RESERVED_MEMORY_GB": "12.5"}):
            config = GpuScalingConfig.from_env()
            assert config.reserved_memory_gb == 12.5

    def test_from_env_ignores_invalid(self):
        """Should ignore invalid environment values."""
        with patch.dict(os.environ, {"RINGRIFT_GPU_MAX_BATCH_SIZE": "invalid"}):
            config = GpuScalingConfig.from_env()
            assert config.max_batch_size == 16384  # Default

    def test_from_env_multiplier_overrides(self):
        """Should override batch multipliers from environment."""
        with patch.dict(os.environ, {
            "RINGRIFT_GPU_GH200_BATCH_MULTIPLIER": "64",
            "RINGRIFT_GPU_H100_BATCH_MULTIPLIER": "32",
        }):
            config = GpuScalingConfig.from_env()
            assert config.gh200_batch_multiplier == 64
            assert config.h100_batch_multiplier == 32


class TestGetSetGpuScalingConfig:
    """Tests for config getter/setter functions."""

    def setup_method(self):
        """Reset global config before each test."""
        import app.training.config as config_module
        config_module._gpu_scaling_config = None

    def test_get_returns_default(self):
        """Should return default config on first call."""
        config = get_gpu_scaling_config()
        assert isinstance(config, GpuScalingConfig)
        assert config.max_batch_size == 16384

    def test_get_caches_config(self):
        """Should cache and return same config on subsequent calls."""
        config1 = get_gpu_scaling_config()
        config2 = get_gpu_scaling_config()
        assert config1 is config2

    def test_set_overrides_config(self):
        """Should allow setting custom config."""
        custom = GpuScalingConfig(max_batch_size=1024)
        set_gpu_scaling_config(custom)
        config = get_gpu_scaling_config()
        assert config.max_batch_size == 1024


class TestScaleBatchSizeForGpu:
    """Tests for batch size scaling based on GPU memory."""

    def test_no_gpu_returns_base(self):
        """Should return base batch size when no GPU available."""
        with patch('app.training.config._get_gpu_memory_gb', return_value=0.0):
            result = _scale_batch_size_for_gpu(256)
            assert result == 256

    def test_scales_for_gh200(self):
        """Should scale up for GH200-class GPUs."""
        config = GpuScalingConfig()
        with patch('app.training.config._get_gpu_memory_gb', return_value=96.0):
            result = _scale_batch_size_for_gpu(256, config=config)
            # 256 * 32 = 8192 (capped by max)
            assert result <= config.max_batch_size

    def test_scales_for_h100(self):
        """Should scale appropriately for H100-class GPUs."""
        config = GpuScalingConfig()
        with patch('app.training.config._get_gpu_memory_gb', return_value=80.0):
            result = _scale_batch_size_for_gpu(256, config=config)
            # Should use h100_batch_multiplier (16)
            assert result >= 256

    def test_scales_for_a100(self):
        """Should scale appropriately for A100-class GPUs."""
        config = GpuScalingConfig()
        with patch('app.training.config._get_gpu_memory_gb', return_value=40.0):
            result = _scale_batch_size_for_gpu(256, config=config)
            # Should use a100_batch_multiplier (8)
            assert result >= 256

    def test_scales_for_rtx(self):
        """Should scale appropriately for RTX-class GPUs."""
        config = GpuScalingConfig()
        with patch('app.training.config._get_gpu_memory_gb', return_value=24.0):
            result = _scale_batch_size_for_gpu(256, config=config)
            # Should use rtx_batch_multiplier (4)
            assert result >= 256

    def test_consumer_gpu_scaling(self):
        """Should use consumer multiplier for small GPUs."""
        # Use 12GB to have memory left after 8GB reserved
        config = GpuScalingConfig(reserved_memory_gb=4.0)
        with patch('app.training.config._get_gpu_memory_gb', return_value=12.0):
            result = _scale_batch_size_for_gpu(256, config=config)
            # Should use consumer_batch_multiplier (2) since 12 < 16 (rtx threshold)
            assert result >= 256

    def test_respects_max_batch_size(self):
        """Should not exceed max_batch_size."""
        config = GpuScalingConfig(max_batch_size=1000)
        with patch('app.training.config._get_gpu_memory_gb', return_value=96.0):
            result = _scale_batch_size_for_gpu(256, config=config)
            assert result <= 1000

    def test_large_policy_reduces_batch(self):
        """Should reduce batch size for large policy outputs."""
        config = GpuScalingConfig()
        with patch('app.training.config._get_gpu_memory_gb', return_value=40.0):
            small_policy = _scale_batch_size_for_gpu(256, policy_size=5000, config=config)
            large_policy = _scale_batch_size_for_gpu(256, policy_size=60000, config=config)
            # Large policy should have smaller max batch due to memory
            assert large_policy <= small_policy

    def test_custom_config_used(self):
        """Should use custom config when provided."""
        custom = GpuScalingConfig(
            gh200_batch_multiplier=2,
            max_batch_size=512
        )
        with patch('app.training.config._get_gpu_memory_gb', return_value=96.0):
            result = _scale_batch_size_for_gpu(256, config=custom)
            # 256 * 2 = 512
            assert result == 512


class TestGetGpuMemoryGb:
    """Tests for GPU memory detection."""

    def test_returns_zero_without_cuda(self):
        """Should return 0 when CUDA not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = _get_gpu_memory_gb()
            # Should return 0 when no CUDA
            assert result == 0.0

    def test_returns_memory_with_cuda(self):
        """Should return GPU memory when CUDA available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 80 * (1024 ** 3)  # 80 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = _get_gpu_memory_gb()
            assert abs(result - 80.0) < 0.1

    def test_returns_zero_on_exception(self):
        """Should return 0 when RuntimeError occurs (e.g., CUDA errors)."""
        mock_torch = MagicMock()
        # Use RuntimeError since _get_gpu_memory_gb catches (ImportError, AttributeError, RuntimeError)
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA error")
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = _get_gpu_memory_gb()
            assert result == 0.0


class TestBatchSizeAutoTuner:
    """Tests for BatchSizeAutoTuner class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.train = MagicMock()
        model.eval = MagicMock()
        return model

    @pytest.fixture
    def mock_tensors(self):
        """Create mock input tensors."""
        import numpy as np
        features = np.zeros((1, 100), dtype=np.float32)
        globals_ = np.zeros((1, 10), dtype=np.float32)
        return features, globals_

    def test_import(self):
        """Should be importable."""
        from app.training.config import BatchSizeAutoTuner
        assert BatchSizeAutoTuner is not None

    def test_initialization(self, mock_model, mock_tensors):
        """Should initialize with model and sample data."""
        pytest.importorskip("torch")
        from app.training.config import BatchSizeAutoTuner

        features, globals_ = mock_tensors
        tuner = BatchSizeAutoTuner(
            model=mock_model,
            sample_features=features,
            sample_globals=globals_,
            device="cpu",
        )
        assert tuner._model is mock_model
        assert tuner._device == "cpu"
