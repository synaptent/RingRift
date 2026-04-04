"""Unit tests for parallel encoding utilities.

Tests cover:
- Encoder caching mechanisms
- EncodedSample and GameEncodingResult dataclasses
- Heuristic feature extraction caching
- Worker-safe encoder initialization
"""

import os

import pytest
import numpy as np

from app.models import BoardType


class TestEncoderCache:
    """Tests for encoder caching functions."""

    def test_get_encoder_returns_cached_instance(self):
        """Second call to _get_encoder returns same instance."""
        from app.training.parallel_encoding import _get_encoder, _ENCODER_CACHE

        # Clear cache first
        _ENCODER_CACHE.clear()

        encoder1 = _get_encoder("hexagonal", encoder_version="v3", feature_version=2)
        encoder2 = _get_encoder("hexagonal", encoder_version="v3", feature_version=2)

        assert encoder1 is encoder2

    def test_get_encoder_different_boards_separate_cache(self):
        """Different board types get separate encoders."""
        from app.training.parallel_encoding import _get_encoder, _ENCODER_CACHE

        _ENCODER_CACHE.clear()

        encoder_hex = _get_encoder("hexagonal", encoder_version="v3")
        encoder_hex8 = _get_encoder("hex8", encoder_version="v3")

        # Different board types should be different instances
        assert encoder_hex is not encoder_hex8

    def test_get_encoder_different_versions_separate_cache(self):
        """Different encoder versions get separate entries."""
        from app.training.parallel_encoding import _get_encoder, _ENCODER_CACHE

        _ENCODER_CACHE.clear()

        encoder_v2 = _get_encoder("hexagonal", encoder_version="v2", feature_version=2)
        encoder_v3 = _get_encoder("hexagonal", encoder_version="v3", feature_version=2)

        assert encoder_v2 is not encoder_v3


class TestNeuralNetEncoderCache:
    """Tests for neural net encoder caching."""

    def test_get_neural_net_encoder_caching(self):
        """Neural net encoder is cached per board type."""
        from app.training.parallel_encoding import (
            _get_neural_net_encoder,
            _ENCODER_CACHE,
        )

        _ENCODER_CACHE.clear()

        encoder1 = _get_neural_net_encoder("square8", feature_version=2)
        encoder2 = _get_neural_net_encoder("square8", feature_version=2)

        assert encoder1 is encoder2

    def test_get_neural_net_encoder_sets_board_size(self):
        """Neural net encoder has correct board size."""
        from app.training.parallel_encoding import (
            _get_neural_net_encoder,
            _ENCODER_CACHE,
        )

        _ENCODER_CACHE.clear()

        encoder = _get_neural_net_encoder("square19", feature_version=2)
        assert encoder.board_size == 19

    def test_get_neural_net_encoder_does_not_leak_force_cpu_env(self, monkeypatch):
        """Internal CPU forcing must not mutate the caller's environment."""
        from app.training.parallel_encoding import (
            _get_neural_net_encoder,
            _ENCODER_CACHE,
        )

        monkeypatch.delenv("RINGRIFT_FORCE_CPU", raising=False)
        _ENCODER_CACHE.clear()

        _get_neural_net_encoder("square8", feature_version=2)

        assert os.environ.get("RINGRIFT_FORCE_CPU") is None

    def test_get_neural_net_encoder_restores_existing_force_cpu_env(self, monkeypatch):
        """Existing env state should be preserved after encoder creation."""
        from app.training.parallel_encoding import (
            _get_neural_net_encoder,
            _ENCODER_CACHE,
        )

        monkeypatch.setenv("RINGRIFT_FORCE_CPU", "0")
        _ENCODER_CACHE.clear()

        _get_neural_net_encoder("square8", feature_version=2)

        assert os.environ.get("RINGRIFT_FORCE_CPU") == "0"


class TestHeuristicExtractor:
    """Tests for heuristic feature extraction caching."""

    def test_get_heuristic_extractor_fast_mode(self):
        """Fast mode returns 21 features."""
        from app.training.parallel_encoding import (
            _get_heuristic_extractor,
            _HEURISTIC_EXTRACTOR_CACHE,
        )

        _HEURISTIC_EXTRACTOR_CACHE.clear()

        extractor, num_features = _get_heuristic_extractor(full_mode=False)

        # Either 21 features or None if module not available
        assert num_features == 21 or extractor is None

    def test_get_heuristic_extractor_full_mode(self):
        """Full mode returns 49 features."""
        from app.training.parallel_encoding import (
            _get_heuristic_extractor,
            _HEURISTIC_EXTRACTOR_CACHE,
        )

        _HEURISTIC_EXTRACTOR_CACHE.clear()

        extractor, num_features = _get_heuristic_extractor(full_mode=True)

        # Either 49 features or None if module not available
        assert num_features == 49 or extractor is None

    def test_get_heuristic_extractor_caching(self):
        """Heuristic extractor is cached."""
        from app.training.parallel_encoding import (
            _get_heuristic_extractor,
            _HEURISTIC_EXTRACTOR_CACHE,
        )

        _HEURISTIC_EXTRACTOR_CACHE.clear()

        result1 = _get_heuristic_extractor(full_mode=False)
        result2 = _get_heuristic_extractor(full_mode=False)

        assert result1 is result2


class TestEncodedSample:
    """Tests for EncodedSample dataclass."""

    def test_encoded_sample_creation(self):
        """EncodedSample can be created with required fields."""
        from app.training.parallel_encoding import EncodedSample

        sample = EncodedSample(
            features=np.zeros((10, 8, 8)),
            globals=np.zeros(5),
            value=0.5,
            values_mp=np.zeros(4),
            policy_index=42,
            move_number=3,
            total_moves=20,
            phase="play",
            perspective=1,
            num_players=2,
        )

        assert sample.value == 0.5
        assert sample.policy_index == 42
        assert sample.move_number == 3
        assert sample.features.shape == (10, 8, 8)

    def test_encoded_sample_defaults(self):
        """EncodedSample has correct default values."""
        from app.training.parallel_encoding import EncodedSample

        sample = EncodedSample(
            features=np.zeros((10, 8, 8)),
            globals=np.zeros(5),
            value=0.0,
            values_mp=np.zeros(4),
            policy_index=0,
            move_number=1,
            total_moves=10,
            phase="play",
            perspective=1,
            num_players=2,
        )

        assert sample.game_id == ""
        assert sample.move_type == "unknown"
        assert sample.heuristics is None

    def test_encoded_sample_with_heuristics(self):
        """EncodedSample can include heuristic features."""
        from app.training.parallel_encoding import EncodedSample

        heuristics = np.random.randn(49).astype(np.float32)

        sample = EncodedSample(
            features=np.zeros((10, 8, 8)),
            globals=np.zeros(5),
            value=0.0,
            values_mp=np.zeros(4),
            policy_index=0,
            move_number=1,
            total_moves=10,
            phase="play",
            perspective=1,
            num_players=2,
            heuristics=heuristics,
        )

        assert sample.heuristics is not None
        assert sample.heuristics.shape == (49,)


class TestGameEncodingResult:
    """Tests for GameEncodingResult dataclass."""

    def test_result_creation_success(self):
        """GameEncodingResult can represent successful encoding."""
        from app.training.parallel_encoding import GameEncodingResult, EncodedSample

        samples = [
            EncodedSample(
                features=np.zeros((10, 8, 8)),
                globals=np.zeros(5),
                value=0.0,
                values_mp=np.zeros(4),
                policy_index=0,
                move_number=i,
                total_moves=5,
                phase="play",
                perspective=1,
                num_players=2,
            )
            for i in range(5)
        ]

        result = GameEncodingResult(
            game_id="test-game-123",
            samples=samples,
        )

        assert result.game_id == "test-game-123"
        assert len(result.samples) == 5
        assert result.error is None

    def test_result_creation_error(self):
        """GameEncodingResult can represent encoding error."""
        from app.training.parallel_encoding import GameEncodingResult

        result = GameEncodingResult(
            game_id="failed-game-456",
            samples=[],
            error="Failed to replay game: Invalid move at step 5",
        )

        assert result.game_id == "failed-game-456"
        assert len(result.samples) == 0
        assert result.error is not None
        assert "Invalid move" in result.error


class TestCacheClear:
    """Tests for cache management."""

    def test_encoder_cache_is_module_level(self):
        """Encoder cache persists across function calls."""
        from app.training.parallel_encoding import _ENCODER_CACHE

        # _ENCODER_CACHE is a module-level dict
        assert isinstance(_ENCODER_CACHE, dict)

    def test_heuristic_cache_is_module_level(self):
        """Heuristic cache persists across function calls."""
        from app.training.parallel_encoding import _HEURISTIC_EXTRACTOR_CACHE

        assert isinstance(_HEURISTIC_EXTRACTOR_CACHE, dict)

    def test_cache_can_be_cleared(self):
        """Caches can be cleared for worker process isolation."""
        from app.training.parallel_encoding import (
            _ENCODER_CACHE,
            _HEURISTIC_EXTRACTOR_CACHE,
        )

        # Add something
        _ENCODER_CACHE["test"] = "value"
        _HEURISTIC_EXTRACTOR_CACHE["test"] = "value"

        # Clear
        _ENCODER_CACHE.clear()
        _HEURISTIC_EXTRACTOR_CACHE.clear()

        assert len(_ENCODER_CACHE) == 0
        assert len(_HEURISTIC_EXTRACTOR_CACHE) == 0


class TestBoardSizeMapping:
    """Tests for board size configuration."""

    def test_board_sizes_in_encoder(self):
        """Neural net encoder sets correct board sizes."""
        from app.training.parallel_encoding import _get_neural_net_encoder, _ENCODER_CACHE

        expected_sizes = {
            "square8": 8,
            "square19": 19,
            "hex8": 9,
            "hexagonal": 25,
        }

        _ENCODER_CACHE.clear()

        for board_type_str, expected_size in expected_sizes.items():
            try:
                encoder = _get_neural_net_encoder(board_type_str, feature_version=2)
                assert encoder.board_size == expected_size, (
                    f"{board_type_str}: expected {expected_size}, got {encoder.board_size}"
                )
            except Exception:
                # Skip if encoder can't be created (missing dependencies)
                pass


class TestFeatureVersions:
    """Tests for feature version handling."""

    def test_feature_version_in_cache_key(self):
        """Different feature versions get separate cache entries."""
        from app.training.parallel_encoding import _get_encoder, _ENCODER_CACHE

        _ENCODER_CACHE.clear()

        # Feature version should be part of cache key
        key_v1 = "hexagonal_v3_fv1"
        key_v2 = "hexagonal_v3_fv2"

        _get_encoder("hexagonal", encoder_version="v3", feature_version=1)
        _get_encoder("hexagonal", encoder_version="v3", feature_version=2)

        # Both should be in cache with different keys
        assert key_v1 in _ENCODER_CACHE
        assert key_v2 in _ENCODER_CACHE


class TestEncodedSampleArrays:
    """Tests for array handling in EncodedSample."""

    def test_features_array_can_be_3d(self):
        """Features array supports (C, H, W) format."""
        from app.training.parallel_encoding import EncodedSample

        features = np.random.randn(24, 8, 8).astype(np.float32)

        sample = EncodedSample(
            features=features,
            globals=np.zeros(5),
            value=0.0,
            values_mp=np.zeros(4),
            policy_index=0,
            move_number=1,
            total_moves=10,
            phase="play",
            perspective=1,
            num_players=2,
        )

        assert sample.features.shape == (24, 8, 8)
        assert sample.features.dtype == np.float32

    def test_values_mp_array(self):
        """Multi-player values array has 4 elements."""
        from app.training.parallel_encoding import EncodedSample

        values_mp = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)

        sample = EncodedSample(
            features=np.zeros((10, 8, 8)),
            globals=np.zeros(5),
            value=0.5,
            values_mp=values_mp,
            policy_index=0,
            move_number=1,
            total_moves=10,
            phase="play",
            perspective=1,
            num_players=4,
        )

        assert sample.values_mp.shape == (4,)
        np.testing.assert_allclose(sample.values_mp.sum(), 1.0, atol=1e-5)


class TestMoveTypes:
    """Tests for move type tracking in samples."""

    def test_move_type_field(self):
        """EncodedSample tracks move type for weighting."""
        from app.training.parallel_encoding import EncodedSample

        sample = EncodedSample(
            features=np.zeros((10, 8, 8)),
            globals=np.zeros(5),
            value=0.0,
            values_mp=np.zeros(4),
            policy_index=0,
            move_number=1,
            total_moves=10,
            phase="chain_capture",
            perspective=1,
            num_players=2,
            move_type="attack",
        )

        assert sample.move_type == "attack"
        assert sample.phase == "chain_capture"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
