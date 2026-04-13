"""Tests for app.training.heuristic_tuning module.

Tests the CMA-ES heuristic weight optimization:
- Flatten/reconstruct heuristic profiles
- Temporary heuristic profile context manager
- Heuristic tier lookup
- Candidate evaluation (mocked)
- CMA-ES optimization loop (basic structure)

Created Dec 2025 as part of Phase 3 test coverage improvement.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.training.tier_eval_config import HEURISTIC_TIER_SPECS


def _assert_has_tier_specs() -> None:
    assert HEURISTIC_TIER_SPECS, "Expected canonical heuristic tier specs"


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Tests for module imports."""

    def test_import_module(self):
        """Test that the module can be imported."""
        from app.training import heuristic_tuning
        assert heuristic_tuning is not None

    def test_import_public_functions(self):
        """Test that public functions can be imported."""
        from app.training.heuristic_tuning import (
            evaluate_heuristic_candidate,
            run_cmaes_heuristic_optimization,
            temporary_heuristic_profile,
        )
        assert evaluate_heuristic_candidate is not None
        assert run_cmaes_heuristic_optimization is not None
        assert temporary_heuristic_profile is not None

    def test_import_from_train(self):
        """Test backward compatibility imports from train.py."""
        from app.training.train import (
            evaluate_heuristic_candidate,
            run_cmaes_heuristic_optimization,
            temporary_heuristic_profile,
        )
        assert evaluate_heuristic_candidate is not None
        assert run_cmaes_heuristic_optimization is not None
        assert temporary_heuristic_profile is not None


# =============================================================================
# Private Function Tests
# =============================================================================


class TestFlattenHeuristicWeights:
    """Tests for _flatten_heuristic_weights function."""

    def test_flatten_valid_profile(self):
        """Test flattening a valid heuristic profile."""
        from app.training.heuristic_tuning import _flatten_heuristic_weights
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_KEYS

        # Create a profile with all required keys
        profile = {k: float(i) for i, k in enumerate(HEURISTIC_WEIGHT_KEYS)}

        keys, values = _flatten_heuristic_weights(profile)

        assert len(keys) == len(HEURISTIC_WEIGHT_KEYS)
        assert len(values) == len(HEURISTIC_WEIGHT_KEYS)
        assert all(isinstance(v, float) for v in values)

    def test_flatten_missing_key_raises(self):
        """Test that missing keys raise KeyError."""
        from app.training.heuristic_tuning import _flatten_heuristic_weights

        # Empty profile should fail
        with pytest.raises(KeyError) as exc_info:
            _flatten_heuristic_weights({})

        assert "Missing heuristic weight" in str(exc_info.value)

    def test_flatten_preserves_order(self):
        """Test that flattening preserves HEURISTIC_WEIGHT_KEYS order."""
        from app.training.heuristic_tuning import _flatten_heuristic_weights
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_KEYS

        profile = {k: float(i) for i, k in enumerate(HEURISTIC_WEIGHT_KEYS)}
        keys, values = _flatten_heuristic_weights(profile)

        assert keys == list(HEURISTIC_WEIGHT_KEYS)


class TestReconstructHeuristicProfile:
    """Tests for _reconstruct_heuristic_profile function."""

    def test_reconstruct_valid(self):
        """Test reconstructing a valid profile."""
        from app.training.heuristic_tuning import _reconstruct_heuristic_profile

        keys = ["key1", "key2", "key3"]
        values = [1.0, 2.0, 3.0]

        profile = _reconstruct_heuristic_profile(keys, values)

        assert profile == {"key1": 1.0, "key2": 2.0, "key3": 3.0}

    def test_reconstruct_length_mismatch_raises(self):
        """Test that length mismatch raises ValueError."""
        from app.training.heuristic_tuning import _reconstruct_heuristic_profile

        keys = ["key1", "key2"]
        values = [1.0, 2.0, 3.0]  # Too many values

        with pytest.raises(ValueError) as exc_info:
            _reconstruct_heuristic_profile(keys, values)

        assert "Length mismatch" in str(exc_info.value)

    def test_reconstruct_converts_to_float(self):
        """Test that values are converted to float."""
        from app.training.heuristic_tuning import _reconstruct_heuristic_profile

        keys = ["key1", "key2"]
        values = [1, 2]  # Integers

        profile = _reconstruct_heuristic_profile(keys, values)

        assert all(isinstance(v, float) for v in profile.values())


# =============================================================================
# Temporary Profile Tests
# =============================================================================


class TestTemporaryHeuristicProfile:
    """Tests for temporary_heuristic_profile context manager."""

    def test_temporary_profile_registered(self):
        """Test that temporary profile is registered during context."""
        from app.training.heuristic_tuning import temporary_heuristic_profile
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES

        profile_id = "test_temp_profile"
        weights = {"test_key": 1.0}

        # Ensure not in registry before
        if profile_id in HEURISTIC_WEIGHT_PROFILES:
            del HEURISTIC_WEIGHT_PROFILES[profile_id]

        with temporary_heuristic_profile(profile_id, weights):
            assert profile_id in HEURISTIC_WEIGHT_PROFILES
            assert HEURISTIC_WEIGHT_PROFILES[profile_id] == {"test_key": 1.0}

        # Should be removed after context
        assert profile_id not in HEURISTIC_WEIGHT_PROFILES

    def test_temporary_profile_restores_existing(self):
        """Test that existing profile is restored after context."""
        from app.training.heuristic_tuning import temporary_heuristic_profile
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES

        profile_id = "test_existing_profile"
        original_weights = {"original": 999.0}
        temp_weights = {"temp": 1.0}

        # Set up existing profile
        HEURISTIC_WEIGHT_PROFILES[profile_id] = original_weights

        try:
            with temporary_heuristic_profile(profile_id, temp_weights):
                assert HEURISTIC_WEIGHT_PROFILES[profile_id] == temp_weights

            # Original should be restored
            assert HEURISTIC_WEIGHT_PROFILES[profile_id] == original_weights
        finally:
            # Cleanup
            if profile_id in HEURISTIC_WEIGHT_PROFILES:
                del HEURISTIC_WEIGHT_PROFILES[profile_id]

    def test_temporary_profile_handles_exception(self):
        """Test that profile is cleaned up even on exception."""
        from app.training.heuristic_tuning import temporary_heuristic_profile
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES

        profile_id = "test_exception_profile"
        weights = {"test": 1.0}

        if profile_id in HEURISTIC_WEIGHT_PROFILES:
            del HEURISTIC_WEIGHT_PROFILES[profile_id]

        with pytest.raises(RuntimeError):
            with temporary_heuristic_profile(profile_id, weights):
                raise RuntimeError("Test exception")

        # Should still be cleaned up
        assert profile_id not in HEURISTIC_WEIGHT_PROFILES


# =============================================================================
# Tier Lookup Tests
# =============================================================================


class TestGetHeuristicTierById:
    """Tests for _get_heuristic_tier_by_id function."""

    def test_lookup_existing_tier(self):
        """Test looking up an existing tier."""
        from app.training.heuristic_tuning import _get_heuristic_tier_by_id

        _assert_has_tier_specs()

        first_tier = HEURISTIC_TIER_SPECS[0]
        result = _get_heuristic_tier_by_id(first_tier.id)

        assert result.id == first_tier.id

    def test_lookup_nonexistent_tier_raises(self):
        """Test that looking up nonexistent tier raises ValueError."""
        from app.training.heuristic_tuning import _get_heuristic_tier_by_id

        with pytest.raises(ValueError) as exc_info:
            _get_heuristic_tier_by_id("nonexistent_tier_id_xyz")

        assert "Unknown heuristic tier_id" in str(exc_info.value)


# =============================================================================
# CMA-ES Validation Tests
# =============================================================================


class TestRunCmaesValidation:
    """Tests for run_cmaes_heuristic_optimization input validation."""

    def test_zero_generations_raises(self):
        """Test that zero generations raises ValueError."""
        from app.training.heuristic_tuning import run_cmaes_heuristic_optimization

        with pytest.raises(ValueError) as exc_info:
            run_cmaes_heuristic_optimization(
                tier_id="test",
                base_profile_id="test",
                generations=0,
            )

        assert "generations must be positive" in str(exc_info.value)

    def test_negative_generations_raises(self):
        """Test that negative generations raises ValueError."""
        from app.training.heuristic_tuning import run_cmaes_heuristic_optimization

        with pytest.raises(ValueError) as exc_info:
            run_cmaes_heuristic_optimization(
                tier_id="test",
                base_profile_id="test",
                generations=-5,
            )

        assert "generations must be positive" in str(exc_info.value)

    def test_zero_population_raises(self):
        """Test that zero population size raises ValueError."""
        from app.training.heuristic_tuning import run_cmaes_heuristic_optimization

        with pytest.raises(ValueError) as exc_info:
            run_cmaes_heuristic_optimization(
                tier_id="test",
                base_profile_id="test",
                generations=5,
                population_size=0,
            )

        assert "population_size must be positive" in str(exc_info.value)

    def test_unknown_base_profile_raises(self):
        """Test that unknown base profile raises ValueError."""
        from app.training.heuristic_tuning import run_cmaes_heuristic_optimization

        _assert_has_tier_specs()

        tier_id = HEURISTIC_TIER_SPECS[0].id

        with pytest.raises(ValueError) as exc_info:
            run_cmaes_heuristic_optimization(
                tier_id=tier_id,
                base_profile_id="nonexistent_profile_xyz",
                generations=1,
            )

        assert "Unknown heuristic base_profile_id" in str(exc_info.value)


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================


class TestEvaluateHeuristicCandidate:
    """Tests for evaluate_heuristic_candidate with mocked evaluation."""

    @patch("app.training.heuristic_tuning.run_heuristic_tier_eval")
    def test_evaluate_returns_fitness_and_result(self, mock_eval):
        """Test that evaluation returns fitness and raw result."""
        from app.training.heuristic_tuning import evaluate_heuristic_candidate
        from app.training.tier_eval_config import HeuristicTierSpec

        _assert_has_tier_specs()

        # Mock the evaluation result
        mock_eval.return_value = {
            "games_played": 10,
            "results": {"wins": 7, "draws": 1, "losses": 2},
            "margins": {"ring_margin_mean": 2.5, "territory_margin_mean": 1.0},
        }

        tier_spec = HEURISTIC_TIER_SPECS[0]
        keys = ["key1", "key2"]
        values = [1.0, 2.0]

        fitness, result = evaluate_heuristic_candidate(
            tier_spec=tier_spec,
            base_profile_id="test_base",
            keys=keys,
            candidate_vector=values,
            rng_seed=42,
        )

        assert isinstance(fitness, float)
        assert fitness > 0  # Win rate should give positive fitness
        assert "games_played" in result
        assert mock_eval.called

    @patch("app.training.heuristic_tuning.run_heuristic_tier_eval")
    def test_fitness_calculation(self, mock_eval):
        """Test fitness calculation from results."""
        from app.training.heuristic_tuning import evaluate_heuristic_candidate

        _assert_has_tier_specs()

        # 100% win rate
        mock_eval.return_value = {
            "games_played": 10,
            "results": {"wins": 10, "draws": 0, "losses": 0},
            "margins": {"ring_margin_mean": 0.0, "territory_margin_mean": 0.0},
        }

        tier_spec = HEURISTIC_TIER_SPECS[0]

        fitness, _ = evaluate_heuristic_candidate(
            tier_spec=tier_spec,
            base_profile_id="test_base",
            keys=["k1"],
            candidate_vector=[1.0],
            rng_seed=42,
        )

        # Win rate = 1.0, margins = 0, so fitness should be ~1.0
        assert 0.99 <= fitness <= 1.01
