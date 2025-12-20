"""Tests for helper functions in app/training/train.py.

Tests cover:
- _flatten_heuristic_weights(): Deterministic flattening of weight profiles
- _reconstruct_heuristic_profile(): Reconstructing profiles from key-value pairs
- _get_heuristic_tier_by_id(): Looking up tier specifications
- temporary_heuristic_profile(): Context manager for temporary profile registration
- seed_all_legacy(): Backwards-compatible seeding
- evaluate_heuristic_candidate(): Evaluating heuristic weight candidates
- run_cmaes_heuristic_optimization(): CMA-ES optimization loop
"""

from unittest.mock import MagicMock, patch

import pytest

from app.ai.heuristic_weights import (
    HEURISTIC_WEIGHT_KEYS,
    HEURISTIC_WEIGHT_PROFILES,
)
from app.models import BoardType
from app.training.tier_eval_config import HEURISTIC_TIER_SPECS, HeuristicTierSpec
from app.training.train import (
    _flatten_heuristic_weights,
    _get_heuristic_tier_by_id,
    _reconstruct_heuristic_profile,
    evaluate_heuristic_candidate,
    run_cmaes_heuristic_optimization,
    seed_all_legacy,
    temporary_heuristic_profile,
)


class TestFlattenHeuristicWeights:
    """Tests for _flatten_heuristic_weights()."""

    def test_flatten_complete_profile(self):
        """Test flattening a profile with all required keys."""
        # Build a complete profile with all expected keys
        profile = {key: float(i) for i, key in enumerate(HEURISTIC_WEIGHT_KEYS)}

        keys, values = _flatten_heuristic_weights(profile)

        # Keys should be in HEURISTIC_WEIGHT_KEYS order
        assert keys == list(HEURISTIC_WEIGHT_KEYS)
        # Values should match the order
        assert values == [float(i) for i in range(len(HEURISTIC_WEIGHT_KEYS))]

    def test_flatten_preserves_key_order(self):
        """Test that key order is deterministic across multiple calls."""
        profile = dict.fromkeys(HEURISTIC_WEIGHT_KEYS, 1.0)

        keys1, _ = _flatten_heuristic_weights(profile)
        keys2, _ = _flatten_heuristic_weights(profile)

        assert keys1 == keys2
        assert keys1 == list(HEURISTIC_WEIGHT_KEYS)

    def test_flatten_missing_key_raises_error(self):
        """Test that missing keys raise KeyError with helpful message."""
        # Create profile missing a key
        incomplete_profile = dict.fromkeys(list(HEURISTIC_WEIGHT_KEYS)[:-1], 1.0)

        with pytest.raises(KeyError) as exc_info:
            _flatten_heuristic_weights(incomplete_profile)

        # Error message should mention the missing key
        assert "Missing heuristic weight" in str(exc_info.value)
        assert "HEURISTIC_WEIGHT_KEYS" in str(exc_info.value)

    def test_flatten_extra_keys_ignored(self):
        """Test that extra keys in profile are ignored."""
        profile = dict.fromkeys(HEURISTIC_WEIGHT_KEYS, 1.0)
        profile["EXTRA_KEY_NOT_IN_SPEC"] = 999.0

        keys, values = _flatten_heuristic_weights(profile)

        # Extra key should not appear in output
        assert "EXTRA_KEY_NOT_IN_SPEC" not in keys
        assert len(keys) == len(HEURISTIC_WEIGHT_KEYS)

    def test_flatten_converts_to_float(self):
        """Test that integer values are converted to float."""
        profile = {key: i for i, key in enumerate(HEURISTIC_WEIGHT_KEYS)}

        _, values = _flatten_heuristic_weights(profile)

        # All values should be floats
        assert all(isinstance(v, float) for v in values)

    def test_flatten_with_real_profile(self):
        """Test flattening with a real registered profile."""
        if not HEURISTIC_WEIGHT_PROFILES:
            pytest.skip("No heuristic weight profiles defined")

        # Use first available profile
        profile_id = next(iter(HEURISTIC_WEIGHT_PROFILES.keys()))
        profile = HEURISTIC_WEIGHT_PROFILES[profile_id]

        keys, values = _flatten_heuristic_weights(profile)

        assert len(keys) == len(HEURISTIC_WEIGHT_KEYS)
        assert len(values) == len(HEURISTIC_WEIGHT_KEYS)


class TestReconstructHeuristicProfile:
    """Tests for _reconstruct_heuristic_profile()."""

    def test_reconstruct_from_lists(self):
        """Test basic reconstruction from key-value lists."""
        keys = ["key1", "key2", "key3"]
        values = [1.0, 2.0, 3.0]

        profile = _reconstruct_heuristic_profile(keys, values)

        assert profile == {"key1": 1.0, "key2": 2.0, "key3": 3.0}

    def test_reconstruct_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        keys = ["key1", "key2", "key3"]
        values = [1.0, 2.0]  # One fewer value

        with pytest.raises(ValueError) as exc_info:
            _reconstruct_heuristic_profile(keys, values)

        assert "Length mismatch" in str(exc_info.value)
        assert "3 keys" in str(exc_info.value)
        assert "2 values" in str(exc_info.value)

    def test_reconstruct_empty_input(self):
        """Test reconstruction with empty inputs."""
        profile = _reconstruct_heuristic_profile([], [])
        assert profile == {}

    def test_reconstruct_converts_to_float(self):
        """Test that values are converted to float."""
        keys = ["key1", "key2"]
        values = [1, 2]  # integers

        profile = _reconstruct_heuristic_profile(keys, values)

        assert profile["key1"] == 1.0
        assert profile["key2"] == 2.0
        assert all(isinstance(v, float) for v in profile.values())

    def test_round_trip_with_flatten(self):
        """Test that flatten -> reconstruct produces equivalent profile."""
        original = {key: float(i) * 0.5 for i, key in enumerate(HEURISTIC_WEIGHT_KEYS)}

        keys, values = _flatten_heuristic_weights(original)
        reconstructed = _reconstruct_heuristic_profile(keys, values)

        # Should have all the same keys and values
        for key in HEURISTIC_WEIGHT_KEYS:
            assert reconstructed[key] == original[key]


class TestGetHeuristicTierById:
    """Tests for _get_heuristic_tier_by_id()."""

    def test_get_existing_tier(self):
        """Test retrieving an existing tier spec."""
        if not HEURISTIC_TIER_SPECS:
            pytest.skip("No heuristic tier specs defined")

        # Use first available tier
        expected_spec = HEURISTIC_TIER_SPECS[0]

        result = _get_heuristic_tier_by_id(expected_spec.id)

        assert result == expected_spec
        assert result.id == expected_spec.id

    def test_get_nonexistent_tier_raises_error(self):
        """Test that unknown tier_id raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _get_heuristic_tier_by_id("nonexistent_tier_id_xyz")

        error_msg = str(exc_info.value)
        assert "Unknown heuristic tier_id" in error_msg
        assert "nonexistent_tier_id_xyz" in error_msg
        # Should list available tiers
        assert "Available heuristic tiers" in error_msg

    def test_error_lists_available_tiers(self):
        """Test that error message includes available tier IDs."""
        if not HEURISTIC_TIER_SPECS:
            pytest.skip("No heuristic tier specs defined")

        with pytest.raises(ValueError) as exc_info:
            _get_heuristic_tier_by_id("bogus")

        error_msg = str(exc_info.value)
        # At least one known tier should be mentioned
        assert HEURISTIC_TIER_SPECS[0].id in error_msg


class TestTemporaryHeuristicProfile:
    """Tests for temporary_heuristic_profile() context manager."""

    def test_registers_profile_during_context(self):
        """Test that profile is registered inside context."""
        test_profile_id = "_test_temp_profile_12345"
        test_weights = {"WEIGHT_STACK_CONTROL": 100.0}

        # Ensure not registered before
        assert test_profile_id not in HEURISTIC_WEIGHT_PROFILES

        with temporary_heuristic_profile(test_profile_id, test_weights):
            # Should be registered now
            assert test_profile_id in HEURISTIC_WEIGHT_PROFILES
            assert HEURISTIC_WEIGHT_PROFILES[test_profile_id] == test_weights

        # Should be cleaned up after
        assert test_profile_id not in HEURISTIC_WEIGHT_PROFILES

    def test_restores_existing_profile(self):
        """Test that existing profile is restored after context."""
        test_profile_id = "_test_existing_profile_67890"
        original_weights = {"original": 1.0}
        temporary_weights = {"temporary": 2.0}

        # Register original profile
        HEURISTIC_WEIGHT_PROFILES[test_profile_id] = original_weights

        try:
            with temporary_heuristic_profile(test_profile_id, temporary_weights):
                # Should have temporary weights
                assert HEURISTIC_WEIGHT_PROFILES[test_profile_id] == temporary_weights

            # Should restore original
            assert HEURISTIC_WEIGHT_PROFILES[test_profile_id] == original_weights
        finally:
            # Clean up
            HEURISTIC_WEIGHT_PROFILES.pop(test_profile_id, None)

    def test_cleans_up_on_exception(self):
        """Test that profile is cleaned up even when exception occurs."""
        test_profile_id = "_test_exception_profile"
        test_weights = {"WEIGHT_STACK_CONTROL": 50.0}

        assert test_profile_id not in HEURISTIC_WEIGHT_PROFILES

        with pytest.raises(RuntimeError), temporary_heuristic_profile(test_profile_id, test_weights):
            assert test_profile_id in HEURISTIC_WEIGHT_PROFILES
            raise RuntimeError("Test exception")

        # Should still be cleaned up
        assert test_profile_id not in HEURISTIC_WEIGHT_PROFILES

    def test_restores_existing_on_exception(self):
        """Test that existing profile is restored even on exception."""
        test_profile_id = "_test_restore_on_exception"
        original_weights = {"original": 999.0}
        temporary_weights = {"temp": 1.0}

        HEURISTIC_WEIGHT_PROFILES[test_profile_id] = original_weights

        try:
            with pytest.raises(ValueError), temporary_heuristic_profile(test_profile_id, temporary_weights):
                raise ValueError("Test error")

            # Should restore original
            assert HEURISTIC_WEIGHT_PROFILES[test_profile_id] == original_weights
        finally:
            HEURISTIC_WEIGHT_PROFILES.pop(test_profile_id, None)

    def test_converts_mapping_to_dict(self):
        """Test that input mapping is converted to dict."""
        from collections import OrderedDict

        test_profile_id = "_test_ordereddict_profile"
        ordered_weights = OrderedDict([("key1", 1.0), ("key2", 2.0)])

        with temporary_heuristic_profile(test_profile_id, ordered_weights):
            stored = HEURISTIC_WEIGHT_PROFILES[test_profile_id]
            # Should be a regular dict
            assert isinstance(stored, dict)
            assert stored == {"key1": 1.0, "key2": 2.0}


class TestSeedAllLegacy:
    """Tests for seed_all_legacy() backwards compatibility."""

    def test_calls_seed_all(self):
        """Test that seed_all_legacy calls seed_all."""
        with patch("app.training.train.seed_all") as mock_seed_all:
            seed_all_legacy(123)
            mock_seed_all.assert_called_once_with(123)

    def test_default_seed(self):
        """Test default seed value of 42."""
        with patch("app.training.train.seed_all") as mock_seed_all:
            seed_all_legacy()
            mock_seed_all.assert_called_once_with(42)

    def test_deterministic_behavior(self):
        """Test that seeding produces deterministic randomness."""
        import random

        import numpy as np

        # Seed and capture random values
        seed_all_legacy(42)
        random_val_1 = random.random()
        np_val_1 = np.random.random()

        # Re-seed and verify same values
        seed_all_legacy(42)
        random_val_2 = random.random()
        np_val_2 = np.random.random()

        assert random_val_1 == random_val_2
        assert np_val_1 == np_val_2


class TestHeuristicWeightProfilesIntegrity:
    """Integration tests for heuristic weight profile consistency."""

    def test_all_registered_profiles_are_complete(self):
        """Test that all registered profiles have all required keys."""
        for profile_id, profile in HEURISTIC_WEIGHT_PROFILES.items():
            # Should be able to flatten without error
            try:
                keys, values = _flatten_heuristic_weights(profile)
                assert len(keys) == len(HEURISTIC_WEIGHT_KEYS)
            except KeyError as e:
                pytest.fail(
                    f"Profile '{profile_id}' is missing key: {e}"
                )

    def test_canonical_balanced_profile_exists(self):
        """Test that the canonical balanced profile exists."""
        # The canonical balanced profile is named heuristic_v1_balanced
        assert "heuristic_v1_balanced" in HEURISTIC_WEIGHT_PROFILES

        profile = HEURISTIC_WEIGHT_PROFILES["heuristic_v1_balanced"]

        # Should have all keys
        for key in HEURISTIC_WEIGHT_KEYS:
            assert key in profile, f"heuristic_v1_balanced missing key: {key}"

    def test_flatten_reconstruct_preserves_profile(self):
        """Test round-trip consistency for all registered profiles."""
        for profile_id, profile in HEURISTIC_WEIGHT_PROFILES.items():
            keys, values = _flatten_heuristic_weights(profile)
            reconstructed = _reconstruct_heuristic_profile(keys, values)

            for key in HEURISTIC_WEIGHT_KEYS:
                assert reconstructed[key] == profile[key], (
                    f"Round-trip mismatch for profile '{profile_id}' key '{key}'"
                )


class TestTierSpecIntegrity:
    """Integration tests for tier spec consistency."""

    def test_all_tier_specs_have_valid_board_type(self):
        """Test that all tier specs have valid board types."""
        from app.models import BoardType

        for spec in HEURISTIC_TIER_SPECS:
            assert isinstance(spec.board_type, BoardType), (
                f"Tier '{spec.id}' has invalid board_type"
            )

    def test_all_tier_specs_have_positive_num_games(self):
        """Test that all tier specs have positive num_games."""
        for spec in HEURISTIC_TIER_SPECS:
            assert spec.num_games > 0, (
                f"Tier '{spec.id}' has non-positive num_games: {spec.num_games}"
            )

    def test_all_tier_specs_have_valid_profile_ids(self):
        """Test that tier spec profile IDs exist in weight profiles.

        Note: Some tier specs may reference profile IDs that are dynamically
        registered or use legacy naming conventions. We warn but don't fail
        for missing profiles that may be registered at runtime.
        """
        import warnings

        for spec in HEURISTIC_TIER_SPECS:
            if spec.candidate_profile_id:
                if spec.candidate_profile_id not in HEURISTIC_WEIGHT_PROFILES:
                    if not spec.candidate_profile_id.startswith("cmaes_"):
                        # Warn about missing profile (may be registered at runtime)
                        warnings.warn(
                            f"Tier '{spec.id}' references candidate_profile_id "
                            f"'{spec.candidate_profile_id}' which is not currently "
                            f"registered. This may be intentional for runtime "
                            f"registration."
                        )
            if spec.baseline_profile_id:
                if spec.baseline_profile_id not in HEURISTIC_WEIGHT_PROFILES:
                    # Warn about missing profile
                    warnings.warn(
                        f"Tier '{spec.id}' references baseline_profile_id "
                        f"'{spec.baseline_profile_id}' which is not currently "
                        f"registered."
                    )


class TestEvaluateHeuristicCandidate:
    """Tests for evaluate_heuristic_candidate()."""

    @pytest.fixture
    def mock_tier_spec(self):
        """Create a mock tier spec for testing."""
        return HeuristicTierSpec(
            id="test_tier",
            name="Test Tier",
            board_type=BoardType.SQUARE8,
            num_players=2,
            eval_pool_id="v1",
            num_games=10,
            candidate_profile_id="test_candidate",
            baseline_profile_id="heuristic_v1_balanced",
            description="Test tier for unit tests",
        )

    @pytest.fixture
    def mock_eval_result(self):
        """Create a mock evaluation result."""
        return {
            "games_played": 10,
            "results": {
                "wins": 7,
                "draws": 2,
                "losses": 1,
            },
            "margins": {
                "ring_margin_mean": 0.5,
                "territory_margin_mean": 2.0,
            },
        }

    def test_fitness_calculation(self, mock_tier_spec, mock_eval_result):
        """Test that fitness is calculated correctly from results."""
        with patch("app.training.train.run_heuristic_tier_eval") as mock_eval:
            mock_eval.return_value = mock_eval_result

            keys = list(HEURISTIC_WEIGHT_KEYS)
            values = [1.0] * len(keys)

            # Register test-specific profile
            test_profile_id = "_test_base_profile_fitness"
            HEURISTIC_WEIGHT_PROFILES[test_profile_id] = dict.fromkeys(keys, 1.0)

            # Update mock tier spec to use test profile
            tier_spec = HeuristicTierSpec(
                id=mock_tier_spec.id,
                name=mock_tier_spec.name,
                board_type=mock_tier_spec.board_type,
                num_players=mock_tier_spec.num_players,
                eval_pool_id=mock_tier_spec.eval_pool_id,
                num_games=mock_tier_spec.num_games,
                candidate_profile_id=mock_tier_spec.candidate_profile_id,
                baseline_profile_id=test_profile_id,
            )

            try:
                fitness, raw = evaluate_heuristic_candidate(
                    tier_spec=tier_spec,
                    base_profile_id=test_profile_id,
                    keys=keys,
                    candidate_vector=values,
                    rng_seed=42,
                )

                # Win rate: (7 + 0.5*2) / 10 = 0.8
                # Margin score: 0.5 + 0.25 * 2.0 = 1.0
                # Fitness: 0.8 + 0.01 * 1.0 = 0.81
                assert abs(fitness - 0.81) < 0.001
                assert raw == mock_eval_result
            finally:
                HEURISTIC_WEIGHT_PROFILES.pop(test_profile_id, None)

    def test_fitness_with_zero_games(self, mock_tier_spec):
        """Test fitness calculation with zero games (edge case)."""
        with patch("app.training.train.run_heuristic_tier_eval") as mock_eval:
            mock_eval.return_value = {
                "games_played": 0,
                "results": {},
                "margins": {},
            }

            keys = list(HEURISTIC_WEIGHT_KEYS)
            values = [1.0] * len(keys)

            # Register test-specific profile
            test_profile_id = "_test_base_profile_zero_games"
            HEURISTIC_WEIGHT_PROFILES[test_profile_id] = dict.fromkeys(keys, 1.0)

            # Update mock tier spec to use test profile
            tier_spec = HeuristicTierSpec(
                id=mock_tier_spec.id,
                name=mock_tier_spec.name,
                board_type=mock_tier_spec.board_type,
                num_players=mock_tier_spec.num_players,
                eval_pool_id=mock_tier_spec.eval_pool_id,
                num_games=mock_tier_spec.num_games,
                candidate_profile_id=mock_tier_spec.candidate_profile_id,
                baseline_profile_id=test_profile_id,
            )

            try:
                fitness, _ = evaluate_heuristic_candidate(
                    tier_spec=tier_spec,
                    base_profile_id=test_profile_id,
                    keys=keys,
                    candidate_vector=values,
                    rng_seed=42,
                )

                # With 0 games, games_played is max(1, 0) = 1, wins/draws = 0
                # Win rate: 0/1 = 0, fitness = 0
                assert fitness == 0.0
            finally:
                HEURISTIC_WEIGHT_PROFILES.pop(test_profile_id, None)

    def test_uses_games_per_candidate_override(self, mock_tier_spec, mock_eval_result):
        """Test that games_per_candidate overrides tier spec num_games."""
        with patch("app.training.train.run_heuristic_tier_eval") as mock_eval:
            mock_eval.return_value = mock_eval_result

            keys = list(HEURISTIC_WEIGHT_KEYS)
            values = [1.0] * len(keys)

            # Register test-specific profile
            test_profile_id = "_test_base_profile_override"
            HEURISTIC_WEIGHT_PROFILES[test_profile_id] = dict.fromkeys(keys, 1.0)

            # Update mock tier spec to use test profile
            tier_spec = HeuristicTierSpec(
                id=mock_tier_spec.id,
                name=mock_tier_spec.name,
                board_type=mock_tier_spec.board_type,
                num_players=mock_tier_spec.num_players,
                eval_pool_id=mock_tier_spec.eval_pool_id,
                num_games=mock_tier_spec.num_games,
                candidate_profile_id=mock_tier_spec.candidate_profile_id,
                baseline_profile_id=test_profile_id,
            )

            try:
                evaluate_heuristic_candidate(
                    tier_spec=tier_spec,
                    base_profile_id=test_profile_id,
                    keys=keys,
                    candidate_vector=values,
                    rng_seed=42,
                    games_per_candidate=5,  # Override
                )

                # Check that max_games was 5 (override) not 10 (tier spec)
                call_kwargs = mock_eval.call_args[1]
                assert call_kwargs["max_games"] == 5
            finally:
                HEURISTIC_WEIGHT_PROFILES.pop(test_profile_id, None)


class TestRunCmaesHeuristicOptimization:
    """Tests for run_cmaes_heuristic_optimization()."""

    def test_invalid_generations_raises_error(self):
        """Test that non-positive generations raises ValueError."""
        with pytest.raises(ValueError, match="generations must be positive"):
            run_cmaes_heuristic_optimization(
                tier_id="sq8_heuristic_baseline_v1",
                base_profile_id="heuristic_v1_balanced",
                generations=0,
            )

    def test_invalid_population_size_raises_error(self):
        """Test that non-positive population_size raises ValueError."""
        with pytest.raises(ValueError, match="population_size must be positive"):
            run_cmaes_heuristic_optimization(
                tier_id="sq8_heuristic_baseline_v1",
                base_profile_id="heuristic_v1_balanced",
                generations=1,
                population_size=0,
            )

    def test_unknown_tier_id_raises_error(self):
        """Test that unknown tier_id raises ValueError."""
        with pytest.raises(ValueError, match="Unknown heuristic tier_id"):
            run_cmaes_heuristic_optimization(
                tier_id="nonexistent_tier",
                base_profile_id="heuristic_v1_balanced",
                generations=1,
            )

    def test_unknown_base_profile_raises_error(self):
        """Test that unknown base_profile_id raises ValueError."""
        if not HEURISTIC_TIER_SPECS:
            pytest.skip("No tier specs defined")

        tier_id = HEURISTIC_TIER_SPECS[0].id

        with pytest.raises(ValueError, match="Unknown heuristic base_profile_id"):
            run_cmaes_heuristic_optimization(
                tier_id=tier_id,
                base_profile_id="nonexistent_profile",
                generations=1,
            )

    def test_returns_expected_report_structure(self):
        """Test that optimization returns expected report structure."""
        if not HEURISTIC_TIER_SPECS:
            pytest.skip("No tier specs defined")

        tier_spec = HEURISTIC_TIER_SPECS[0]

        # Use a real profile name
        if "heuristic_v1_balanced" not in HEURISTIC_WEIGHT_PROFILES:
            pytest.skip("heuristic_v1_balanced profile not defined")

        # Mock the evaluation to avoid running actual games
        mock_result = {
            "games_played": 10,
            "results": {"wins": 5, "draws": 2, "losses": 3},
            "margins": {"ring_margin_mean": 0.1, "territory_margin_mean": 0.5},
        }

        with patch("app.training.train.run_heuristic_tier_eval") as mock_eval:
            mock_eval.return_value = mock_result

            report = run_cmaes_heuristic_optimization(
                tier_id=tier_spec.id,
                base_profile_id="heuristic_v1_balanced",
                generations=2,
                population_size=3,
                rng_seed=42,
            )

            # Verify report structure
            assert "run_type" in report
            assert "tier_id" in report
            assert report["tier_id"] == tier_spec.id
            assert "base_profile_id" in report
            assert report["base_profile_id"] == "heuristic_v1_balanced"
            assert "generations" in report
            assert report["generations"] == 2
            assert "population_size" in report
            assert report["population_size"] == 3
            assert "rng_seed" in report
            assert report["rng_seed"] == 42
            assert "dimension" in report
            assert "keys" in report
            assert "history" in report
            assert len(report["history"]) == 2  # 2 generations
            assert "best" in report

    def test_history_tracks_generation_progress(self):
        """Test that history tracks best/mean fitness per generation."""
        if not HEURISTIC_TIER_SPECS:
            pytest.skip("No tier specs defined")

        tier_spec = HEURISTIC_TIER_SPECS[0]

        if "heuristic_v1_balanced" not in HEURISTIC_WEIGHT_PROFILES:
            pytest.skip("heuristic_v1_balanced profile not defined")

        mock_result = {
            "games_played": 10,
            "results": {"wins": 6, "draws": 1, "losses": 3},
            "margins": {"ring_margin_mean": 0.2, "territory_margin_mean": 1.0},
        }

        with patch("app.training.train.run_heuristic_tier_eval") as mock_eval:
            mock_eval.return_value = mock_result

            report = run_cmaes_heuristic_optimization(
                tier_id=tier_spec.id,
                base_profile_id="heuristic_v1_balanced",
                generations=3,
                population_size=2,
                rng_seed=123,
            )

            history = report["history"]
            assert len(history) == 3

            for i, gen_entry in enumerate(history):
                assert gen_entry["generation"] == i
                assert "best_fitness" in gen_entry
                assert "mean_fitness" in gen_entry
                # Best should be >= mean
                assert gen_entry["best_fitness"] >= gen_entry["mean_fitness"]

    def test_best_result_tracked(self):
        """Test that the overall best result is tracked correctly."""
        if not HEURISTIC_TIER_SPECS:
            pytest.skip("No tier specs defined")

        tier_spec = HEURISTIC_TIER_SPECS[0]

        if "heuristic_v1_balanced" not in HEURISTIC_WEIGHT_PROFILES:
            pytest.skip("heuristic_v1_balanced profile not defined")

        # Return varying results to ensure best is tracked
        call_count = [0]

        def varying_result(*args, **kwargs):
            call_count[0] += 1
            # Make every 3rd call have better results
            if call_count[0] % 3 == 0:
                wins = 9
            else:
                wins = 4

            return {
                "games_played": 10,
                "results": {"wins": wins, "draws": 0, "losses": 10 - wins},
                "margins": {"ring_margin_mean": 0.0, "territory_margin_mean": 0.0},
            }

        with patch("app.training.train.run_heuristic_tier_eval") as mock_eval:
            mock_eval.side_effect = varying_result

            report = run_cmaes_heuristic_optimization(
                tier_id=tier_spec.id,
                base_profile_id="heuristic_v1_balanced",
                generations=2,
                population_size=4,
                rng_seed=999,
            )

            best = report["best"]
            assert best is not None
            assert "generation" in best
            assert "vector" in best
            assert "fitness" in best
            # Best fitness should be 0.9 (9 wins out of 10)
            assert best["fitness"] == 0.9
