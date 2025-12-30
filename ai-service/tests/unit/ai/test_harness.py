"""Unit tests for AI harness abstraction layer.

Tests the harness system which provides unified interface for evaluating
AI models under different search algorithms (Gumbel MCTS, Minimax, etc.).

Dec 2025: Phase 1 testing for NN/NNUE multi-harness evaluation system.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestHarnessType:
    """Tests for HarnessType enum."""

    def test_all_harness_types_exist(self):
        """Test all expected harness types are defined."""
        from app.ai.harness.base_harness import HarnessType

        expected_types = [
            "GUMBEL_MCTS",
            "GPU_GUMBEL",
            "MINIMAX",
            "MAXN",
            "BRS",
            "POLICY_ONLY",
            "DESCENT",
            "HEURISTIC",
        ]
        for name in expected_types:
            assert hasattr(HarnessType, name)

    def test_harness_type_values(self):
        """Test harness type string values."""
        from app.ai.harness.base_harness import HarnessType

        assert HarnessType.GUMBEL_MCTS.value == "gumbel_mcts"
        assert HarnessType.MINIMAX.value == "minimax"
        assert HarnessType.MAXN.value == "maxn"
        assert HarnessType.BRS.value == "brs"
        assert HarnessType.POLICY_ONLY.value == "policy_only"

    def test_harness_type_comparison(self):
        """Test harness type enum comparison."""
        from app.ai.harness.base_harness import HarnessType

        assert HarnessType.GUMBEL_MCTS == HarnessType.GUMBEL_MCTS
        assert HarnessType.MINIMAX != HarnessType.MAXN


class TestModelType:
    """Tests for ModelType enum."""

    def test_all_model_types_exist(self):
        """Test all expected model types are defined."""
        from app.ai.harness.base_harness import ModelType

        expected_types = ["NEURAL_NET", "NNUE", "HEURISTIC"]
        for name in expected_types:
            assert hasattr(ModelType, name)

    def test_model_type_values(self):
        """Test model type string values."""
        from app.ai.harness.base_harness import ModelType

        assert ModelType.NEURAL_NET.value == "nn"
        assert ModelType.NNUE.value == "nnue"
        assert ModelType.HEURISTIC.value == "heuristic"


class TestHarnessConfig:
    """Tests for HarnessConfig dataclass."""

    def test_default_config(self):
        """Test default HarnessConfig values."""
        from app.ai.harness.base_harness import HarnessConfig, HarnessType, ModelType

        config = HarnessConfig()
        assert config.harness_type == HarnessType.GUMBEL_MCTS
        assert config.model_type == ModelType.NEURAL_NET
        assert config.model_path is None
        assert config.model_id == ""
        assert config.board_type is None
        assert config.num_players == 2
        assert config.difficulty == 5
        assert config.think_time_ms is None
        assert config.simulations == 200
        assert config.depth == 3
        assert config.extra == {}

    def test_config_with_custom_values(self):
        """Test HarnessConfig with custom values."""
        from app.ai.harness.base_harness import HarnessConfig, HarnessType, ModelType

        config = HarnessConfig(
            harness_type=HarnessType.MINIMAX,
            model_type=ModelType.NNUE,
            model_path="/path/to/model.pth",
            model_id="nnue_v1",
            num_players=4,
            difficulty=7,
            simulations=400,
            depth=5,
            extra={"alpha_beta": True},
        )
        assert config.harness_type == HarnessType.MINIMAX
        assert config.model_type == ModelType.NNUE
        assert config.model_path == "/path/to/model.pth"
        assert config.model_id == "nnue_v1"
        assert config.num_players == 4
        assert config.difficulty == 7
        assert config.simulations == 400
        assert config.depth == 5
        assert config.extra == {"alpha_beta": True}

    def test_config_hash_mcts(self):
        """Test config hash for MCTS harness (includes simulations)."""
        from app.ai.harness.base_harness import HarnessConfig, HarnessType

        config1 = HarnessConfig(
            harness_type=HarnessType.GUMBEL_MCTS,
            difficulty=5,
            simulations=200,
        )
        config2 = HarnessConfig(
            harness_type=HarnessType.GUMBEL_MCTS,
            difficulty=5,
            simulations=400,
        )
        # Different simulations = different hash
        assert config1.get_config_hash() != config2.get_config_hash()

    def test_config_hash_minimax(self):
        """Test config hash for Minimax harness (includes depth)."""
        from app.ai.harness.base_harness import HarnessConfig, HarnessType

        config1 = HarnessConfig(
            harness_type=HarnessType.MINIMAX,
            difficulty=5,
            depth=3,
        )
        config2 = HarnessConfig(
            harness_type=HarnessType.MINIMAX,
            difficulty=5,
            depth=5,
        )
        # Different depth = different hash
        assert config1.get_config_hash() != config2.get_config_hash()

    def test_config_hash_same_config(self):
        """Test identical configs have same hash."""
        from app.ai.harness.base_harness import HarnessConfig, HarnessType

        config1 = HarnessConfig(
            harness_type=HarnessType.GUMBEL_MCTS,
            difficulty=5,
            simulations=200,
        )
        config2 = HarnessConfig(
            harness_type=HarnessType.GUMBEL_MCTS,
            difficulty=5,
            simulations=200,
        )
        assert config1.get_config_hash() == config2.get_config_hash()

    def test_config_hash_is_md5_prefix(self):
        """Test config hash is 8-character MD5 prefix."""
        from app.ai.harness.base_harness import HarnessConfig

        config = HarnessConfig()
        hash_val = config.get_config_hash()
        assert len(hash_val) == 8
        # Should be hex characters
        assert all(c in "0123456789abcdef" for c in hash_val)


class TestEvaluationMetadata:
    """Tests for EvaluationMetadata dataclass."""

    def test_default_metadata(self):
        """Test default EvaluationMetadata values."""
        from app.ai.harness.evaluation_metadata import EvaluationMetadata

        metadata = EvaluationMetadata()
        assert metadata.value_estimate == 0.0
        assert metadata.visit_distribution is None
        assert metadata.policy_distribution is None
        assert metadata.search_depth is None
        assert metadata.nodes_visited == 0
        assert metadata.time_ms == 0.0
        assert metadata.harness_type == ""
        assert metadata.model_type == ""
        assert metadata.model_id == ""
        assert metadata.config_hash == ""
        assert metadata.simulations is None
        assert metadata.extra == {}

    def test_metadata_with_values(self):
        """Test EvaluationMetadata with full values."""
        from app.ai.harness.evaluation_metadata import EvaluationMetadata

        visit_dist = {"move_a": 0.7, "move_b": 0.3}
        policy_dist = {"move_a": 0.6, "move_b": 0.4}

        metadata = EvaluationMetadata(
            value_estimate=0.85,
            visit_distribution=visit_dist,
            policy_distribution=policy_dist,
            search_depth=12,
            nodes_visited=5000,
            time_ms=125.5,
            harness_type="gumbel_mcts",
            model_type="nn",
            model_id="canonical_hex8_2p",
            config_hash="a1b2c3d4",
            simulations=400,
            extra={"temperature": 1.0},
        )
        assert metadata.value_estimate == 0.85
        assert metadata.visit_distribution == visit_dist
        assert metadata.policy_distribution == policy_dist
        assert metadata.search_depth == 12
        assert metadata.nodes_visited == 5000
        assert metadata.time_ms == 125.5
        assert metadata.harness_type == "gumbel_mcts"
        assert metadata.model_type == "nn"
        assert metadata.model_id == "canonical_hex8_2p"
        assert metadata.config_hash == "a1b2c3d4"
        assert metadata.simulations == 400
        assert metadata.extra == {"temperature": 1.0}

    def test_metadata_to_dict(self):
        """Test metadata conversion to dictionary."""
        from app.ai.harness.evaluation_metadata import EvaluationMetadata

        metadata = EvaluationMetadata(
            value_estimate=0.5,
            time_ms=100.0,
            harness_type="minimax",
            model_type="nnue",
        )
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["value_estimate"] == 0.5
        assert data["time_ms"] == 100.0
        assert data["harness_type"] == "minimax"
        assert data["model_type"] == "nnue"


class TestHarnessCompatibility:
    """Tests for HarnessCompatibility matrix."""

    def test_harness_compatibility_exists(self):
        """Test HARNESS_COMPATIBILITY dict exists."""
        from app.ai.harness.harness_registry import HARNESS_COMPATIBILITY

        assert isinstance(HARNESS_COMPATIBILITY, dict)
        assert len(HARNESS_COMPATIBILITY) > 0

    def test_gumbel_mcts_compatibility(self):
        """Test Gumbel MCTS compatibility (NN only, requires policy)."""
        from app.ai.harness.base_harness import HarnessType
        from app.ai.harness.harness_registry import HARNESS_COMPATIBILITY

        compat = HARNESS_COMPATIBILITY[HarnessType.GUMBEL_MCTS]
        assert compat.supports_nn is True
        assert compat.supports_nnue is False  # Needs policy head
        assert compat.requires_policy_head is True

    def test_minimax_compatibility(self):
        """Test Minimax compatibility (supports both NN and NNUE)."""
        from app.ai.harness.base_harness import HarnessType
        from app.ai.harness.harness_registry import HARNESS_COMPATIBILITY

        compat = HARNESS_COMPATIBILITY[HarnessType.MINIMAX]
        assert compat.supports_nn is True
        assert compat.supports_nnue is True
        assert compat.requires_policy_head is False

    def test_maxn_compatibility(self):
        """Test MaxN compatibility (supports both NN and NNUE)."""
        from app.ai.harness.base_harness import HarnessType
        from app.ai.harness.harness_registry import HARNESS_COMPATIBILITY

        compat = HARNESS_COMPATIBILITY[HarnessType.MAXN]
        assert compat.supports_nn is True
        assert compat.supports_nnue is True
        assert compat.requires_policy_head is False

    def test_brs_compatibility(self):
        """Test BRS (Best Reply Search) compatibility."""
        from app.ai.harness.base_harness import HarnessType
        from app.ai.harness.harness_registry import HARNESS_COMPATIBILITY

        compat = HARNESS_COMPATIBILITY[HarnessType.BRS]
        assert compat.supports_nn is True
        assert compat.supports_nnue is True
        assert compat.requires_policy_head is False


class TestHarnessRegistryFunctions:
    """Tests for harness registry factory functions."""

    def test_get_harness_compatibility(self):
        """Test get_harness_compatibility function."""
        from app.ai.harness.base_harness import HarnessType
        from app.ai.harness.harness_registry import get_harness_compatibility

        compat = get_harness_compatibility(HarnessType.GUMBEL_MCTS)
        assert compat is not None
        assert compat.harness_type == HarnessType.GUMBEL_MCTS

    def test_get_all_harness_types(self):
        """Test get_all_harness_types returns all types."""
        from app.ai.harness.base_harness import HarnessType
        from app.ai.harness.harness_registry import get_all_harness_types

        types = get_all_harness_types()
        assert len(types) >= 5  # At least 5 harness types
        assert HarnessType.GUMBEL_MCTS in types
        assert HarnessType.MINIMAX in types

    def test_get_compatible_harnesses_nn(self):
        """Test getting harnesses compatible with NN."""
        from app.ai.harness.base_harness import ModelType
        from app.ai.harness.harness_registry import get_compatible_harnesses

        harnesses = get_compatible_harnesses(ModelType.NEURAL_NET)
        # All harnesses should support NN
        assert len(harnesses) >= 5

    def test_get_compatible_harnesses_nnue(self):
        """Test getting harnesses compatible with NNUE."""
        from app.ai.harness.base_harness import HarnessType, ModelType
        from app.ai.harness.harness_registry import get_compatible_harnesses

        harnesses = get_compatible_harnesses(ModelType.NNUE)
        # NNUE compatible: MINIMAX, MAXN, BRS, others
        assert len(harnesses) >= 3
        assert HarnessType.MINIMAX in harnesses
        # Gumbel MCTS should NOT be in NNUE compatible
        assert HarnessType.GUMBEL_MCTS not in harnesses


class TestAIHarnessBase:
    """Tests for AIHarness abstract base class."""

    def test_cannot_instantiate_abstract_base(self):
        """Test AIHarness cannot be instantiated directly."""
        from app.ai.harness.base_harness import AIHarness, HarnessConfig

        with pytest.raises(TypeError):
            AIHarness(HarnessConfig())

    def test_concrete_harness_creation(self):
        """Test creating a concrete harness subclass."""
        from app.ai.harness.base_harness import AIHarness, HarnessConfig

        class MockHarness(AIHarness):
            def _create_underlying_ai(self, player_number):
                return MagicMock()

            def _select_move_impl(self, game_state, player_number):
                return None, {"value_estimate": 0.5, "nodes_visited": 100}

        config = HarnessConfig()
        harness = MockHarness(config)
        assert harness.config == config
        assert harness._underlying_ai is None

    def test_harness_evaluate_returns_metadata(self):
        """Test evaluate method returns EvaluationMetadata."""
        from app.ai.harness.base_harness import AIHarness, HarnessConfig
        from app.ai.harness.evaluation_metadata import EvaluationMetadata

        class MockHarness(AIHarness):
            def _create_underlying_ai(self, player_number):
                return MagicMock()

            def _select_move_impl(self, game_state, player_number):
                return "move_1", {
                    "value_estimate": 0.75,
                    "nodes_visited": 500,
                    "search_depth": 8,
                }

        config = HarnessConfig(model_id="test_model")
        harness = MockHarness(config)

        mock_state = MagicMock()
        move, metadata = harness.evaluate(mock_state, player_number=1)

        assert move == "move_1"
        assert isinstance(metadata, EvaluationMetadata)
        assert metadata.value_estimate == 0.75
        assert metadata.nodes_visited == 500
        assert metadata.search_depth == 8
        assert metadata.model_id == "test_model"

    def test_harness_reset(self):
        """Test harness reset clears state."""
        from app.ai.harness.base_harness import AIHarness, HarnessConfig

        class MockHarness(AIHarness):
            def _create_underlying_ai(self, player_number):
                ai = MagicMock()
                ai.reset = MagicMock()
                return ai

            def _select_move_impl(self, game_state, player_number):
                return None, {}

        config = HarnessConfig()
        harness = MockHarness(config)
        harness._last_visit_distribution = {"a": 0.5}
        harness._underlying_ai = MagicMock()
        harness._underlying_ai.reset = MagicMock()

        harness.reset()

        assert harness._last_visit_distribution is None
        harness._underlying_ai.reset.assert_called_once()

    def test_composite_participant_id(self):
        """Test composite participant ID generation."""
        from app.ai.harness.base_harness import AIHarness, HarnessConfig, HarnessType

        class MockHarness(AIHarness):
            def _create_underlying_ai(self, player_number):
                return MagicMock()

            def _select_move_impl(self, game_state, player_number):
                return None, {}

        config = HarnessConfig(
            harness_type=HarnessType.GUMBEL_MCTS,
            model_id="canonical_hex8_2p",
            difficulty=5,
            simulations=200,
        )
        harness = MockHarness(config)

        participant_id = harness.get_composite_participant_id()
        # Format: {model_id}:{harness_type}:{config_hash}
        parts = participant_id.split(":")
        assert len(parts) == 3
        assert parts[0] == "canonical_hex8_2p"
        assert parts[1] == "gumbel_mcts"
        assert len(parts[2]) == 8  # Hash is 8 chars

    def test_harness_repr(self):
        """Test harness string representation."""
        from app.ai.harness.base_harness import (
            AIHarness,
            HarnessConfig,
            HarnessType,
            ModelType,
        )

        class MockHarness(AIHarness):
            def _create_underlying_ai(self, player_number):
                return MagicMock()

            def _select_move_impl(self, game_state, player_number):
                return None, {}

        config = HarnessConfig(
            harness_type=HarnessType.MINIMAX,
            model_type=ModelType.NNUE,
            model_id="test_nnue",
        )
        harness = MockHarness(config)

        repr_str = repr(harness)
        assert "MockHarness" in repr_str
        assert "minimax" in repr_str
        assert "nnue" in repr_str
        assert "test_nnue" in repr_str


class TestHarnessModuleExports:
    """Tests for harness package exports."""

    def test_package_exports_types(self):
        """Test package exports core types."""
        from app.ai.harness import AIHarness, HarnessType, ModelType

        assert AIHarness is not None
        assert HarnessType is not None
        assert ModelType is not None

    def test_package_exports_metadata(self):
        """Test package exports EvaluationMetadata."""
        from app.ai.harness import EvaluationMetadata

        assert EvaluationMetadata is not None

    def test_package_exports_registry_functions(self):
        """Test package exports registry functions."""
        from app.ai.harness import (
            HarnessCompatibility,
            create_harness,
            get_all_harness_types,
            get_compatible_harnesses,
            get_harness_compatibility,
        )

        assert HarnessCompatibility is not None
        assert create_harness is not None
        assert get_harness_compatibility is not None
        assert get_compatible_harnesses is not None
        assert get_all_harness_types is not None
