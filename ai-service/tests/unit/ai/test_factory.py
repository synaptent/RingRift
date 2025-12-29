"""
Unit tests for app.ai.factory module.

Tests cover:
- DifficultyProfile structure
- Canonical difficulty profiles (1-24)
- Multiplayer and large board overrides
- Helper functions (get_difficulty_profile, select_ai_type, etc.)
- AIFactory class methods
- Tournament AI creation
- MCTS mode creation
- Environment variable overrides

Created: December 2025
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from app.ai.factory import (
    # Type definitions
    DifficultyProfile,
    # Canonical profiles
    CANONICAL_DIFFICULTY_PROFILES,
    MULTIPLAYER_DIFFICULTY_OVERRIDES,
    LARGE_BOARD_DIFFICULTY_OVERRIDES,
    DIFFICULTY_DESCRIPTIONS,
    LARGE_BOARD_TYPES,
    EBMO_DIFFICULTY_OVERRIDE,
    IG_GMO_DIFFICULTY_OVERRIDE,
    HYBRID_NN_D7_OVERRIDE,
    # Helper functions
    get_difficulty_profile,
    select_ai_type,
    get_randomness_for_difficulty,
    get_think_time_for_difficulty,
    uses_neural_net,
    get_all_difficulties,
    get_difficulty_description,
    # Factory class
    AIFactory,
    # Module-level aliases
    create_ai,
    create_ai_from_difficulty,
    create_tournament_ai,
    create_mcts,
)
from app.models.core import AIConfig, AIType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config():
    """Create a basic AI config."""
    return AIConfig(difficulty=5, think_time=1000)


@pytest.fixture(autouse=True)
def clear_factory_cache():
    """Clear the AIFactory cache before each test."""
    AIFactory.clear_cache()
    yield
    AIFactory.clear_cache()


@pytest.fixture(autouse=True)
def clean_custom_registry():
    """Clean up custom registry after each test."""
    # Store original state
    original = dict(AIFactory._custom_registry)
    yield
    # Restore original state
    AIFactory._custom_registry = original


# =============================================================================
# Canonical Difficulty Profiles Tests
# =============================================================================


class TestCanonicalDifficultyProfiles:
    """Tests for canonical difficulty profile structure."""

    def test_profiles_exist_for_levels_1_to_24(self):
        """All difficulty levels 1-24 have profiles defined."""
        for level in range(1, 25):
            assert level in CANONICAL_DIFFICULTY_PROFILES, f"Missing profile for level {level}"

    def test_profile_has_required_fields(self):
        """Each profile has all required TypedDict fields."""
        required_fields = {"ai_type", "randomness", "think_time_ms", "profile_id", "use_neural_net"}
        for level, profile in CANONICAL_DIFFICULTY_PROFILES.items():
            assert required_fields.issubset(profile.keys()), f"Profile {level} missing fields"

    def test_ai_type_is_valid(self):
        """Each profile has a valid AIType."""
        for level, profile in CANONICAL_DIFFICULTY_PROFILES.items():
            assert isinstance(profile["ai_type"], AIType), f"Level {level} has invalid ai_type"

    def test_randomness_in_valid_range(self):
        """Randomness values are between 0 and 1."""
        for level, profile in CANONICAL_DIFFICULTY_PROFILES.items():
            assert 0.0 <= profile["randomness"] <= 1.0, f"Level {level} has invalid randomness"

    def test_think_time_is_positive(self):
        """Think time values are positive."""
        for level, profile in CANONICAL_DIFFICULTY_PROFILES.items():
            assert profile["think_time_ms"] > 0, f"Level {level} has invalid think_time"

    def test_profile_id_is_string(self):
        """Profile IDs are non-empty strings."""
        for level, profile in CANONICAL_DIFFICULTY_PROFILES.items():
            assert isinstance(profile["profile_id"], str), f"Level {level} has invalid profile_id"
            assert len(profile["profile_id"]) > 0, f"Level {level} has empty profile_id"

    def test_use_neural_net_is_boolean(self):
        """use_neural_net values are booleans."""
        for level, profile in CANONICAL_DIFFICULTY_PROFILES.items():
            assert isinstance(profile["use_neural_net"], bool), f"Level {level} has invalid use_neural_net"

    def test_difficulty_increases_with_level(self):
        """Higher levels generally have lower randomness and higher think time."""
        # Compare beginner vs expert levels
        beginner = CANONICAL_DIFFICULTY_PROFILES[1]
        expert = CANONICAL_DIFFICULTY_PROFILES[8]
        assert beginner["randomness"] >= expert["randomness"], "Expert should have less randomness"
        assert beginner["think_time_ms"] <= expert["think_time_ms"], "Expert should have more think time"

    def test_neural_net_enabled_at_higher_levels(self):
        """Neural network is enabled for higher difficulty levels."""
        # Level 1-2 should not use NN, level 8+ should use NN
        assert not CANONICAL_DIFFICULTY_PROFILES[1]["use_neural_net"]
        assert not CANONICAL_DIFFICULTY_PROFILES[2]["use_neural_net"]
        assert CANONICAL_DIFFICULTY_PROFILES[8]["use_neural_net"]
        assert CANONICAL_DIFFICULTY_PROFILES[11]["use_neural_net"]


class TestMultiplayerOverrides:
    """Tests for multiplayer difficulty overrides."""

    def test_multiplayer_overrides_exist(self):
        """Multiplayer overrides exist for key difficulty levels."""
        # Overrides should cover D4-D10
        assert 4 in MULTIPLAYER_DIFFICULTY_OVERRIDES
        assert 5 in MULTIPLAYER_DIFFICULTY_OVERRIDES
        assert 10 in MULTIPLAYER_DIFFICULTY_OVERRIDES

    def test_multiplayer_uses_maxn_or_brs(self):
        """Multiplayer overrides use MaxN or BRS instead of Minimax."""
        for level, profile in MULTIPLAYER_DIFFICULTY_OVERRIDES.items():
            ai_type = profile["ai_type"]
            assert ai_type in {AIType.MAXN, AIType.BRS}, f"Level {level} should use MaxN or BRS"

    def test_multiplayer_profile_has_mp_suffix(self):
        """Multiplayer profile IDs have -mp suffix."""
        for level, profile in MULTIPLAYER_DIFFICULTY_OVERRIDES.items():
            assert "-mp" in profile["profile_id"], f"Level {level} profile should have -mp suffix"


class TestLargeBoardOverrides:
    """Tests for large board difficulty overrides."""

    def test_large_board_overrides_exist(self):
        """Large board overrides exist for key difficulty levels."""
        assert 4 in LARGE_BOARD_DIFFICULTY_OVERRIDES
        assert 5 in LARGE_BOARD_DIFFICULTY_OVERRIDES

    def test_large_board_uses_descent_or_mcts(self):
        """Large board overrides use Descent or MCTS instead of Minimax."""
        for level, profile in LARGE_BOARD_DIFFICULTY_OVERRIDES.items():
            ai_type = profile["ai_type"]
            assert ai_type in {AIType.DESCENT, AIType.MCTS}, f"Level {level} should use Descent or MCTS"

    def test_large_board_types_set(self):
        """LARGE_BOARD_TYPES contains expected board types."""
        assert "square19" in LARGE_BOARD_TYPES
        assert "hexagonal" in LARGE_BOARD_TYPES


class TestDifficultyDescriptions:
    """Tests for difficulty level descriptions."""

    def test_descriptions_exist_for_main_levels(self):
        """Descriptions exist for main difficulty levels 1-11."""
        for level in range(1, 12):
            assert level in DIFFICULTY_DESCRIPTIONS, f"Missing description for level {level}"

    def test_descriptions_are_strings(self):
        """All descriptions are non-empty strings."""
        for level, desc in DIFFICULTY_DESCRIPTIONS.items():
            assert isinstance(desc, str), f"Level {level} description should be string"
            assert len(desc) > 0, f"Level {level} has empty description"

    def test_experimental_descriptions_exist(self):
        """Descriptions exist for experimental AI levels (12+)."""
        for level in range(12, 25):
            assert level in DIFFICULTY_DESCRIPTIONS, f"Missing description for experimental level {level}"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetDifficultyProfile:
    """Tests for get_difficulty_profile function."""

    def test_returns_canonical_profile(self):
        """Returns canonical profile for standard 2-player games."""
        profile = get_difficulty_profile(5)
        assert profile == CANONICAL_DIFFICULTY_PROFILES[5]

    def test_clamps_low_difficulty(self):
        """Difficulty below 1 is clamped to 1."""
        profile = get_difficulty_profile(0)
        assert profile == CANONICAL_DIFFICULTY_PROFILES[1]

        profile = get_difficulty_profile(-5)
        assert profile == CANONICAL_DIFFICULTY_PROFILES[1]

    def test_clamps_high_difficulty(self):
        """Difficulty above max is clamped to max."""
        max_level = max(CANONICAL_DIFFICULTY_PROFILES.keys())
        profile = get_difficulty_profile(100)
        assert profile == CANONICAL_DIFFICULTY_PROFILES[max_level]

    def test_returns_multiplayer_override_for_3_players(self):
        """Returns multiplayer override for 3-player games."""
        profile = get_difficulty_profile(5, num_players=3)
        assert profile == MULTIPLAYER_DIFFICULTY_OVERRIDES[5]

    def test_returns_multiplayer_override_for_4_players(self):
        """Returns multiplayer override for 4-player games."""
        profile = get_difficulty_profile(5, num_players=4)
        assert profile == MULTIPLAYER_DIFFICULTY_OVERRIDES[5]

    def test_returns_large_board_override(self):
        """Returns large board override for large board types."""
        profile = get_difficulty_profile(5, num_players=2, board_type="square19")
        assert profile == LARGE_BOARD_DIFFICULTY_OVERRIDES[5]

    def test_returns_canonical_for_small_boards(self):
        """Returns canonical profile for small boards."""
        profile = get_difficulty_profile(5, num_players=2, board_type="square8")
        assert profile == CANONICAL_DIFFICULTY_PROFILES[5]

    def test_multiplayer_takes_precedence_over_large_board(self):
        """Multiplayer override takes precedence over large board override."""
        profile = get_difficulty_profile(5, num_players=3, board_type="square19")
        # Should use multiplayer override since num_players >= 3
        assert profile == MULTIPLAYER_DIFFICULTY_OVERRIDES[5]

    @patch.dict(os.environ, {"RINGRIFT_USE_IG_GMO": "1"})
    def test_ig_gmo_override_at_d3(self):
        """IG-GMO override is used at D3 when env var is set."""
        profile = get_difficulty_profile(3)
        assert profile == IG_GMO_DIFFICULTY_OVERRIDE

    @patch.dict(os.environ, {"RINGRIFT_USE_EBMO": "1"})
    def test_ebmo_override_at_d3(self):
        """EBMO override is used at D3 when env var is set."""
        profile = get_difficulty_profile(3)
        assert profile == EBMO_DIFFICULTY_OVERRIDE

    @patch.dict(os.environ, {"RINGRIFT_USE_HYBRID_D7": "1"})
    def test_hybrid_nn_override_at_d7(self):
        """HybridNN override is used at D7 when env var is set."""
        profile = get_difficulty_profile(7)
        assert profile == HYBRID_NN_D7_OVERRIDE


class TestSelectAIType:
    """Tests for select_ai_type function."""

    def test_returns_random_for_level_1(self):
        """Level 1 returns RANDOM AI type."""
        assert select_ai_type(1) == AIType.RANDOM

    def test_returns_heuristic_for_level_2(self):
        """Level 2 returns HEURISTIC AI type."""
        assert select_ai_type(2) == AIType.HEURISTIC

    def test_returns_gumbel_mcts_for_high_levels(self):
        """High levels return GUMBEL_MCTS AI type."""
        assert select_ai_type(8) == AIType.GUMBEL_MCTS
        assert select_ai_type(11) == AIType.GUMBEL_MCTS


class TestGetRandomnessForDifficulty:
    """Tests for get_randomness_for_difficulty function."""

    def test_returns_high_randomness_for_easy(self):
        """Easy levels have high randomness."""
        randomness = get_randomness_for_difficulty(1)
        assert randomness > 0.3

    def test_returns_zero_randomness_for_expert(self):
        """Expert levels have zero randomness."""
        randomness = get_randomness_for_difficulty(11)
        assert randomness == 0.0


class TestGetThinkTimeForDifficulty:
    """Tests for get_think_time_for_difficulty function."""

    def test_returns_short_time_for_easy(self):
        """Easy levels have short think time."""
        think_time = get_think_time_for_difficulty(1)
        assert think_time < 500

    def test_returns_long_time_for_ultimate(self):
        """Ultimate level has long think time."""
        think_time = get_think_time_for_difficulty(11)
        assert think_time >= 60000  # 60 seconds


class TestUsesNeuralNet:
    """Tests for uses_neural_net function."""

    def test_false_for_random(self):
        """Level 1 (random) does not use neural net."""
        assert not uses_neural_net(1)

    def test_true_for_policy_only(self):
        """Level 3 (policy-only) uses neural net."""
        assert uses_neural_net(3)

    def test_true_for_gumbel_mcts(self):
        """High levels using Gumbel MCTS use neural net."""
        assert uses_neural_net(8)
        assert uses_neural_net(11)


class TestGetAllDifficulties:
    """Tests for get_all_difficulties function."""

    def test_returns_copy_of_profiles(self):
        """Returns a copy, not the original."""
        profiles = get_all_difficulties()
        assert profiles == CANONICAL_DIFFICULTY_PROFILES
        # Modifying returned dict should not affect original
        profiles[1] = {"modified": True}
        assert CANONICAL_DIFFICULTY_PROFILES[1] != {"modified": True}


class TestGetDifficultyDescription:
    """Tests for get_difficulty_description function."""

    def test_returns_description_for_valid_level(self):
        """Returns description for valid levels."""
        desc = get_difficulty_description(5)
        assert "Mid" in desc or "mid" in desc.lower()

    def test_clamps_low_level(self):
        """Low levels are clamped to 1."""
        desc = get_difficulty_description(-1)
        assert desc == DIFFICULTY_DESCRIPTIONS[1]

    def test_clamps_high_level(self):
        """High levels are clamped to 11."""
        desc = get_difficulty_description(100)
        assert desc == DIFFICULTY_DESCRIPTIONS.get(11, f"Difficulty 100")


# =============================================================================
# AIFactory Class Tests
# =============================================================================


class TestAIFactoryRegistration:
    """Tests for AIFactory custom AI registration."""

    def test_register_custom_ai(self):
        """Can register a custom AI constructor."""
        mock_constructor = MagicMock()
        AIFactory.register("test_ai", mock_constructor)
        assert "test_ai" in AIFactory._custom_registry
        assert AIFactory._custom_registry["test_ai"] == mock_constructor

    def test_unregister_custom_ai(self):
        """Can unregister a custom AI."""
        mock_constructor = MagicMock()
        AIFactory.register("test_ai", mock_constructor)
        result = AIFactory.unregister("test_ai")
        assert result is True
        assert "test_ai" not in AIFactory._custom_registry

    def test_unregister_nonexistent_returns_false(self):
        """Unregistering nonexistent AI returns False."""
        result = AIFactory.unregister("nonexistent_ai")
        assert result is False

    def test_list_registered_includes_builtin(self):
        """list_registered includes built-in AI types."""
        registered = AIFactory.list_registered()
        assert AIType.RANDOM.value in registered
        assert AIType.HEURISTIC.value in registered
        assert AIType.MCTS.value in registered

    def test_list_registered_includes_custom(self):
        """list_registered includes custom AI types."""
        mock_constructor = MagicMock()
        mock_constructor.__doc__ = "Test AI for testing"
        AIFactory.register("test_custom", mock_constructor)
        registered = AIFactory.list_registered()
        assert "test_custom" in registered
        assert "Custom:" in registered["test_custom"]


class TestAIFactoryCreate:
    """Tests for AIFactory.create method."""

    def test_create_random_ai(self):
        """Can create RandomAI."""
        config = AIConfig(difficulty=1, think_time=100)
        ai = AIFactory.create(AIType.RANDOM, player_number=1, config=config)
        assert ai is not None
        assert ai.player_number == 1

    def test_create_heuristic_ai(self):
        """Can create HeuristicAI."""
        config = AIConfig(difficulty=2, think_time=200)
        ai = AIFactory.create(AIType.HEURISTIC, player_number=1, config=config)
        assert ai is not None
        assert ai.player_number == 1

    def test_create_caches_class(self):
        """AI class is cached after first creation."""
        config = AIConfig(difficulty=1, think_time=100)
        AIFactory.create(AIType.RANDOM, player_number=1, config=config)
        assert AIType.RANDOM in AIFactory._class_cache

    def test_neural_demo_fallback_without_env(self):
        """NEURAL_DEMO falls back to HEURISTIC without env var."""
        config = AIConfig(difficulty=5, think_time=1000)
        with patch.dict(os.environ, {}, clear=True):
            ai = AIFactory.create(AIType.NEURAL_DEMO, player_number=1, config=config)
            # Should fall back to HeuristicAI
            from app.ai.heuristic_ai import HeuristicAI
            assert isinstance(ai, HeuristicAI)

    def test_unsupported_ai_type_raises(self):
        """Unsupported AI type raises ValueError."""
        config = AIConfig(difficulty=5, think_time=1000)
        # Create a mock AI type that doesn't exist
        with pytest.raises(ValueError, match="Unsupported AI type"):
            # We need to bypass the enum, so we'll test with the internal method
            AIFactory._get_ai_class(MagicMock(value="nonexistent"))


class TestAIFactoryCreateFromDifficulty:
    """Tests for AIFactory.create_from_difficulty method."""

    def test_create_from_difficulty_level_1(self):
        """Creates appropriate AI for difficulty 1."""
        ai = AIFactory.create_from_difficulty(1, player_number=1)
        assert ai is not None
        # RandomAI for level 1
        from app.ai.random_ai import RandomAI
        assert isinstance(ai, RandomAI)

    def test_create_from_difficulty_level_2(self):
        """Creates appropriate AI for difficulty 2."""
        ai = AIFactory.create_from_difficulty(2, player_number=1)
        assert ai is not None
        from app.ai.heuristic_ai import HeuristicAI
        assert isinstance(ai, HeuristicAI)

    def test_create_with_think_time_override(self):
        """Respects think_time_override."""
        ai = AIFactory.create_from_difficulty(5, player_number=1, think_time_override=9999)
        assert ai.config.think_time == 9999

    def test_create_with_randomness_override(self):
        """Respects randomness_override."""
        ai = AIFactory.create_from_difficulty(5, player_number=1, randomness_override=0.75)
        assert ai.config.randomness == 0.75

    def test_create_with_rng_seed(self):
        """Respects rng_seed."""
        ai = AIFactory.create_from_difficulty(2, player_number=1, rng_seed=12345)
        assert ai.config.rng_seed == 12345

    def test_create_uses_multiplayer_override(self):
        """Uses multiplayer profile for 3+ players."""
        ai = AIFactory.create_from_difficulty(5, player_number=1, num_players=4)
        # Should use MaxN or BRS instead of Minimax
        from app.ai.maxn_ai import MaxNAI
        assert isinstance(ai, MaxNAI)


class TestAIFactoryCreateCustom:
    """Tests for AIFactory.create_custom method."""

    def test_create_custom_ai(self):
        """Can create a custom registered AI."""
        mock_ai = MagicMock()
        mock_constructor = MagicMock(return_value=mock_ai)
        AIFactory.register("test_custom", mock_constructor)

        config = AIConfig(difficulty=5, think_time=1000)
        result = AIFactory.create_custom("test_custom", player_number=1, config=config)

        assert result == mock_ai
        mock_constructor.assert_called_once_with(1, config)

    def test_create_unknown_custom_raises(self):
        """Creating unknown custom AI raises ValueError."""
        config = AIConfig(difficulty=5, think_time=1000)
        with pytest.raises(ValueError, match="Unknown custom AI identifier"):
            AIFactory.create_custom("unknown_custom", player_number=1, config=config)


class TestAIFactoryCreateForTournament:
    """Tests for AIFactory.create_for_tournament method."""

    def test_create_random_agent(self):
        """Creates random agent for tournament."""
        ai = AIFactory.create_for_tournament("random", player_number=1)
        from app.ai.random_ai import RandomAI
        assert isinstance(ai, RandomAI)

    def test_create_heuristic_agent(self):
        """Creates heuristic agent for tournament."""
        ai = AIFactory.create_for_tournament("heuristic", player_number=1)
        from app.ai.heuristic_ai import HeuristicAI
        assert isinstance(ai, HeuristicAI)

    def test_create_difficulty_agent(self):
        """Creates agent from difficulty_N format."""
        ai = AIFactory.create_for_tournament("difficulty_1", player_number=1)
        from app.ai.random_ai import RandomAI
        assert isinstance(ai, RandomAI)

    def test_create_level_agent(self):
        """Creates agent from level_N format."""
        ai = AIFactory.create_for_tournament("level_2", player_number=1)
        from app.ai.heuristic_ai import HeuristicAI
        assert isinstance(ai, HeuristicAI)

    def test_unknown_agent_defaults_to_heuristic(self):
        """Unknown agent ID defaults to heuristic."""
        ai = AIFactory.create_for_tournament("unknown_agent_xyz", player_number=1)
        from app.ai.heuristic_ai import HeuristicAI
        assert isinstance(ai, HeuristicAI)

    def test_gumbel_mcts_agent(self):
        """Creates Gumbel MCTS agent."""
        ai = AIFactory.create_for_tournament(
            "gumbel_mcts",
            player_number=1,
            board_type="square8",
            num_players=2,
        )
        from app.ai.gumbel_mcts_ai import GumbelMCTSAI
        assert isinstance(ai, GumbelMCTSAI)

    def test_gumbel_with_budget(self):
        """Creates Gumbel MCTS agent with custom budget."""
        ai = AIFactory.create_for_tournament(
            "gumbel_200",
            player_number=1,
            board_type="square8",
            num_players=2,
        )
        from app.ai.gumbel_mcts_ai import GumbelMCTSAI
        assert isinstance(ai, GumbelMCTSAI)
        assert ai.config.gumbel_simulation_budget == 200

    def test_policy_only_agent(self):
        """Creates policy-only agent."""
        ai = AIFactory.create_for_tournament(
            "policy_only",
            player_number=1,
            board_type="square8",
            num_players=2,
        )
        from app.ai.policy_only_ai import PolicyOnlyAI
        assert isinstance(ai, PolicyOnlyAI)

    def test_mcts_agent_with_sims(self):
        """Creates MCTS agent with simulation count."""
        ai = AIFactory.create_for_tournament("mcts_500", player_number=1)
        from app.ai.mcts_ai import MCTSAI
        assert isinstance(ai, MCTSAI)

    def test_custom_registered_agent(self):
        """Creates custom registered agent in tournament."""
        mock_ai = MagicMock()
        mock_constructor = MagicMock(return_value=mock_ai)
        AIFactory.register("custom_tourney", mock_constructor)

        result = AIFactory.create_for_tournament("custom_tourney", player_number=1)
        assert result == mock_ai


class TestAIFactoryClearCache:
    """Tests for AIFactory.clear_cache method."""

    def test_clear_cache_removes_cached_classes(self):
        """clear_cache removes all cached AI classes."""
        config = AIConfig(difficulty=1, think_time=100)
        AIFactory.create(AIType.RANDOM, player_number=1, config=config)
        assert len(AIFactory._class_cache) > 0

        AIFactory.clear_cache()
        assert len(AIFactory._class_cache) == 0


class TestAIFactoryCreateMCTS:
    """Tests for AIFactory.create_mcts method."""

    def test_create_standard_mode(self):
        """Creates standard mode MCTS."""
        mcts = AIFactory.create_mcts("square8", mode="standard")
        from app.ai.gumbel_mcts_ai import GumbelMCTSAI
        assert isinstance(mcts, GumbelMCTSAI)

    def test_create_with_custom_budget(self):
        """Creates MCTS with custom simulation budget."""
        mcts = AIFactory.create_mcts("square8", mode="standard", simulation_budget=300)
        assert mcts.config.gumbel_simulation_budget == 300

    def test_create_with_custom_sampled_actions(self):
        """Creates MCTS with custom num_sampled_actions."""
        mcts = AIFactory.create_mcts("square8", mode="standard", num_sampled_actions=8)
        assert mcts.config.gumbel_num_sampled_actions == 8

    def test_unknown_mode_raises(self):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown MCTS mode"):
            AIFactory.create_mcts("square8", mode="unknown_mode")


# =============================================================================
# Module-Level Alias Tests
# =============================================================================


class TestModuleLevelAliases:
    """Tests for module-level function aliases."""

    def test_create_ai_is_factory_create(self):
        """create_ai is an alias for AIFactory.create."""
        assert create_ai == AIFactory.create

    def test_create_ai_from_difficulty_is_factory_method(self):
        """create_ai_from_difficulty is an alias for AIFactory.create_from_difficulty."""
        assert create_ai_from_difficulty == AIFactory.create_from_difficulty

    def test_create_tournament_ai_is_factory_method(self):
        """create_tournament_ai is an alias for AIFactory.create_for_tournament."""
        assert create_tournament_ai == AIFactory.create_for_tournament

    def test_create_mcts_is_factory_method(self):
        """create_mcts is an alias for AIFactory.create_mcts."""
        assert create_mcts == AIFactory.create_mcts


# =============================================================================
# Experimental Override Tests
# =============================================================================


class TestExperimentalOverrides:
    """Tests for experimental AI overrides."""

    def test_ebmo_override_structure(self):
        """EBMO override has correct structure."""
        assert EBMO_DIFFICULTY_OVERRIDE["ai_type"] == AIType.EBMO
        assert EBMO_DIFFICULTY_OVERRIDE["use_neural_net"] is True
        assert "profile_id" in EBMO_DIFFICULTY_OVERRIDE

    def test_ig_gmo_override_structure(self):
        """IG-GMO override has correct structure."""
        assert IG_GMO_DIFFICULTY_OVERRIDE["ai_type"] == AIType.IG_GMO
        assert IG_GMO_DIFFICULTY_OVERRIDE["use_neural_net"] is True
        assert "profile_id" in IG_GMO_DIFFICULTY_OVERRIDE

    def test_hybrid_nn_override_structure(self):
        """HybridNN override has correct structure."""
        assert HYBRID_NN_D7_OVERRIDE["ai_type"] == AIType.HYBRID_NN
        assert HYBRID_NN_D7_OVERRIDE["use_neural_net"] is True
        assert HYBRID_NN_D7_OVERRIDE["think_time_ms"] == 500  # Fast response


# =============================================================================
# Deprecation Warning Tests
# =============================================================================


class TestDeprecationWarnings:
    """Tests for deprecated AI type warnings."""

    def test_ebmo_emits_deprecation_warning(self):
        """EBMO creation emits deprecation warning."""
        config = AIConfig(difficulty=5, think_time=1000)
        with pytest.warns(DeprecationWarning, match="EBMO is deprecated"):
            AIFactory.create(AIType.EBMO, player_number=1, config=config)

    def test_gmo_emits_deprecation_warning(self):
        """GMO creation emits deprecation warning."""
        config = AIConfig(difficulty=5, think_time=1000)
        with pytest.warns(DeprecationWarning, match="GMO is deprecated"):
            AIFactory.create(AIType.GMO, player_number=1, config=config)

    def test_gmo_v2_emits_deprecation_warning(self):
        """GMO v2 creation emits deprecation warning."""
        config = AIConfig(difficulty=5, think_time=1000)
        with pytest.warns(DeprecationWarning, match="GMO v2 is deprecated"):
            AIFactory.create(AIType.GMO_V2, player_number=1, config=config)

    def test_ig_gmo_emits_deprecation_warning(self):
        """IG-GMO creation emits deprecation warning."""
        config = AIConfig(difficulty=5, think_time=1000)
        with pytest.warns(DeprecationWarning, match="IG-GMO is deprecated"):
            AIFactory.create(AIType.IG_GMO, player_number=1, config=config)
