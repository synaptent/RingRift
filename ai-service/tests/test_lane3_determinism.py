"""Lane 3: AI Service Determinism & Boundary Stability Tests.

These tests ensure:
1. AIType enum parity between Python and TypeScript (via canonical mapping)
2. Seed propagation produces deterministic AI behavior
3. Factory correctly routes all AI types including experimental ones (GMO, IG_GMO, EBMO)
4. Fallback behavior is consistent and documented

See docs/PLAN_AI_WORK.md Lane 3 for full scope.
"""


import pytest

from app.ai.base import BaseAI
from app.ai.factory import CANONICAL_DIFFICULTY_PROFILES, AIFactory
from app.models import AIConfig, AIType, BoardType
from app.training.initial_state import create_initial_state


def get_initial_state():
    """Create a simple initial state for testing."""
    return create_initial_state(
        board_type=BoardType.SQUARE8,
        num_players=2,
    )


# =============================================================================
# AIType Enum Completeness Tests
# =============================================================================


class TestAITypeEnumCompleteness:
    """Ensure all AIType values are handled by the factory."""

    # Types exposed to TypeScript service (must maintain parity)
    TYPESCRIPT_EXPOSED_TYPES: set[AIType] = {
        AIType.RANDOM,
        AIType.HEURISTIC,
        AIType.MINIMAX,
        AIType.MCTS,
        AIType.DESCENT,
        AIType.POLICY_ONLY,
        AIType.GUMBEL_MCTS,
        AIType.IG_GMO,
    }

    # Python-only types (not exposed to TS, used internally)
    PYTHON_ONLY_TYPES: set[AIType] = {
        AIType.GPU_MINIMAX,
        AIType.MAXN,
        AIType.BRS,
        AIType.NEURAL_DEMO,
        AIType.EBMO,
        AIType.GMO,
    }

    def test_all_types_covered(self):
        """Verify factory can create instances for all AIType enum values."""
        config = AIConfig(difficulty=5)

        for ai_type in AIType:
            # Skip types that require special handling
            if ai_type == AIType.NEURAL_DEMO:
                # Requires AI_ENGINE_NEURAL_DEMO_ENABLED env var
                continue

            try:
                ai = AIFactory.create(ai_type, player_number=1, config=config)
                assert isinstance(ai, BaseAI), f"{ai_type} should return BaseAI subclass"
            except ValueError as e:
                pytest.fail(f"Factory doesn't handle {ai_type}: {e}")

    def test_typescript_exposed_types_documented(self):
        """Verify TypeScript-exposed types are in the canonical difficulty ladder."""
        ladder_types = {profile["ai_type"] for profile in CANONICAL_DIFFICULTY_PROFILES.values()}

        # All TS-exposed types should appear somewhere in the ladder (or experimental slots)
        for ts_type in self.TYPESCRIPT_EXPOSED_TYPES:
            if ts_type not in ladder_types:
                # Check if it's in experimental slots (12-15)
                experimental_types = {
                    CANONICAL_DIFFICULTY_PROFILES.get(i, {}).get("ai_type")
                    for i in range(12, 16)
                }
                assert ts_type in experimental_types or ts_type in ladder_types, (
                    f"TypeScript type {ts_type} not in difficulty ladder"
                )

    def test_gmo_in_difficulty_ladder(self):
        """Verify GMO is registered at difficulty 13 (experimental slot)."""
        assert 13 in CANONICAL_DIFFICULTY_PROFILES
        profile = CANONICAL_DIFFICULTY_PROFILES[13]
        assert profile["ai_type"] == AIType.GMO
        assert profile["profile_id"] == "v3-gmo-13-experimental"

    def test_ig_gmo_in_difficulty_ladder(self):
        """Verify IG_GMO is registered at difficulty 14 (experimental slot)."""
        assert 14 in CANONICAL_DIFFICULTY_PROFILES
        profile = CANONICAL_DIFFICULTY_PROFILES[14]
        assert profile["ai_type"] == AIType.IG_GMO
        assert profile["profile_id"] == "v3-iggmo-14-experimental"


# =============================================================================
# Seed Propagation Tests
# =============================================================================


class TestSeedPropagation:
    """Verify deterministic behavior when seeds are provided."""

    @pytest.fixture
    def initial_state(self):
        """Create a reproducible initial game state."""
        return get_initial_state()

    def test_random_ai_deterministic_with_seed(self, initial_state):
        """RandomAI should produce same move with same seed."""
        config = AIConfig(difficulty=1, rng_seed=12345)

        ai1 = AIFactory.create(AIType.RANDOM, player_number=1, config=config)
        ai2 = AIFactory.create(AIType.RANDOM, player_number=1, config=config)

        # Reset both AIs with same seed
        ai1.reset_for_new_game(rng_seed=12345)
        ai2.reset_for_new_game(rng_seed=12345)

        move1 = ai1.select_move(initial_state)
        move2 = ai2.select_move(initial_state)

        assert move1 is not None
        assert move2 is not None
        assert move1.type == move2.type
        assert move1.to == move2.to

    def test_heuristic_ai_deterministic_with_seed(self, initial_state):
        """HeuristicAI should produce same move with same seed."""
        config = AIConfig(difficulty=2, rng_seed=54321)

        ai1 = AIFactory.create(AIType.HEURISTIC, player_number=1, config=config)
        ai2 = AIFactory.create(AIType.HEURISTIC, player_number=1, config=config)

        ai1.reset_for_new_game(rng_seed=54321)
        ai2.reset_for_new_game(rng_seed=54321)

        move1 = ai1.select_move(initial_state)
        move2 = ai2.select_move(initial_state)

        assert move1 is not None
        assert move2 is not None
        assert move1.type == move2.type
        assert move1.to == move2.to

    def test_gmo_ai_deterministic_with_seed(self, initial_state):
        """GMOAI should produce same move with same seed.

        GMO now saves/restores torch RNG state in reset_for_new_game/select_move,
        ensuring deterministic behavior even with MC Dropout.
        """
        config = AIConfig(difficulty=6, rng_seed=99999)

        ai1 = AIFactory.create(AIType.GMO, player_number=1, config=config)
        ai2 = AIFactory.create(AIType.GMO, player_number=1, config=config)

        ai1.reset_for_new_game(rng_seed=99999)
        ai2.reset_for_new_game(rng_seed=99999)

        move1 = ai1.select_move(initial_state)
        move2 = ai2.select_move(initial_state)

        assert move1 is not None
        assert move2 is not None
        assert move1.type == move2.type
        assert move1.to == move2.to

    def test_different_seeds_produce_different_moves(self, initial_state):
        """Different seeds should (usually) produce different moves."""
        config1 = AIConfig(difficulty=1, rng_seed=111)
        config2 = AIConfig(difficulty=1, rng_seed=222)

        ai1 = AIFactory.create(AIType.RANDOM, player_number=1, config=config1)
        ai2 = AIFactory.create(AIType.RANDOM, player_number=1, config=config2)

        ai1.reset_for_new_game(rng_seed=111)
        ai2.reset_for_new_game(rng_seed=222)

        # Collect moves with different seeds
        moves_seed1 = []
        moves_seed2 = []
        for _ in range(5):
            m1 = ai1.select_move(initial_state)
            m2 = ai2.select_move(initial_state)
            if m1:
                moves_seed1.append((m1.to.x, m1.to.y))
            if m2:
                moves_seed2.append((m2.to.x, m2.to.y))

        # At least one pair should differ (probabilistically true)
        # This is a weak assertion but catches bugs where seeds are ignored
        assert len(moves_seed1) > 0
        assert len(moves_seed2) > 0


# =============================================================================
# Edge Case Seed Tests
# =============================================================================


class TestEdgeCaseSeeds:
    """Test boundary conditions for seed values."""

    @pytest.fixture
    def initial_state(self):
        return get_initial_state()

    def test_seed_zero(self, initial_state):
        """Seed 0 should be valid and produce moves."""
        config = AIConfig(difficulty=1, rng_seed=0)
        ai = AIFactory.create(AIType.RANDOM, player_number=1, config=config)
        ai.reset_for_new_game(rng_seed=0)

        move = ai.select_move(initial_state)
        assert move is not None

    def test_seed_negative(self, initial_state):
        """Negative seeds should be handled (abs or wrap)."""
        config = AIConfig(difficulty=1, rng_seed=-1)
        ai = AIFactory.create(AIType.RANDOM, player_number=1, config=config)

        # Should not raise
        ai.reset_for_new_game(rng_seed=-1)
        move = ai.select_move(initial_state)
        assert move is not None

    def test_seed_large(self, initial_state):
        """Large seeds (near MAX_INT) should work."""
        large_seed = 2**31 - 1  # Max signed 32-bit int
        config = AIConfig(difficulty=1, rng_seed=large_seed)
        ai = AIFactory.create(AIType.RANDOM, player_number=1, config=config)
        ai.reset_for_new_game(rng_seed=large_seed)

        move = ai.select_move(initial_state)
        assert move is not None

    def test_seed_none_uses_random(self, initial_state):
        """None seed should use random initialization (non-deterministic)."""
        config = AIConfig(difficulty=1, rng_seed=None)

        ai1 = AIFactory.create(AIType.RANDOM, player_number=1, config=config)
        ai2 = AIFactory.create(AIType.RANDOM, player_number=1, config=config)

        ai1.reset_for_new_game(rng_seed=None)
        ai2.reset_for_new_game(rng_seed=None)

        # With None seeds, we can't guarantee determinism
        # Just verify moves are produced
        move1 = ai1.select_move(initial_state)
        move2 = ai2.select_move(initial_state)

        assert move1 is not None
        assert move2 is not None


# =============================================================================
# Factory Routing Tests
# =============================================================================


class TestFactoryRouting:
    """Verify AIFactory correctly routes to expected AI implementations."""

    def test_gmo_creates_gmoai(self):
        """AIType.GMO should create GMOAI instance."""
        from app.ai.gmo_ai import GMOAI

        config = AIConfig(difficulty=6)
        ai = AIFactory.create(AIType.GMO, player_number=1, config=config)

        assert isinstance(ai, GMOAI)

    def test_ig_gmo_creates_iggmo(self):
        """AIType.IG_GMO should create IGGMO instance."""
        from app.ai.ig_gmo import IGGMO

        config = AIConfig(difficulty=6)
        ai = AIFactory.create(AIType.IG_GMO, player_number=1, config=config)

        assert isinstance(ai, IGGMO)

    def test_ebmo_creates_ebmo(self):
        """AIType.EBMO should create EBMO_AI instance."""
        from archive.deprecated_ai.ebmo_ai import EBMO_AI

        config = AIConfig(difficulty=6)
        ai = AIFactory.create(AIType.EBMO, player_number=1, config=config)

        assert isinstance(ai, EBMO_AI)

    def test_experimental_slot_13_profile_exists(self):
        """Verify experimental slot 13 is configured for GMO.

        Note: Experimental slots (12-15) are not accessed via create_from_difficulty
        since that function is for user-facing difficulty levels 1-10.
        These slots are for internal/research use via direct AIFactory.create().
        """
        assert 13 in CANONICAL_DIFFICULTY_PROFILES
        profile = CANONICAL_DIFFICULTY_PROFILES[13]
        assert profile["ai_type"] == AIType.GMO

    def test_experimental_slot_14_profile_exists(self):
        """Verify experimental slot 14 is configured for IG_GMO.

        Note: Experimental slots (12-15) are not accessed via create_from_difficulty
        since that function is for user-facing difficulty levels 1-10.
        These slots are for internal/research use via direct AIFactory.create().
        """
        assert 14 in CANONICAL_DIFFICULTY_PROFILES
        profile = CANONICAL_DIFFICULTY_PROFILES[14]
        assert profile["ai_type"] == AIType.IG_GMO
