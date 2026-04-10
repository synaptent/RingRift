import pytest

from app.ai.descent_ai import DescentAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.ig_gmo import IGGMO
from app.ai.mcts_ai import MCTSAI
from app.ai.minimax_ai import MinimaxAI
from app.ai.random_ai import RandomAI
from app.main import (
    _create_ai_instance,
    _get_difficulty_profile,
    _select_ai_type,
)
from app.models import AIConfig, AIType

# Test timeout guards to prevent hanging in CI
TEST_TIMEOUT_SECONDS = 30


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_select_ai_type():
    # Canonical mapping (updated 2025):
    #   1→Random, 2→Heuristic, 3–4→Minimax, 5–6→Descent,
    #   7–8→MCTS, 9–10→Gumbel MCTS
    assert _select_ai_type(1) == AIType.RANDOM
    assert _select_ai_type(2) == AIType.HEURISTIC
    assert _select_ai_type(3) == AIType.MINIMAX
    assert _select_ai_type(4) == AIType.MINIMAX
    assert _select_ai_type(5) == AIType.DESCENT
    assert _select_ai_type(6) == AIType.DESCENT
    assert _select_ai_type(7) == AIType.MCTS
    assert _select_ai_type(8) == AIType.MCTS
    assert _select_ai_type(9) == AIType.GUMBEL_MCTS
    assert _select_ai_type(10) == AIType.GUMBEL_MCTS


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_difficulty_profile_mapping():
    """Canonical ladder profiles for difficulties 1–10 (updated 2025)."""
    profiles = {
        1: (AIType.RANDOM, 0.5, 150, "v1-random-1"),
        2: (AIType.HEURISTIC, 0.3, 200, "v1-heuristic-2"),
        3: (AIType.MINIMAX, 0.15, 1800, "v1-minimax-3"),
        4: (AIType.MINIMAX, 0.08, 2800, "v1-minimax-4-nnue"),
        5: (AIType.DESCENT, 0.05, 4000, "ringrift_best_sq8_2p"),
        6: (AIType.DESCENT, 0.02, 5500, "ringrift_best_sq8_2p"),
        7: (AIType.MCTS, 0.0, 7500, "v1-mcts-7"),
        8: (AIType.MCTS, 0.0, 9600, "ringrift_best_sq8_2p"),
        9: (AIType.GUMBEL_MCTS, 0.0, 12600, "ringrift_best_sq8_2p"),
        10: (AIType.GUMBEL_MCTS, 0.0, 16000, "ringrift_best_sq8_2p"),
    }

    for difficulty, (
        expected_type,
        expected_randomness,
        expected_think_ms,
        expected_profile_id,
    ) in profiles.items():
        profile = _get_difficulty_profile(difficulty)
        assert profile["ai_type"] == expected_type
        assert profile["randomness"] == pytest.approx(expected_randomness)
        assert profile["think_time_ms"] == expected_think_ms
        assert profile["profile_id"] == expected_profile_id


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_difficulty_profile_clamping():
    """Out-of-range difficulties are clamped into [1, 10] consistently."""
    low = _get_difficulty_profile(0)
    high = _get_difficulty_profile(11)

    assert low == _get_difficulty_profile(1)
    assert high == _get_difficulty_profile(10)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_create_ai_instance(monkeypatch):
    """Factory should construct the expected concrete AI types.

    For heavy engines such as DescentAI we monkeypatch the underlying
    HexNeuralNet to avoid expensive model initialisation in CI.
    """
    config = AIConfig(difficulty=5, randomness=0.1, rngSeed=None)

    # Test Random
    ai = _create_ai_instance(AIType.RANDOM, 1, config)
    assert isinstance(ai, RandomAI)

    # Test Heuristic
    ai = _create_ai_instance(AIType.HEURISTIC, 1, config)
    assert isinstance(ai, HeuristicAI)

    # Test Minimax
    ai = _create_ai_instance(AIType.MINIMAX, 1, config)
    assert isinstance(ai, MinimaxAI)

    # Test MCTS
    ai = _create_ai_instance(AIType.MCTS, 1, config)
    assert isinstance(ai, MCTSAI)

    # Test IG-GMO (experimental)
    ai = _create_ai_instance(AIType.IG_GMO, 1, config)
    assert isinstance(ai, IGGMO)

    # Patch NeuralNetAI used by DescentAI to a lightweight stub so that
    # constructing a DescentAI instance does not allocate large tensors.
    # NeuralNetAI is lazily imported inside DescentAI.__init__, so we patch
    # the source module (app.ai.neural_net) rather than descent_ai.
    import app.ai.neural_net as neural_net_mod

    class DummyNeuralNetAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def to(self, *_args, **_kwargs):
            return self

    monkeypatch.setattr(
        neural_net_mod,
        "NeuralNetAI",
        DummyNeuralNetAI,
        raising=True,
    )

    # Test Descent
    ai = _create_ai_instance(AIType.DESCENT, 1, config)
    assert isinstance(ai, DescentAI)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_create_ai_instance_neural_demo_gated(monkeypatch):
    """NEURAL_DEMO should be gated by AI_ENGINE_NEURAL_DEMO_ENABLED."""
    config = AIConfig(difficulty=5, randomness=0.0, rngSeed=None)

    # When the flag is not set, NEURAL_DEMO should fall back to HeuristicAI.
    monkeypatch.delenv("AI_ENGINE_NEURAL_DEMO_ENABLED", raising=False)
    ai = _create_ai_instance(AIType.NEURAL_DEMO, 1, config)
    assert isinstance(ai, HeuristicAI)

    # When the flag is enabled, _create_ai_instance should attempt to build
    # a NeuralNetAI instance. Patch the underlying class to avoid heavy
    # initialisation and filesystem access in tests.
    import app.ai.neural_net as nn_mod

    class DummyNeuralAI:
        def __init__(self, player_number, cfg):
            self.player_number = player_number
            self.config = cfg

    monkeypatch.setenv("AI_ENGINE_NEURAL_DEMO_ENABLED", "1")
    monkeypatch.setattr(nn_mod, "NeuralNetAI", DummyNeuralAI, raising=True)

    ai2 = _create_ai_instance(AIType.NEURAL_DEMO, 1, config)
    assert isinstance(ai2, DummyNeuralAI)
