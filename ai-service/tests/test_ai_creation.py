import pytest
from app.main import (
    _create_ai_instance,
    _select_ai_type,
    _get_difficulty_profile,
)
from app.models import AIType, AIConfig
from app.ai.minimax_ai import MinimaxAI
from app.ai.mcts_ai import MCTSAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI
from app.ai.descent_ai import DescentAI

# Test timeout guards to prevent hanging in CI
TEST_TIMEOUT_SECONDS = 30


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_select_ai_type():
    # Canonical mapping:
    #   1→Random, 2→Heuristic, 3–6→Minimax, 7–8→MCTS, 9–10→Descent
    assert _select_ai_type(1) == AIType.RANDOM
    assert _select_ai_type(2) == AIType.HEURISTIC
    assert _select_ai_type(3) == AIType.MINIMAX
    assert _select_ai_type(4) == AIType.MINIMAX
    assert _select_ai_type(6) == AIType.MINIMAX
    assert _select_ai_type(7) == AIType.MCTS
    assert _select_ai_type(8) == AIType.MCTS
    assert _select_ai_type(9) == AIType.DESCENT
    assert _select_ai_type(10) == AIType.DESCENT


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_difficulty_profile_mapping():
    """Canonical ladder profiles for difficulties 1–10."""
    profiles = {
        1: (AIType.RANDOM, 0.5, 150, "v1-random-1"),
        2: (AIType.HEURISTIC, 0.3, 200, "v1-heuristic-2"),
        3: (AIType.MINIMAX, 0.2, 1250, "v1-minimax-3"),
        4: (AIType.MINIMAX, 0.1, 2100, "v1-minimax-4"),
        5: (AIType.MINIMAX, 0.05, 3500, "v1-minimax-5"),
        6: (AIType.MINIMAX, 0.02, 4800, "v1-minimax-6"),
        7: (AIType.MCTS, 0.0, 7000, "v1-mcts-7"),
        8: (AIType.MCTS, 0.0, 9600, "v1-mcts-8"),
        9: (AIType.DESCENT, 0.0, 12600, "v1-descent-9"),
        10: (AIType.DESCENT, 0.0, 16000, "v1-descent-10"),
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

    # Patch HexNeuralNet used by DescentAI to a lightweight stub so that
    # constructing a DescentAI instance does not allocate large tensors.
    import app.ai.descent_ai as descent_mod

    class DummyHexNet:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def to(self, *_args, **_kwargs):
            return self

    monkeypatch.setattr(
        descent_mod,
        "HexNeuralNet",
        DummyHexNet,
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
