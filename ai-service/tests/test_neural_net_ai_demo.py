"""Smoke tests for the NeuralNetAI demo engine path (AIType.NEURAL_DEMO)."""

from __future__ import annotations

import os
import sys

import pytest

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.main import _create_ai_instance  # noqa: E402
from app.models import AIConfig, AIType, BoardType  # noqa: E402
from app.training.generate_data import create_initial_state  # noqa: E402
from app.rules.default_engine import DefaultRulesEngine  # noqa: E402
from app.ai.neural_net import NeuralNetAI  # noqa: E402


TEST_TIMEOUT_SECONDS = 60


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_neural_demo_select_move_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instantiate NeuralNetAI via NEURAL_DEMO on Square-8 2-player.

    This test exercises the experimental NEURAL_DEMO engine path end-to-end:

    - Enables the AI_ENGINE_NEURAL_DEMO_ENABLED gate.
    - Forces CPU to avoid unexpected MPS/CUDA interactions in CI.
    - Uses create_initial_state for a canonical Square-8 2-player start.
    - Constructs an AIConfig with allow_fresh_weights semantics via
      nn_model_id and default NeuralNetAI behaviour.
    - Verifies that select_move returns a non-None, legal move.
    """
    # Suppress torch.compile/dynamo errors which can fail on some platforms
    # (e.g., macOS missing libc++.1.dylib for inductor)
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except ImportError:
        pass  # torch._dynamo not available

    # Enable the experimental neural engine and force CPU usage for stability.
    monkeypatch.setenv("AI_ENGINE_NEURAL_DEMO_ENABLED", "1")
    monkeypatch.setenv("RINGRIFT_FORCE_CPU", "1")

    # Simple Square-8 2-player initial state.
    state = create_initial_state(BoardType.SQUARE8, num_players=2)

    cfg = AIConfig(
        difficulty=9,
        think_time=0,
        randomness=0.0,
        rngSeed=123,
        nn_model_id="sq8_nn_demo",
        allow_fresh_weights=True,
    )

    ai = _create_ai_instance(AIType.NEURAL_DEMO, 1, cfg)
    assert isinstance(ai, NeuralNetAI)

    # Use the canonical DefaultRulesEngine to compute legal moves for sanity.
    engine = DefaultRulesEngine()
    legal_moves = engine.get_valid_moves(state, state.current_player)
    assert legal_moves, "Expected legal moves from initial Square-8 state"

    move = ai.select_move(state)
    assert move is not None
    assert move in legal_moves
