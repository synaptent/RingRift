import pytest
from datetime import datetime

from app.ai.mcts_ai import MCTSAI, MCTSNode
from app.ai.descent_ai import DescentAI
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    TimeControl,
)


def _minimal_square8_state() -> GameState:
    now = datetime.now()
    return GameState(
        id="test-game",
        boardType=BoardType.SQUARE8,
        board=BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={},
            markers={},
            collapsedSpaces={},
            eliminatedRings={},
        ),
        players=[],
        currentPhase=GamePhase.MOVEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=18,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
    )


class _ExplodingNeuralNet:
    def evaluate_batch(self, _states, **_kwargs):
        raise RuntimeError("boom")


def test_mcts_require_neural_net_raises_on_runtime_eval_failure():
    config = AIConfig(difficulty=5, think_time=0, randomness=0.0)
    ai = MCTSAI(player_number=1, config=config)

    ai.neural_net = _ExplodingNeuralNet()
    ai.require_neural_net = True

    state = _minimal_square8_state()
    root = MCTSNode(state)
    leaves = [(root, state, [])]

    with pytest.raises(RuntimeError, match="boom"):
        ai._evaluate_leaves_legacy(leaves, root)


def test_mcts_without_require_neural_net_falls_back_and_disables_nn(monkeypatch):
    config = AIConfig(difficulty=5, think_time=0, randomness=0.0)
    ai = MCTSAI(player_number=1, config=config)

    ai.neural_net = _ExplodingNeuralNet()
    ai.require_neural_net = False

    # Avoid calling the real heuristic rollout in this unit test.
    monkeypatch.setattr(ai, "_heuristic_rollout_legacy", lambda _state: 0.0)

    state = _minimal_square8_state()
    root = MCTSNode(state)
    leaves = [(root, state, [])]

    ai._evaluate_leaves_legacy(leaves, root)
    assert ai.neural_net is None


def test_descent_require_neural_net_raises_on_batch_eval_failure():
    config = AIConfig(difficulty=9, think_time=0, randomness=0.0, use_neural_net=False)
    ai = DescentAI(player_number=1, config=config)

    ai.neural_net = _ExplodingNeuralNet()
    ai.nn_batcher = None
    ai.enable_async_nn_eval = False
    ai.require_neural_net = True

    with pytest.raises(RuntimeError, match="boom"):
        ai._batch_evaluate_positions([_minimal_square8_state()])


def test_descent_without_require_neural_net_falls_back_and_disables_nn(monkeypatch):
    config = AIConfig(difficulty=9, think_time=0, randomness=0.0, use_neural_net=False)
    ai = DescentAI(player_number=1, config=config)

    ai.neural_net = _ExplodingNeuralNet()
    ai.nn_batcher = None
    ai.enable_async_nn_eval = False
    ai.require_neural_net = False

    monkeypatch.setattr(ai, "evaluate_position", lambda _state: 0.0)

    values = ai._batch_evaluate_positions([_minimal_square8_state()])
    assert values == [0.0]
    assert ai.neural_net is None
