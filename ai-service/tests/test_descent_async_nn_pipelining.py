from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from app.ai.descent_ai import DescentAI
from app.models import AIConfig, BoardType


def _future_result(values: list[float]) -> Future:
    fut: Future = Future()
    fut.set_result((values, None))
    return fut


def test_descent_async_pipelines_child_batches_in_legacy_expansion() -> None:
    config = AIConfig(difficulty=5, think_time=0, use_neural_net=False)
    ai = DescentAI(player_number=1, config=config)
    ai.use_incremental_search = False

    # Inject a mock NN + batcher and enable async mode.
    ai.neural_net = MagicMock()
    ai.neural_net.encode_move.return_value = 0
    ai.nn_batcher = MagicMock()
    ai.nn_batcher.evaluate.return_value = ([0.0], [[1.0]])
    ai.nn_batcher.submit.side_effect = [
        _future_result([0.1, 0.2]),
        _future_result([0.3, 0.4]),
        _future_result([0.5]),
    ]
    ai.enable_async_nn_eval = True

    with patch.object(ai, "_default_nn_batch_size", return_value=2):
        ai.transposition_table = MagicMock()
        ai.transposition_table.get.return_value = None
        ai.transposition_table.put.return_value = None

        rules_engine = MagicMock()
        ai.rules_engine = rules_engine

        move1, move2, move3, move4, move5 = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        rules_engine.get_valid_moves.return_value = [
            move1,
            move2,
            move3,
            move4,
            move5,
        ]

        def apply_move(state, move):
            child = MagicMock()
            child.game_status = "active"
            child.winner = None
            child.players = state.players
            child.current_player = 2
            child.board = state.board
            child.zobrist_hash = None
            return child

        rules_engine.apply_move.side_effect = apply_move

        root = MagicMock()
        root.game_status = "active"
        root.current_player = 1
        root.players = [MagicMock(), MagicMock()]
        root.board = MagicMock()
        root.board.type = BoardType.SQUARE8
        root.zobrist_hash = 123

        val = ai._descent_iteration(root)

    # With batch_size=2 and 5 non-terminal children => 3 submits.
    assert ai.nn_batcher.submit.call_count == 3
    # Values are from current-player perspective; children are opponent-to-move => negated.
    assert val == -0.1

