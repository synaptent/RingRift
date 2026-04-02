from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pytest

# Ensure app.* imports resolve when running pytest from repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.ai.mcts_ai import MCTSAI, MCTSNode
from app.ai.neural_net import INVALID_MOVE_INDEX
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Player,
    Position,
    TimeControl,
)


def _make_square8_state() -> GameState:
    now = datetime.now()
    board = BoardState(type=BoardType.SQUARE8, size=8)
    players = [
        Player(
            id="p1",
            username="p1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="p2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]
    total_rings_in_play = sum(player.rings_in_hand for player in players)
    return GameState(
        id="async-eval-test",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=total_rings_in_play,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=777,
    )


class _StubNN:
    def __init__(self, probs: list[float], idx_by_move_id: dict[str, int]):
        self._probs = probs
        self._idx = idx_by_move_id
        self.device = "cuda"

    def evaluate_batch(self, game_states: list[GameState], value_head: int | None = None):
        policy = np.array([self._probs], dtype=np.float32)
        values = [0.5 for _ in game_states]
        return values, policy

    def encode_move(self, move: Move, board_context):
        return self._idx.get(move.id, INVALID_MOVE_INDEX)


def test_async_prepare_finish_updates_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _make_square8_state()
    now = state.created_at
    moves = [
        Move(
            id="m1",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=0, y=0),
            timestamp=now,
            thinkTime=0,
            moveNumber=1,
        ),
        Move(
            id="m2",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=1, y=0),
            timestamp=now,
            thinkTime=0,
            moveNumber=2,
        ),
    ]

    config = AIConfig(difficulty=6, use_neural_net=False, use_incremental_search=False)
    mcts = MCTSAI(player_number=1, config=config)
    mcts.neural_net = _StubNN(
        probs=[0.2, 0.8],
        idx_by_move_id={"m1": 0, "m2": 1},
    )

    monkeypatch.setattr(
        mcts.rules_engine,
        "get_valid_moves",
        lambda s, pid: moves,
    )

    root = MCTSNode(state)
    leaves = [(root, state, [])]

    batch, future = mcts._prepare_leaf_evaluation_legacy(leaves)  # type: ignore[attr-defined]
    mcts._finish_leaf_evaluation_legacy(batch, future)  # type: ignore[attr-defined]

    assert root.visits == 1
    assert root.wins == pytest.approx(0.5)
    assert root.policy_map[str(moves[1])] > root.policy_map[str(moves[0])]
    assert sum(root.policy_map.values()) == pytest.approx(1.0)
