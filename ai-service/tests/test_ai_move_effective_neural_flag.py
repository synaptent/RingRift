import os
import sys
from datetime import datetime
from typing import Any

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

# Ensure app package is importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import app.main as main_mod
from app.config.ladder_config import LadderTierConfig
from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)


def _make_square8_2p_state() -> GameState:
    now = datetime.now()
    return GameState(
        id="ai-move-effective-neural-flag-test",
        boardType=BoardType.SQUARE8,
        rngSeed=None,
        board=BoardState(type=BoardType.SQUARE8, size=8),
        players=[
            Player(
                id="p1",
                username="P1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=20,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
            Player(
                id="p2",
                username="P2",
                type="human",
                playerNumber=2,
                isReady=True,
                timeRemaining=600,
                ringsInHand=20,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
        ],
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        spectators=[],
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
        mustMoveFromStackKey=None,
        zobristHash=0,
    )


@pytest.mark.timeout(30)
def test_ai_move_use_neural_net_reflects_effective_mcts(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(main_mod.app)

    ladder_cfg = LadderTierConfig(
        difficulty=6,
        board_type=BoardType.SQUARE8,
        num_players=2,
        ai_type=main_mod.AIType.MCTS,
        model_id="ringrift_best_sq8_2p",
        heuristic_profile_id="heuristic_v1_sq8_2p",
        randomness=0.0,
        think_time_ms=10,
        use_neural_net=True,
        notes="test ladder cfg",
    )

    monkeypatch.setattr(main_mod, "get_ladder_tier_config", lambda *_args, **_kwargs: ladder_cfg)
    monkeypatch.setattr(main_mod, "_should_cache_ai", lambda *_args, **_kwargs: False)

    class DummyAI:
        def __init__(self, neural_net: Any | None):
            self.neural_net = neural_net

        def select_move(self, _game_state: GameState):
            return None

        def evaluate_position(self, _game_state: GameState) -> float:
            return 0.0

    state = _make_square8_2p_state()
    payload = {
        "game_state": jsonable_encoder(state, by_alias=True),
        "player_number": 1,
        "difficulty": 6,
    }

    monkeypatch.setattr(
        main_mod,
        "_create_ai_instance",
        lambda _ai_type, _player_number, _config, board_type=None: DummyAI(neural_net=None),
    )
    response = client.post("/ai/move", json=payload)
    assert response.status_code == 200, response.text
    assert response.json().get("use_neural_net") is False

    monkeypatch.setattr(
        main_mod,
        "_create_ai_instance",
        lambda _ai_type, _player_number, _config, board_type=None: DummyAI(neural_net=object()),
    )
    response = client.post("/ai/move", json=payload)
    assert response.status_code == 200, response.text
    assert response.json().get("use_neural_net") is True


@pytest.mark.timeout(30)
def test_ai_move_use_neural_net_reflects_effective_minimax_nnue(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(main_mod.app)

    ladder_cfg = LadderTierConfig(
        difficulty=4,
        board_type=BoardType.SQUARE8,
        num_players=2,
        ai_type=main_mod.AIType.MINIMAX,
        model_id="nnue_square8_2p",
        heuristic_profile_id="heuristic_v1_sq8_2p",
        randomness=0.0,
        think_time_ms=10,
        use_neural_net=True,
        notes="test ladder cfg",
    )

    monkeypatch.setattr(main_mod, "get_ladder_tier_config", lambda *_args, **_kwargs: ladder_cfg)
    monkeypatch.setattr(main_mod, "_should_cache_ai", lambda *_args, **_kwargs: False)

    class DummyAI:
        def __init__(self, use_nnue: bool):
            self.use_nnue = use_nnue

        def select_move(self, _game_state: GameState):
            return None

        def evaluate_position(self, _game_state: GameState) -> float:
            return 0.0

    state = _make_square8_2p_state()
    payload = {
        "game_state": jsonable_encoder(state, by_alias=True),
        "player_number": 1,
        "difficulty": 4,
    }

    monkeypatch.setattr(
        main_mod,
        "_create_ai_instance",
        lambda _ai_type, _player_number, _config, board_type=None: DummyAI(use_nnue=False),
    )
    response = client.post("/ai/move", json=payload)
    assert response.status_code == 200, response.text
    assert response.json().get("use_neural_net") is False

    monkeypatch.setattr(
        main_mod,
        "_create_ai_instance",
        lambda _ai_type, _player_number, _config, board_type=None: DummyAI(use_nnue=True),
    )
    response = client.post("/ai/move", json=payload)
    assert response.status_code == 200, response.text
    assert response.json().get("use_neural_net") is True
