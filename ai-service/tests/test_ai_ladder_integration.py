import os
import sys
from datetime import datetime
from typing import Any, Dict

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

# Ensure app package is importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import app.main as main_mod  # noqa: E402
from app.config.ladder_config import LadderTierConfig  # noqa: E402
from app.models import (  # noqa: E402
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)


def _make_square8_2p_state() -> GameState:
    """Construct a minimal square8 2p GameState for /ai/move tests."""
    now = datetime.now()
    return GameState(
        id="ladder-integration-test",
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
        victoryThreshold=18,  # RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=0,
        lpsRoundIndex=0,
        lpsExclusivePlayerForCompletedRound=None,
    )


@pytest.mark.timeout(30)
def test_ai_move_uses_ladder_config_overrides(monkeypatch) -> None:
    """get_ai_move should honour LadderTierConfig for square8 2p tiers."""
    client = TestClient(main_mod.app)

    sentinel_randomness = 0.42
    sentinel_think_time = 1234
    sentinel_profile = "heuristic_v1_2p_test"

    ladder_cfg = LadderTierConfig(
        difficulty=2,
        board_type=BoardType.SQUARE8,
        num_players=2,
        ai_type=main_mod.AIType.HEURISTIC,
        model_id=None,
        heuristic_profile_id=sentinel_profile,
        randomness=sentinel_randomness,
        think_time_ms=sentinel_think_time,
        notes="test override",
    )

    def _fake_get_ladder_tier_config(
        difficulty: int,
        board_type: BoardType,
        num_players: int,
    ) -> LadderTierConfig:
        assert difficulty == 2
        assert board_type == BoardType.SQUARE8
        assert num_players == 2
        return ladder_cfg

    monkeypatch.setattr(
        main_mod,
        "get_ladder_tier_config",
        _fake_get_ladder_tier_config,
    )

    captured: Dict[str, Any] = {}

    def _fake_create_ai_instance(ai_type, player_number, config):
        captured["ai_type"] = ai_type
        captured["config"] = config

        class DummyAI:
            def __init__(self, player_number: int, cfg: Any) -> None:
                self.player_number = player_number
                self.config = cfg

            def select_move(self, game_state: GameState):
                return None

            def evaluate_position(self, game_state: GameState) -> float:
                return 0.0

        return DummyAI(player_number, config)

    monkeypatch.setattr(
        main_mod,
        "_create_ai_instance",
        _fake_create_ai_instance,
    )

    state = _make_square8_2p_state()
    payload = {
        "game_state": jsonable_encoder(state, by_alias=True),
        "player_number": 1,
        "difficulty": 2,
    }

    response = client.post("/ai/move", json=payload)
    assert response.status_code == 200, response.text

    assert "config" in captured
    cfg = captured["config"]

    assert cfg.difficulty == 2
    assert cfg.randomness == pytest.approx(sentinel_randomness)
    assert cfg.think_time == sentinel_think_time
    assert cfg.heuristic_profile_id == sentinel_profile