import os
import sys
import unittest
from datetime import datetime

from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder

# Ensure app package is importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.main import app  # noqa: E402
from app.models import (  # noqa: E402
    GameState,
    BoardState,
    BoardType,
    GamePhase,
    GameStatus,
    TimeControl,
    Player,
)


def make_initial_state() -> GameState:
  """Construct a minimal SQUARE8 GameState for /ai/evaluate_position tests."""
  return GameState(
      id="eval-position-test-game",
      boardType=BoardType.SQUARE8,
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
      createdAt=datetime.now(),
      lastMoveAt=datetime.now(),
      isRated=False,
      maxPlayers=2,
      totalRingsInPlay=36,
      totalRingsEliminated=0,
      victoryThreshold=18,  # RR-CANON-R061: ringsPerPlayer
      territoryVictoryThreshold=33,
      chainCaptureState=None,
      mustMoveFromStackKey=None,
      zobristHash=0,
  )


class TestEvaluatePositionAPI(unittest.TestCase):
  def setUp(self) -> None:
      self.client = TestClient(app)

  def test_evaluate_position_returns_per_player_scores(self) -> None:
      state = make_initial_state()

      payload = {
          "game_state": jsonable_encoder(state, by_alias=True),
      }

      response = self.client.post("/ai/evaluate_position", json=payload)
      self.assertEqual(
          response.status_code,
          200,
          msg=f"status={response.status_code}, body={response.text}",
      )

      body = response.json()

      # Basic top-level fields
      self.assertEqual(body["game_id"], state.id)
      self.assertEqual(body["board_type"], state.board_type.value)
      self.assertEqual(body["move_number"], 0)
      self.assertEqual(body["evaluation_scale"], "zero_sum_margin")
      self.assertIn("engine_profile", body)
      self.assertIn("generated_at", body)

      # Per-player evaluations should be present and roughly zero-sum.
      per_player = body["per_player"]
      self.assertIn("1", per_player)
      self.assertIn("2", per_player)

      total_sum = float(per_player["1"]["totalEval"]) + float(per_player["2"]["totalEval"])
      self.assertAlmostEqual(total_sum, 0.0, places=3)

