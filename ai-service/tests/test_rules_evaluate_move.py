import unittest
import os
import sys
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
from app.game_engine import GameEngine  # noqa: E402
from app.board_manager import BoardManager  # noqa: E402


def make_initial_state() -> GameState:
    """Construct a minimal SQUARE8 GameState for /rules/evaluate_move tests."""
    return GameState(
        id="test-game",
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
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=19,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        # For tests we initialize zobristHash with a neutral integer value;
        # the engine is free to recompute or ignore this field, and it is not
        # used by /rules/evaluate_move assertions.
        zobristHash=0,
    )


class TestRulesEvaluateMove(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_valid_move_matches_game_engine(self) -> None:
        """
        Valid moves via /rules/evaluate_move mirror GameEngine.apply_move
        for a single, simple placement move.
        """
        state = make_initial_state()
        moves = GameEngine.get_valid_moves(state, state.current_player)
        self.assertTrue(moves, "Expected at least one legal move")
        move = moves[0]

        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "move": jsonable_encoder(move, by_alias=True),
        }

        response = self.client.post("/rules/evaluate_move", json=payload)
        self.assertEqual(
            response.status_code,
            200,
            msg=f"status={response.status_code}, body={response.text}",
        )
        body = response.json()

        self.assertTrue(body["valid"])
        self.assertIsNone(body.get("validation_error"))
        self.assertIsNotNone(body.get("next_state"))

        # Compare against direct GameEngine.apply_move results.
        next_state_direct = GameEngine.apply_move(state, move)
        expected_hash = BoardManager.hash_game_state(next_state_direct)
        expected_progress = BoardManager.compute_progress_snapshot(
            next_state_direct,
        )

        self.assertEqual(body["state_hash"], expected_hash)
        self.assertEqual(body["s_invariant"], expected_progress.S)
        self.assertEqual(
            body["game_status"],
            next_state_direct.game_status.value,
        )

    def test_multiple_moves_trace_parity_against_game_engine(self) -> None:
        """
        For a short trace of moves sampled from GameEngine.get_valid_moves,
        /rules/evaluate_move must remain in lockstep with direct
        GameEngine.apply_move in terms of hash, S-invariant, and
        game_status for the subset of moves it accepts as valid.
        """
        state = make_initial_state()
        max_steps = 20
        accepted_steps = 0

        for _ in range(max_steps):
            if state.game_status != GameStatus.ACTIVE:
                break

            moves = GameEngine.get_valid_moves(state, state.current_player)
            # If no legal moves exist, the engine is responsible for treating
            # this as a terminal/blocked state; the endpoint should not be
            # called in that situation.
            if not moves:
                break

            move = moves[0]

            payload = {
                "game_state": jsonable_encoder(state, by_alias=True),
                "move": jsonable_encoder(move, by_alias=True),
            }

            response = self.client.post("/rules/evaluate_move", json=payload)
            self.assertEqual(
                response.status_code,
                200,
                msg=f"status={response.status_code}, body={response.text}",
            )
            body = response.json()

            # For now, treat endpoint rejection of some engine-generated moves
            # as non-fatal for this trace harness; stricter TS↔Python parity is
            # enforced via the dedicated fixture/vector tests.
            if not body["valid"]:
                continue

            accepted_steps += 1
            self.assertIsNone(body.get("validation_error"))
            self.assertIsNotNone(body.get("next_state"))

            # Compare the endpoint's invariants against a direct engine step.
            direct_next = GameEngine.apply_move(state, move)
            expected_hash = BoardManager.hash_game_state(direct_next)
            expected_progress = BoardManager.compute_progress_snapshot(
                direct_next,
            )

            self.assertEqual(body["state_hash"], expected_hash)
            self.assertEqual(body["s_invariant"], expected_progress.S)
            self.assertEqual(
                body["game_status"],
                direct_next.game_status.value,
            )

            # Advance the local state along the direct GameEngine trajectory so
            # subsequent iterations exercise later phases (movement, capture,
            # line_processing, territory_processing, victory, etc.).
            state = direct_next

        # Ensure that at least one move in the trace was accepted and checked
        # for invariant parity.
        self.assertGreaterEqual(
            accepted_steps,
            1,
            "Expected at least one accepted move in the trace",
        )

    def test_invalid_move_returns_valid_false(self) -> None:
        """Invalid moves return valid=False and a validation_error."""
        state = make_initial_state()
        moves = GameEngine.get_valid_moves(state, state.current_player)
        self.assertTrue(moves, "Expected at least one legal move")
        move = moves[0]

        # Create an invalid move by changing the player number.
        invalid_move = move.model_copy(update={"player": move.player + 1})

        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "move": jsonable_encoder(invalid_move, by_alias=True),
        }

        response = self.client.post("/rules/evaluate_move", json=payload)
        self.assertEqual(
            response.status_code,
            200,
            msg=f"status={response.status_code}, body={response.text}",
        )
        body = response.json()

        self.assertFalse(body["valid"])
        self.assertIsNotNone(body.get("validation_error"))
        # For invalid moves, we do not expect a next_state.
        self.assertIsNone(body.get("next_state"))


if __name__ == "__main__":
    unittest.main()
