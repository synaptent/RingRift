import unittest
from unittest.mock import MagicMock
from datetime import datetime

import numpy as np
import torch

from app.ai.descent_ai import DescentAI
from app.models import (
    GameState,
    AIConfig,
    BoardType,
    GamePhase,
    GameStatus,
    TimeControl,
    BoardState,
    Player,
)


class TestDescentAIHex(unittest.TestCase):
    """Smoke tests for DescentAI on hex boards.

    These tests do not assert on search quality; they simply ensure that the
    hex-specific neural-network path (HexNeuralNet + ActionEncoderHex) can be
    exercised without errors when the board type is HEXAGONAL.
    """

    def setUp(self) -> None:
        self.config = AIConfig(difficulty=5, think_time=1000, randomness=0.1)
        self.ai = DescentAI(player_number=1, config=self.config)

        board = BoardState(
            type=BoardType.HEXAGONAL,
            size=11,  # canonical radius-10 hex (2N+1 = 21 frame)
            stacks={},
            markers={},
            collapsedSpaces={},
            eliminatedRings={},
        )

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

        now = datetime.now()

        self.game_state = GameState(
            id="hex-descent-test",
            boardType=BoardType.HEXAGONAL,
            board=board,
            players=players,
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=60,
                increment=0,
                type="blitz",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt=now,
            lastMoveAt=now,
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=19,
            territoryVictoryThreshold=33,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
        )

    def test_select_move_uses_hex_network_for_hex_board(self) -> None:
        """Selecting a move on a hex board should hit the hex NN path.

        The test stubs out the rules engine, neural-net feature encoder,
        hex encoder, and hex model so that DescentAI's hex-specific policy
        evaluation runs in a controlled way and returns a legal move.
        """
        # Keep the search very small for test speed.
        self.ai.config.think_time = 10  # milliseconds

        # Single mocked move that is always legal.
        mock_move = MagicMock()
        mock_move.type = "move_stack"
        mock_move.to = MagicMock()
        mock_move.to.x = 0
        mock_move.to.y = 0

        # Stub the rules engine used by BaseAI / DescentAI.
        rules_engine = MagicMock()
        rules_engine.get_valid_moves.return_value = [mock_move]
        rules_engine.apply_move.side_effect = lambda state, move: state
        self.ai.rules_engine = rules_engine

        # Mock out the shared NeuralNetAI encoder.
        self.ai.neural_net = MagicMock()
        self.ai.neural_net._extract_features.return_value = (
            np.zeros((10, 21, 21), dtype=np.float32),
            np.zeros((10,), dtype=np.float32),
        )

        # Hex encoder: always map moves to index 0 in the (small) policy
        # vector returned by the fake hex model.
        self.ai.hex_encoder = MagicMock()
        self.ai.hex_encoder.encode_move.return_value = 0

        # Fake hex model that returns zero values and a single-logit policy
        # vector per state. This keeps the shapes valid while avoiding any
        # dependency on real model weights.
        def fake_hex_model(x, globals_vec, hex_mask=None):
            batch = x.shape[0]
            values = torch.zeros((batch, 1))
            logits = torch.zeros((batch, 1))
            return values, logits

        self.ai.hex_model = MagicMock(side_effect=fake_hex_model)

        move = self.ai.select_move(self.game_state)

        self.assertIsNotNone(move)
        self.assertEqual(move, mock_move)
        # Ensure that the hex model path was exercised.
        self.assertTrue(self.ai.hex_model.called)


if __name__ == "__main__":
    unittest.main()
