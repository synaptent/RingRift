import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.ai.descent_ai import DescentAI
from app.ai.mcts_ai import MCTSAI
from app.ai.minimax_ai import MinimaxAI
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
    RingStack,
    TimeControl,
)


def _make_swap_state() -> GameState:
    now = datetime.now()
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

    # Strong P1 opening: a center-adjacent stack.
    stack_pos = Position(x=3, y=3)
    stack = RingStack(
        position=stack_pos,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )

    p1_move = Move(
        id="p1-open",
        type=MoveType.PLACE_RING,
        player=1,
        to=Position(x=3, y=3),
        placementCount=1,
        placedOnStack=False,
        timestamp=now,
        think_time=0,
        move_number=1,
    )

    return GameState(
        id="swap-test",
        boardType=BoardType.SQUARE8,
        board=BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={stack_pos.to_key(): stack},
            markers={},
            collapsedSpaces={},
            eliminatedRings={},
        ),
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=2,
        moveHistory=[p1_move],
        timeControl=TimeControl(
            initialTime=600,
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
        victoryThreshold=18,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        rulesOptions={"swapRuleEnabled": True},
    )


class TestSwapSupportForSearchAIs(unittest.TestCase):
    def test_descent_selects_swap_when_classifier_positive(self) -> None:
        game_state = _make_swap_state()
        ai = DescentAI(player_number=2, config=AIConfig(difficulty=5, think_time=1))

        swap_move = Move(
            id="swap",
            type=MoveType.SWAP_SIDES,
            player=2,
            timestamp=datetime.now(),
            think_time=0,
            move_number=2,
        )
        other_move = MagicMock()

        with patch(
            "app.ai.swap_evaluation.SwapEvaluator.evaluate_swap_with_classifier",
            return_value=5.0,
        ), patch.object(ai, "get_valid_moves", return_value=[swap_move, other_move]):
            selected = ai.select_move(game_state)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.type, MoveType.SWAP_SIDES)

    def test_minimax_selects_swap_when_classifier_positive(self) -> None:
        game_state = _make_swap_state()
        ai = MinimaxAI(player_number=2, config=AIConfig(difficulty=5, think_time=1))

        swap_move = Move(
            id="swap",
            type=MoveType.SWAP_SIDES,
            player=2,
            timestamp=datetime.now(),
            think_time=0,
            move_number=2,
        )
        other_move = MagicMock()

        with patch(
            "app.ai.swap_evaluation.SwapEvaluator.evaluate_swap_with_classifier",
            return_value=5.0,
        ), patch.object(ai, "get_valid_moves", return_value=[swap_move, other_move]):
            selected = ai.select_move(game_state)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.type, MoveType.SWAP_SIDES)

    def test_mcts_selects_swap_and_returns_one_hot_policy(self) -> None:
        game_state = _make_swap_state()
        ai = MCTSAI(player_number=2, config=AIConfig(difficulty=6, think_time=1))

        swap_move = Move(
            id="swap",
            type=MoveType.SWAP_SIDES,
            player=2,
            timestamp=datetime.now(),
            think_time=0,
            move_number=2,
        )
        other_move = MagicMock()

        with patch(
            "app.ai.swap_evaluation.SwapEvaluator.evaluate_swap_with_classifier",
            return_value=5.0,
        ), patch.object(ai, "get_valid_moves", return_value=[swap_move, other_move]):
            selected, policy = ai.select_move_and_policy(game_state)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.type, MoveType.SWAP_SIDES)
        self.assertEqual(policy, {str(swap_move): 1.0})
