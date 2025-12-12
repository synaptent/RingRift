import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.ai.descent_ai import DescentAI
from app.ai.mcts_ai import MCTSAI, MCTSNode, MCTSNodeLite
from app.ai.minimax_ai import MinimaxAI
from app.ai.game_state_utils import select_threat_opponent, victory_progress_for_player
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)
from app.rules.mutable_state import MoveUndo


def _make_square8_state(num_players: int) -> GameState:
    now = datetime.now()
    players = [
        Player(
            id=f"p{i}",
            username=f"p{i}",
            type="human",
            playerNumber=i,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        )
        for i in range(1, num_players + 1)
    ]

    return GameState(
        id=f"mp-{num_players}p",
        boardType=BoardType.SQUARE8,
        board=BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={},
            markers={},
            collapsedSpaces={},
            eliminatedRings={},
        ),
        players=players,
        currentPhase=GamePhase.MOVEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(
            initialTime=600,
            increment=0,
            type="blitz",
        ),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=18 * num_players,
        totalRingsEliminated=0,
        victoryThreshold=18,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
    )


class TestMultiplayerAIRouting(unittest.TestCase):
    def test_minimax_3p_uses_search_path(self) -> None:
        config = AIConfig(difficulty=5, use_neural_net=False)
        ai = MinimaxAI(player_number=1, config=config)
        game_state = _make_square8_state(num_players=3)

        mock_move = MagicMock()
        with patch.object(ai, "get_valid_moves", return_value=[mock_move]), patch.object(
            ai, "should_pick_random_move", return_value=False
        ), patch.object(ai, "_select_move_incremental", return_value=mock_move) as mocked:
            ai.use_incremental_search = True
            selected = ai.select_move(game_state)

        self.assertIs(selected, mock_move)
        mocked.assert_called_once()

    def test_descent_3p_uses_search_path_and_keeps_neural(self) -> None:
        config = AIConfig(difficulty=5, think_time=10, use_neural_net=False)
        ai = DescentAI(player_number=1, config=config)
        game_state = _make_square8_state(num_players=3)

        # Simulate a preloaded NN so the multi-player path keeps it.
        ai.neural_net = MagicMock()
        ai.hex_encoder = MagicMock()

        mock_move = MagicMock()
        with patch.object(ai, "get_valid_moves", return_value=[mock_move]), patch.object(
            ai, "should_pick_random_move", return_value=False
        ), patch.object(ai, "_select_move_incremental", return_value=mock_move) as mocked:
            ai.use_incremental_search = True
            selected = ai.select_move(game_state)

        self.assertIs(selected, mock_move)
        mocked.assert_called_once()
        self.assertIsNotNone(ai.neural_net)

    def test_mcts_3p_uses_search_path_and_keeps_neural(self) -> None:
        config = AIConfig(difficulty=5, think_time=10, use_neural_net=False)
        ai = MCTSAI(player_number=1, config=config)
        game_state = _make_square8_state(num_players=3)

        # Simulate a preloaded NN so the multi-player path keeps it.
        ai.neural_net = MagicMock()
        ai.hex_encoder = MagicMock()
        ai.hex_model = MagicMock()

        mock_move = MagicMock()
        mock_policy = {"mock": 1.0}
        with patch.object(ai, "get_valid_moves", return_value=[mock_move]), patch.object(
            ai, "should_pick_random_move", return_value=False
        ), patch.object(ai, "_search_incremental", return_value=(mock_move, mock_policy)) as mocked:
            ai.use_incremental_search = True
            selected, policy = ai.select_move_and_policy(game_state)

        self.assertIs(selected, mock_move)
        self.assertIs(policy, mock_policy)
        mocked.assert_called_once()
        self.assertIsNotNone(ai.neural_net)


class TestMCTSParanoidBackprop(unittest.TestCase):
    def test_legacy_backprop_does_not_flip_between_opponents(self) -> None:
        ai = MCTSAI(
            player_number=1,
            config=AIConfig(difficulty=1, think_time=1, use_neural_net=False),
        )
        ai.neural_net = None

        root = MCTSNode(MagicMock())
        n1 = MCTSNode(MagicMock(), parent=root, move=MagicMock())
        n2 = MCTSNode(MagicMock(), parent=n1, move=MagicMock())
        leaf = MCTSNode(MagicMock(), parent=n2, move=MagicMock())

        # Path: P1 -> P2 -> P3, leaf to-move is P2 (still opponent coalition).
        played_moves = [
            MagicMock(player=1),
            MagicMock(player=2),
            MagicMock(player=3),
        ]
        leaf_state = MagicMock(current_player=2)

        with patch.object(ai, "_heuristic_rollout_legacy", return_value=10.0):
            ai._evaluate_leaves_legacy([(leaf, leaf_state, played_moves)], root)

        # Coalition nodes should all see the same sign; only the final step back
        # to the root player should flip.
        self.assertEqual(n2.wins, -10.0)
        self.assertEqual(root.wins, 10.0)


class TestThreatOpponentSelection(unittest.TestCase):
    def test_select_threat_opponent_prefers_territory_leader(self) -> None:
        game_state = _make_square8_state(num_players=3)

        # Player 2 is one space from territory victory.
        game_state.players[1].territory_spaces = (
            game_state.territory_victory_threshold - 1
        )
        game_state.players[1].eliminated_rings = 0

        # Player 3 has some eliminations but is further from any win path.
        game_state.players[2].eliminated_rings = 10
        game_state.players[2].territory_spaces = 0

        threat = select_threat_opponent(game_state, perspective_player_number=1)
        self.assertEqual(threat, 2)

    def test_select_threat_opponent_accounts_for_lps_proximity(self) -> None:
        game_state = _make_square8_state(num_players=3)

        # LPS threat: player 3 has already been exclusive for one round.
        game_state.lps_consecutive_exclusive_player = 3
        game_state.lps_consecutive_exclusive_rounds = 1

        # Even if player 2 is ahead on material, LPS proximity should dominate.
        game_state.players[1].territory_spaces = 20  # ~60% progress
        game_state.players[1].eliminated_rings = 9   # 50% progress

        threat = select_threat_opponent(game_state, perspective_player_number=1)
        self.assertEqual(threat, 3)

        self.assertGreater(
            victory_progress_for_player(game_state, 3),
            victory_progress_for_player(game_state, 2),
        )

    def test_incremental_backprop_does_not_flip_between_opponents(self) -> None:
        class FakeMutableState:
            def __init__(self, current_player: int) -> None:
                self.current_player = current_player

            def make_move(self, move) -> None:  # type: ignore[no-untyped-def]
                self.current_player = int(getattr(move, "next_player", self.current_player))

            def unmake_move(self, undo: MoveUndo) -> None:
                self.current_player = int(getattr(undo, "prev_player", self.current_player))

        ai = MCTSAI(
            player_number=1,
            config=AIConfig(difficulty=1, think_time=1, use_neural_net=False),
        )
        ai.neural_net = None

        root = MCTSNodeLite()
        n1 = MCTSNodeLite(parent=root, move=MagicMock())
        n2 = MCTSNodeLite(parent=n1, move=MagicMock())
        leaf = MCTSNodeLite(parent=n2, move=MagicMock())

        move1 = MagicMock(player=1, next_player=2)
        move2 = MagicMock(player=2, next_player=3)
        move3 = MagicMock(player=3, next_player=2)
        path_undos = [
            MoveUndo(move=move1, prev_player=1),
            MoveUndo(move=move2, prev_player=2),
            MoveUndo(move=move3, prev_player=3),
        ]
        played_moves = [move1, move2, move3]

        mutable_state = FakeMutableState(current_player=1)
        with patch.object(ai, "_heuristic_rollout_mutable", return_value=10.0):
            ai._evaluate_leaves_incremental(
                [(leaf, path_undos, played_moves)],
                mutable_state,
                root,
            )

        self.assertEqual(n2.wins, -10.0)
        self.assertEqual(root.wins, 10.0)
