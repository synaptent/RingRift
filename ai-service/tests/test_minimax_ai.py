import unittest
from unittest.mock import MagicMock, patch

from app.ai.bounded_transposition_table import BoundedTranspositionTable
from app.ai.minimax_ai import MinimaxAI
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    Position,
    RingStack,
    TimeControl,
)


class TestMinimaxAI(unittest.TestCase):
    """Unit tests for MinimaxAI that do NOT leak global GameEngine mocks.

    Earlier versions of this test assigned MagicMocks directly to
    ``app.game_engine.GameEngine.get_valid_moves`` and ``apply_move``, which
    persisted after the test finished and broke later tests such as
    ``tests/test_rules_evaluate_move.py``.

    Here we patch methods on the MinimaxAI instance (and its rules_engine)
    so that changes are strictly local to these tests.
    """

    def setUp(self) -> None:
        self.config = AIConfig(difficulty=5)
        self.ai = MinimaxAI(player_number=1, config=self.config)

        # Minimal but valid game state for MinimaxAI to consume.
        self.game_state = GameState(
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
            timeControl=TimeControl(
                initialTime=600,
                increment=0,
                type="blitz",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt="2023-01-01T00:00:00Z",
            lastMoveAt="2023-01-01T00:00:00Z",
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=18,  # RR-CANON-R061: = ringsPerPlayer for 2-player games
            territoryVictoryThreshold=33,
        )

    def test_select_move_returns_move(self) -> None:
        """MinimaxAI.select_move returns a move when legal moves exist.

        We patch ``self.ai.get_valid_moves`` and ``self.ai.rules_engine.apply_move``
        so that no global GameEngine state is modified.
        """

        mock_move1 = MagicMock()
        mock_move1.type = "move_stack"
        mock_move2 = MagicMock()
        mock_move2.type = "move_stack"

        with patch.object(
            self.ai,
            "get_valid_moves",
            return_value=[mock_move1, mock_move2],
        ), patch.object(
            self.ai.rules_engine,
            "apply_move",
            return_value=self.game_state,
        ):
            # Simplify evaluation logic for the purposes of this unit test.
            self.ai.evaluate_position = MagicMock(return_value=10)
            self.ai.should_pick_random_move = MagicMock(return_value=False)

            move = self.ai.select_move(self.game_state)

        self.assertIsNotNone(move)
        self.assertEqual(move, mock_move1)

    def test_select_move_no_valid_moves(self) -> None:
        """MinimaxAI.select_move returns None when there are no legal moves."""

        with patch.object(
            self.ai,
            "get_valid_moves",
            return_value=[],
        ):
            move = self.ai.select_move(self.game_state)

        self.assertIsNone(move)


class TestMinimaxAIMemory(unittest.TestCase):
    """Tests for MinimaxAI memory safety (bounded transposition tables)."""

    def setUp(self) -> None:
        self.config = AIConfig(difficulty=5)
        self.ai = MinimaxAI(player_number=1, config=self.config)

    def test_transposition_table_is_bounded(self) -> None:
        """Verify transposition_table uses BoundedTranspositionTable."""
        self.assertIsInstance(self.ai.transposition_table, BoundedTranspositionTable)
        self.assertEqual(self.ai.transposition_table.max_entries, 100000)

    def test_killer_moves_is_bounded(self) -> None:
        """Verify killer_moves uses BoundedTranspositionTable."""
        self.assertIsInstance(self.ai.killer_moves, BoundedTranspositionTable)
        self.assertEqual(self.ai.killer_moves.max_entries, 10000)

    def test_transposition_table_bounded_respects_limit(self) -> None:
        """Verify transposition table doesn't grow infinitely."""
        # Fill the transposition table beyond capacity
        for i in range(150000):  # 50% more than max_entries
            self.ai.transposition_table.put(i, (10.0, 5))

        # Table size should be bounded at max_entries
        self.assertLessEqual(len(self.ai.transposition_table), 100000)

    def test_killer_moves_bounded_respects_limit(self) -> None:
        """Verify killer moves table doesn't grow infinitely."""
        # Fill the killer moves table beyond capacity
        for i in range(15000):  # 50% more than max_entries
            self.ai.killer_moves.put(i, [MagicMock(), MagicMock()])

        # Table size should be bounded at max_entries
        self.assertLessEqual(len(self.ai.killer_moves), 10000)

    def test_transposition_table_evicts_old_entries(self) -> None:
        """Verify LRU eviction works correctly on transposition table."""
        # Fill the table
        for i in range(100000):
            self.ai.transposition_table.put(i, (float(i), i))

        # Add new entries to trigger eviction
        for i in range(100000, 100100):
            self.ai.transposition_table.put(i, (float(i), i))

        # Oldest entries should be evicted
        self.assertIsNone(self.ai.transposition_table.get(0))
        # Newest entries should exist
        self.assertIsNotNone(self.ai.transposition_table.get(100099))


class TestMinimaxAIIncrementalSearch(unittest.TestCase):
    """Tests for MinimaxAI incremental (mutable state) search."""

    def _create_game_state(self) -> GameState:
        """Create a minimal game state for testing."""
        return GameState(
            id="test-game",
            boardType=BoardType.SQUARE8,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={
                    "2,2": RingStack(
                        position=Position(x=2, y=2),
                        rings=[1],
                        stackHeight=1,
                        capHeight=1,
                        controllingPlayer=1,
                    ),
                    "5,5": RingStack(
                        position=Position(x=5, y=5),
                        rings=[2],
                        stackHeight=1,
                        capHeight=1,
                        controllingPlayer=2,
                    ),
                },
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            ),
            players=[
                Player(
                    id="player1",
                    username="Player1",
                    type="human",
                    playerNumber=1,
                    isReady=True,
                    timeRemaining=600000,
                    ringsInHand=17,
                    eliminatedRings=0,
                    territorySpaces=0,
                ),
                Player(
                    id="player2",
                    username="Player2",
                    type="ai",
                    playerNumber=2,
                    isReady=True,
                    timeRemaining=600000,
                    ringsInHand=17,
                    eliminatedRings=0,
                    territorySpaces=0,
                ),
            ],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=600000,
                increment=5000,
                type="fischer",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt="2023-01-01T00:00:00Z",
            lastMoveAt="2023-01-01T00:00:00Z",
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=6,
            territoryVictoryThreshold=20,
        )

    def test_use_incremental_search_defaults_to_true(self) -> None:
        """Verify use_incremental_search defaults to True."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=1, config=config)
        self.assertTrue(ai.use_incremental_search)

    def test_use_incremental_search_can_be_disabled(self) -> None:
        """Verify use_incremental_search can be set to False."""
        config = AIConfig(difficulty=5, use_incremental_search=False)
        ai = MinimaxAI(player_number=1, config=config)
        self.assertFalse(ai.use_incremental_search)

    def test_select_move_uses_incremental_when_enabled(self) -> None:
        """Verify select_move routes to incremental search when enabled."""
        config = AIConfig(difficulty=3, use_incremental_search=True)
        ai = MinimaxAI(player_number=1, config=config)

        mock_move = MagicMock()
        mock_move.type = "move_stack"

        with patch.object(
            ai,
            "get_valid_moves",
            return_value=[mock_move],
        ), patch.object(
            ai,
            "_select_move_incremental",
            return_value=mock_move,
        ) as mock_incremental, patch.object(
            ai,
            "_select_move_legacy",
            return_value=mock_move,
        ) as mock_legacy:
            ai.should_pick_random_move = MagicMock(return_value=False)
            ai.select_move(self._create_game_state())

        mock_incremental.assert_called_once()
        mock_legacy.assert_not_called()

    def test_select_move_uses_legacy_when_disabled(self) -> None:
        """Verify select_move routes to legacy search when disabled."""
        config = AIConfig(difficulty=3, use_incremental_search=False)
        ai = MinimaxAI(player_number=1, config=config)

        mock_move = MagicMock()
        mock_move.type = "move_stack"

        with patch.object(
            ai,
            "get_valid_moves",
            return_value=[mock_move],
        ), patch.object(
            ai,
            "_select_move_incremental",
            return_value=mock_move,
        ) as mock_incremental, patch.object(
            ai,
            "_select_move_legacy",
            return_value=mock_move,
        ) as mock_legacy:
            ai.should_pick_random_move = MagicMock(return_value=False)
            ai.select_move(self._create_game_state())

        mock_legacy.assert_called_once()
        mock_incremental.assert_not_called()

    def test_get_max_depth_returns_correct_values(self) -> None:
        """Verify _get_max_depth returns expected values for difficulty."""
        test_cases = [
            (1, 2),   # difficulty 1 -> depth 2
            (3, 2),   # difficulty 3 -> depth 2
            (4, 3),   # difficulty 4 -> depth 3
            (6, 3),   # difficulty 6 -> depth 3
            (7, 4),   # difficulty 7 -> depth 4
            (8, 4),   # difficulty 8 -> depth 4
            (9, 5),   # difficulty 9 -> depth 5
            (10, 5),  # difficulty 10 -> depth 5
        ]

        for difficulty, expected_depth in test_cases:
            config = AIConfig(difficulty=difficulty)
            ai = MinimaxAI(player_number=1, config=config)
            self.assertEqual(
                ai._get_max_depth(),
                expected_depth,
                f"Difficulty {difficulty} should produce depth {expected_depth}"
            )

    def test_moves_equal_same_moves(self) -> None:
        """Verify _moves_equal correctly identifies equal moves."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=1, config=config)

        move1 = MagicMock()
        move1.type = "move_stack"
        move1.to = Position(x=3, y=4)
        move1.from_pos = Position(x=1, y=2)

        move2 = MagicMock()
        move2.type = "move_stack"
        move2.to = Position(x=3, y=4)
        move2.from_pos = Position(x=1, y=2)

        self.assertTrue(ai._moves_equal(move1, move2))

    def test_moves_equal_different_moves(self) -> None:
        """Verify _moves_equal correctly identifies different moves."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=1, config=config)

        move1 = MagicMock()
        move1.type = "move_stack"
        move1.to = Position(x=3, y=4)
        move1.from_pos = Position(x=1, y=2)

        move2 = MagicMock()
        move2.type = "move_stack"
        move2.to = Position(x=5, y=6)
        move2.from_pos = Position(x=1, y=2)

        self.assertFalse(ai._moves_equal(move1, move2))

    def test_score_noisy_moves_sorts_by_priority(self) -> None:
        """Verify _score_noisy_moves sorts moves by priority."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=1, config=config)

        move_capture = MagicMock()
        move_capture.type = "overtaking_capture"

        move_territory = MagicMock()
        move_territory.type = "territory_claim"

        move_line = MagicMock()
        move_line.type = "line_formation"

        move_chain = MagicMock()
        move_chain.type = "chain_capture"

        # Pass in arbitrary order
        moves = [move_capture, move_territory, move_line, move_chain]
        scored = ai._score_noisy_moves(moves)

        # Should be sorted: territory (4), line (3), chain (2), capture (1)
        self.assertEqual(scored[0][0], 4)  # territory_claim
        self.assertEqual(scored[1][0], 3)  # line_formation
        self.assertEqual(scored[2][0], 2)  # chain_capture
        self.assertEqual(scored[3][0], 1)  # overtaking_capture

    def test_store_killer_move_limits_to_two(self) -> None:
        """Verify _store_killer_move keeps at most 2 killer moves."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=1, config=config)

        move1 = MagicMock()
        move2 = MagicMock()
        move3 = MagicMock()

        depth = 3
        ai._store_killer_move(move1, depth)
        ai._store_killer_move(move2, depth)
        ai._store_killer_move(move3, depth)

        killer_list = ai.killer_moves.get(depth)
        self.assertIsNotNone(killer_list)
        self.assertLessEqual(len(killer_list), 2)


class TestMinimaxAISearchEquivalence(unittest.TestCase):
    """Tests verifying incremental and legacy search produce similar results.

    These tests compare the behavior of both search paths to ensure the
    incremental search implementation is functionally equivalent.
    """

    def _create_simple_game_state(self) -> GameState:
        """Create a simple game state for equivalence testing."""
        return GameState(
            id="test-game",
            boardType=BoardType.SQUARE8,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={
                    "3,3": RingStack(
                        position=Position(x=3, y=3),
                        rings=[1],
                        stackHeight=1,
                        capHeight=1,
                        controllingPlayer=1,
                    ),
                },
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            ),
            players=[
                Player(
                    id="player1",
                    username="Player1",
                    type="ai",
                    playerNumber=1,
                    isReady=True,
                    timeRemaining=600000,
                    ringsInHand=17,
                    eliminatedRings=0,
                    territorySpaces=0,
                ),
                Player(
                    id="player2",
                    username="Player2",
                    type="ai",
                    playerNumber=2,
                    isReady=True,
                    timeRemaining=600000,
                    ringsInHand=18,
                    eliminatedRings=0,
                    territorySpaces=0,
                ),
            ],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=600000,
                increment=5000,
                type="fischer",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt="2023-01-01T00:00:00Z",
            lastMoveAt="2023-01-01T00:00:00Z",
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=6,
            territoryVictoryThreshold=20,
        )

    def test_both_search_paths_return_valid_move(self) -> None:
        """Verify both search paths return a valid move (not None)."""
        game_state = self._create_simple_game_state()

        # Test incremental search
        config_inc = AIConfig(
            difficulty=2,
            use_incremental_search=True,
            think_time=100,
        )
        ai_inc = MinimaxAI(player_number=1, config=config_inc)

        # Test legacy search
        config_leg = AIConfig(
            difficulty=2,
            use_incremental_search=False,
            think_time=100,
        )
        ai_leg = MinimaxAI(player_number=1, config=config_leg)

        # Mock get_valid_moves to return predictable moves
        mock_move1 = MagicMock()
        mock_move1.type = "move_stack"
        mock_move1.to = Position(x=4, y=4)
        mock_move1.from_pos = Position(x=3, y=3)

        with patch.object(
            ai_inc, "get_valid_moves", return_value=[mock_move1]
        ), patch.object(
            ai_inc.rules_engine, "apply_move", return_value=game_state
        ):
            ai_inc.evaluate_position = MagicMock(return_value=10.0)
            move_inc = ai_inc.select_move(game_state)

        with patch.object(
            ai_leg, "get_valid_moves", return_value=[mock_move1]
        ), patch.object(
            ai_leg.rules_engine, "apply_move", return_value=game_state
        ):
            ai_leg.evaluate_position = MagicMock(return_value=10.0)
            move_leg = ai_leg.select_move(game_state)

        self.assertIsNotNone(move_inc)
        self.assertIsNotNone(move_leg)

    def test_incremental_search_uses_mutable_state(self) -> None:
        """Verify incremental search creates and uses MutableGameState."""
        from app.rules.mutable_state import MutableGameState

        config = AIConfig(
            difficulty=2,
            use_incremental_search=True,
            think_time=100,
        )
        ai = MinimaxAI(player_number=1, config=config)
        game_state = self._create_simple_game_state()

        mock_move = MagicMock()
        mock_move.type = "move_stack"
        mock_move.to = Position(x=4, y=4)
        mock_move.from_pos = Position(x=3, y=3)

        # Patch MutableGameState.from_immutable to track calls
        original_from_immutable = MutableGameState.from_immutable
        call_count = [0]

        def tracked_from_immutable(state):
            call_count[0] += 1
            return original_from_immutable(state)

        with patch.object(
            ai, "get_valid_moves", return_value=[mock_move]
        ), patch.object(
            ai.rules_engine, "apply_move", return_value=game_state
        ), patch.object(
            MutableGameState,
            "from_immutable",
            side_effect=tracked_from_immutable,
        ):
            ai.evaluate_position = MagicMock(return_value=10.0)
            ai.should_pick_random_move = MagicMock(return_value=False)
            ai.select_move(game_state)

        # MutableGameState.from_immutable should be called at least once
        self.assertGreaterEqual(call_count[0], 1)


if __name__ == "__main__":
    unittest.main()
