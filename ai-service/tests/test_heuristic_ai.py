import unittest
from datetime import datetime

from app.ai.heuristic_ai import HeuristicAI
from app.models import (
    GameState,
    AIConfig,
    BoardType,
    GamePhase,
    GameStatus,
    TimeControl,
    BoardState,
    Position,
    RingStack,
    MarkerInfo,
    Player,
    Move,
    MoveType,
)


def _make_mock_move(player: int, x: int, y: int) -> Move:
    """Create a valid Move object for testing with all required fields."""
    return Move(
        id=f"m-{x}-{y}",
        type=MoveType.PLACE_RING,
        player=player,
        to=Position(x=x, y=y),
        timestamp=datetime.now(),
        thinkTime=0,
        moveNumber=1,
    )


class TestHeuristicAI(unittest.TestCase):
    def setUp(self):
        self.config = AIConfig(difficulty=5)
        self.ai = HeuristicAI(player_number=1, config=self.config)

        # Create a mock game state
        self.game_state = GameState(
            id="test-game",
            boardType=BoardType.SQUARE8,
            rngSeed=None,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={},
                markers={},
                collapsed_spaces={},
                eliminatedRings={},
            ),
            players=[
                Player(
                    id="p1",
                    username="P1",
                    type="human",
                    playerNumber=1,
                    isReady=True,
                    timeRemaining=600,
                    ringsInHand=10,
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
                    ringsInHand=10,
                    eliminatedRings=0,
                    territorySpaces=0,
                    aiDifficulty=None,
                ),
            ],
            currentPhase=GamePhase.MOVEMENT,
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
            victoryThreshold=19,
            territoryVictoryThreshold=33,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
            lpsRoundIndex=0,
            lpsCurrentRoundActorMask={},
            lpsExclusivePlayerForCompletedRound=None,
        )

    def test_evaluate_line_connectivity(self):
        # Setup: Two markers with a gap of 1
        self.game_state.board.markers["3,3"] = MarkerInfo(
            player=1,
            position=Position(x=3, y=3),
            type="regular",
        )
        self.game_state.board.markers["5,3"] = MarkerInfo(
            player=1,
            position=Position(x=5, y=3),
            type="regular",
        )

        score = self.ai._evaluate_line_connectivity(self.game_state)

        # Should have some positive score
        self.assertGreater(score, 0)

    def test_evaluate_territory_safety(self):
        # Setup: My marker near opponent stack
        self.game_state.board.markers["3,3"] = MarkerInfo(
            player=1,
            position=Position(x=3, y=3),
            type="regular",
        )
        self.game_state.board.stacks["4,3"] = RingStack(
            position=Position(x=4, y=3),
            rings=[2],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=2,
        )

        score = self.ai._evaluate_territory_safety(self.game_state)

        # Should have negative score (penalty)
        self.assertLess(score, 0)

    def test_evaluate_stack_mobility(self):
        # Setup: My stack surrounded by collapsed spaces
        self.game_state.board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        # Block all 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                self.game_state.board.collapsed_spaces[f"{3+dx},{3+dy}"] = 1

        score = self.ai._evaluate_stack_mobility(self.game_state)

        # Should have negative score (penalty for being blocked)
        self.assertLess(score, 0)

    def test_evaluate_opponent_victory_threat_when_ahead(self):
        """Opponent victory threat should be zero when we are ahead."""
        # Configure thresholds and players such that player 1 is
        # closer to victory.
        self.game_state.victory_threshold = 10
        self.game_state.territory_victory_threshold = 33

        p1, p2 = self.game_state.players
        p1.eliminated_rings = 5
        p1.territory_spaces = 10
        p2.eliminated_rings = 1
        p2.territory_spaces = 5

        score = self.ai._evaluate_opponent_victory_threat(self.game_state)

        # When we are ahead of all opponents, the relative threat is
        # clamped to zero.
        self.assertEqual(score, 0.0)

    def test_evaluate_opponent_victory_threat_when_behind(self):
        """
        Opponent victory threat should be a negative penalty when an
        opponent leads.
        """
        self.game_state.victory_threshold = 10
        self.game_state.territory_victory_threshold = 33

        p1, p2 = self.game_state.players
        p1.eliminated_rings = 0
        p1.territory_spaces = 0
        p2.eliminated_rings = 8
        p2.territory_spaces = 30

        score = self.ai._evaluate_opponent_victory_threat(self.game_state)

        self.assertLess(score, 0.0)

    def test_evaluate_forced_elimination_risk_penalizes_many_stacks_few_actions(
        self,
    ):
        """
        Forced-elimination risk should be more negative when we have many
        stacks but almost no real actions compared to an open-board
        configuration.
        """
        board = self.game_state.board
        board.stacks.clear()
        board.collapsed_spaces = {}

        # High-risk configuration: two stacks, but every other space is
        # collapsed.
        board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        board.stacks["4,4"] = RingStack(
            position=Position(x=4, y=4),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        for x in range(board.size):
            for y in range(board.size):
                key = f"{x},{y}"
                if key not in board.stacks:
                    board.collapsed_spaces[key] = 1

        high_risk = self.ai._evaluate_forced_elimination_risk(self.game_state)

        # Low-risk configuration: same stacks, but open board (no
        # collapsed spaces).
        board.stacks.clear()
        board.collapsed_spaces = {}
        board.stacks.clear()
        board.collapsed_spaces = {}

        board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        board.stacks["4,4"] = RingStack(
            position=Position(x=4, y=4),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        low_risk = self.ai._evaluate_forced_elimination_risk(self.game_state)

        self.assertLess(high_risk, 0.0)
        self.assertLess(high_risk, low_risk)

    def test_evaluate_forced_elimination_risk_zero_when_no_stacks(self):
        """Forced-elimination risk should be zero when we control no stacks."""
        self.game_state.board.stacks.clear()
        score = self.ai._evaluate_forced_elimination_risk(self.game_state)
        self.assertEqual(score, 0.0)

    def test_victory_proximity_increases_with_territory_progress(self):
        """
        Victory proximity should increase as we get closer to the
        territory victory threshold.
        """
        # Baseline: no territory spaces for either player.
        self.game_state.victory_threshold = 19
        self.game_state.territory_victory_threshold = 33
        p1, p2 = self.game_state.players
        p1.eliminated_rings = 0
        p1.territory_spaces = 0
        p2.eliminated_rings = 0
        p2.territory_spaces = 0

        base = self.ai._evaluate_victory_proximity(self.game_state)

        # Near-victory territory configuration: Player 1 at 32/33 spaces.
        p1.territory_spaces = 32
        closer = self.ai._evaluate_victory_proximity(self.game_state)

        self.assertGreater(closer, base)

    def test_victory_proximity_increases_with_elimination_progress(self):
        """
        Victory proximity should increase as we get closer to the
        ring-elimination threshold.
        """
        self.game_state.victory_threshold = 10
        self.game_state.territory_victory_threshold = 100

        p1, p2 = self.game_state.players
        p1.eliminated_rings = 0
        p1.territory_spaces = 0
        p2.eliminated_rings = 0
        p2.territory_spaces = 0

        base = self.ai._evaluate_victory_proximity(self.game_state)

        # Move Player 1 to a near-elimination state (9/10 rings).
        p1.eliminated_rings = 9
        nearer = self.ai._evaluate_victory_proximity(self.game_state)

        self.assertGreater(nearer, base)

    def test_evaluate_lps_action_advantage_rewards_unique_actions(self):
        """
        LPS action advantage should reward being one of the few players
        with real actions remaining in a 3+ player game.
        """
        board = self.game_state.board
        board.stacks.clear()
        board.collapsed_spaces = {}

        # Only player 1 has a mobile stack plus rings in hand.
        board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        players = [
            Player(
                id="p1",
                username="P1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=1,
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
                ringsInHand=0,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
            Player(
                id="p3",
                username="P3",
                type="human",
                playerNumber=3,
                isReady=True,
                timeRemaining=600,
                ringsInHand=0,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
        ]
        self.game_state.players = players

        score = self.ai._evaluate_lps_action_advantage(self.game_state)

        self.assertGreater(score, 0.0)

    def test_evaluate_lps_action_advantage_penalizes_when_self_has_no_actions(
        self,
    ):
        """
        LPS action advantage should penalise us when we have no actions
        but at least one opponent still does.
        """
        board = self.game_state.board
        board.stacks.clear()
        board.collapsed_spaces = {}

        # Only player 2 has a mobile stack; player 1 has no stacks and no rings.
        board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[2],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=2,
        )

        players = [
            Player(
                id="p1",
                username="P1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=0,
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
                ringsInHand=1,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
            Player(
                id="p3",
                username="P3",
                type="human",
                playerNumber=3,
                isReady=True,
                timeRemaining=600,
                ringsInHand=0,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
        ]
        self.game_state.players = players

        score = self.ai._evaluate_lps_action_advantage(self.game_state)

        self.assertLess(score, 0.0)

    def test_evaluate_multi_leader_threat_zero_for_two_player_games(self):
        """Multi-leader threat is disabled in 2-player games."""
        score = self.ai._evaluate_multi_leader_threat(self.game_state)
        self.assertEqual(score, 0.0)

    def test_evaluate_multi_leader_threat_penalizes_large_leader_gap(self):
        """
        Multi-leader threat should assign a larger penalty when one
        opponent is far ahead of other opponents than when the gap is
        small.
        """
        self.game_state.victory_threshold = 10
        self.game_state.territory_victory_threshold = 100

        # Small leader gap between opponents.
        small_gap_players = [
            Player(
                id="p1",
                username="P1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=0,
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
                ringsInHand=0,
                eliminatedRings=5,
                territorySpaces=0,
                aiDifficulty=None,
            ),
            Player(
                id="p3",
                username="P3",
                type="human",
                playerNumber=3,
                isReady=True,
                timeRemaining=600,
                ringsInHand=0,
                eliminatedRings=4,
                territorySpaces=0,
                aiDifficulty=None,
            ),
        ]
        self.game_state.players = small_gap_players
        small_gap = self.ai._evaluate_multi_leader_threat(self.game_state)

        # Large leader gap between top opponent and the rest.
        large_gap_players = [
            Player(
                id="p1",
                username="P1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=0,
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
                ringsInHand=0,
                eliminatedRings=9,
                territorySpaces=0,
                aiDifficulty=None,
            ),
            Player(
                id="p3",
                username="P3",
                type="human",
                playerNumber=3,
                isReady=True,
                timeRemaining=600,
                ringsInHand=0,
                eliminatedRings=1,
                territorySpaces=0,
                aiDifficulty=None,
            ),
        ]
        self.game_state.players = large_gap_players
        large_gap = self.ai._evaluate_multi_leader_threat(self.game_state)

        # Both configurations should be non-positive (penalising leader
        # threat), and the large-gap scenario should be strictly worse.
        self.assertLessEqual(small_gap, 0.0)
        self.assertLess(large_gap, small_gap)


class TestHeuristicAIConfig(unittest.TestCase):
    """Test HeuristicAI configuration options."""

    def test_use_incremental_search_defaults_to_true(self):
        """Verify use_incremental_search defaults to True."""
        config = AIConfig(difficulty=5, randomness=None, rng_seed=None)
        ai = HeuristicAI(player_number=1, config=config)
        self.assertTrue(ai.use_incremental_search)

    def test_use_incremental_search_can_be_disabled(self):
        """Verify use_incremental_search can be set to False."""
        config = AIConfig(
            difficulty=5,
            randomness=None,
            rng_seed=None,
            use_incremental_search=False,
        )
        ai = HeuristicAI(player_number=1, config=config)
        self.assertFalse(ai.use_incremental_search)

    def test_use_incremental_search_can_be_enabled_explicitly(self):
        """Verify use_incremental_search can be explicitly set to True."""
        config = AIConfig(
            difficulty=5,
            randomness=None,
            rng_seed=None,
            use_incremental_search=True,
        )
        ai = HeuristicAI(player_number=1, config=config)
        self.assertTrue(ai.use_incremental_search)

    def test_heuristic_eval_mode_gates_tier2_features(self):
        """`heuristic_eval_mode=\"light\"` should zero Tier-2 features."""
        # Shared minimal game state (reuse the structure from TestHeuristicAI)
        game_state = GameState(
            id="config-test-game",
            boardType=BoardType.SQUARE8,
            rngSeed=None,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={},
                markers={},
                collapsed_spaces={},
                eliminatedRings={},
            ),
            players=[
                Player(
                    id="p1",
                    username="P1",
                    type="human",
                    playerNumber=1,
                    isReady=True,
                    timeRemaining=600,
                    ringsInHand=10,
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
                    ringsInHand=10,
                    eliminatedRings=0,
                    territorySpaces=0,
                    aiDifficulty=None,
                ),
            ],
            currentPhase=GamePhase.MOVEMENT,
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
            victoryThreshold=19,
            territoryVictoryThreshold=33,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
            lpsRoundIndex=0,
            lpsCurrentRoundActorMask={},
            lpsExclusivePlayerForCompletedRound=None,
        )

        # Full-mode AI: Tier-2 evaluators are called and their values surface.
        full_config = AIConfig(
            difficulty=5,
            randomness=None,
            rng_seed=None,
            heuristic_eval_mode="full",
        )
        full_ai = HeuristicAI(player_number=1, config=full_config)

        # Light-mode AI: Tier-2 evaluators must not be called and scores are 0.0.
        light_config = AIConfig(
            difficulty=5,
            randomness=None,
            rng_seed=None,
            heuristic_eval_mode="light",
        )
        light_ai = HeuristicAI(player_number=1, config=light_config)

        tier2_names = [
            "_evaluate_vulnerability",
            "_evaluate_overtake_potential",
            "_evaluate_territory_closure",
            "_evaluate_line_connectivity",
            "_evaluate_territory_safety",
            "_evaluate_forced_elimination_risk",
            "_evaluate_lps_action_advantage",
        ]

        # Patch Tier-2 evaluators on the full-mode AI to return a sentinel value.
        def _tier2_stub(self, _game_state):
            return 1.0

        for name in tier2_names:
            setattr(full_ai, name, _tier2_stub.__get__(full_ai, HeuristicAI))

        # Patch Tier-2 evaluators on the light-mode AI to raise if ever called.
        def _tier2_should_not_be_called(self, _game_state):
            raise AssertionError(f"{self.__class__.__name__}.{_game_state} Tier-2 evaluator should not be called in light mode")

        for name in tier2_names:
            setattr(
                light_ai,
                name,
                _tier2_should_not_be_called.__get__(light_ai, HeuristicAI),
            )

        full_scores = full_ai._compute_component_scores(game_state)
        light_scores = light_ai._compute_component_scores(game_state)

        # In full mode, Tier-2 scores reflect the stubbed value.
        for name in tier2_names:
            key = name.replace("_evaluate_", "")
            self.assertIn(key, full_scores)
            self.assertEqual(full_scores[key], 1.0)

        # In light mode, Tier-2 scores are forced to 0.0 and evaluators are never called.
        for name in tier2_names:
            key = name.replace("_evaluate_", "")
            self.assertIn(key, light_scores)
            self.assertEqual(light_scores[key], 0.0)


class TestHeuristicAIMoveSampling(unittest.TestCase):
    """Test HeuristicAI move sampling for training."""

    def test_sample_moves_returns_all_when_no_limit(self):
        """When training_move_sample_limit is None, all moves are returned."""
        config = AIConfig(
            difficulty=5,
            randomness=None,
            rng_seed=None,
            training_move_sample_limit=None,
        )
        ai = HeuristicAI(player_number=1, config=config)

        # Create mock moves
        moves = [_make_mock_move(player=1, x=i, y=0) for i in range(20)]

        result = ai._sample_moves_for_training(moves)
        self.assertEqual(len(result), 20)
        self.assertEqual(result, moves)

    def test_sample_moves_returns_all_when_under_limit(self):
        """When moves are under the limit, all are returned."""
        config = AIConfig(
            difficulty=5,
            randomness=None,
            rng_seed=None,
            training_move_sample_limit=100,
        )
        ai = HeuristicAI(player_number=1, config=config)

        moves = [_make_mock_move(player=1, x=i, y=0) for i in range(50)]

        result = ai._sample_moves_for_training(moves)
        self.assertEqual(len(result), 50)
        self.assertEqual(result, moves)

    def test_sample_moves_limits_when_over(self):
        """When moves exceed the limit, sample is taken."""
        config = AIConfig(
            difficulty=5,
            randomness=None,
            rng_seed=42,  # Seeded for determinism
            training_move_sample_limit=10,
        )
        ai = HeuristicAI(player_number=1, config=config)

        moves = [_make_mock_move(player=1, x=i, y=0) for i in range(100)]

        result = ai._sample_moves_for_training(moves)
        self.assertEqual(len(result), 10)
        # All sampled moves should be from the original list
        for move in result:
            self.assertIn(move, moves)

    def test_sample_moves_is_deterministic_with_seed(self):
        """With same seed and move_count, sampling is deterministic."""
        config = AIConfig(
            difficulty=5,
            randomness=None,
            rng_seed=12345,
            training_move_sample_limit=5,
        )

        moves = [_make_mock_move(player=1, x=i, y=0) for i in range(50)]

        # Create two AIs with same config
        ai1 = HeuristicAI(player_number=1, config=config)
        ai2 = HeuristicAI(player_number=1, config=config)

        result1 = ai1._sample_moves_for_training(moves)
        result2 = ai2._sample_moves_for_training(moves)

        # Should get identical samples
        self.assertEqual(result1, result2)

    def test_sample_moves_determinism_changes_with_move_count(self):
        """Sampling should vary based on move_count for diversity."""
        config = AIConfig(
            difficulty=5,
            randomness=None,
            rng_seed=42,
            training_move_sample_limit=5,
        )
        ai = HeuristicAI(player_number=1, config=config)

        moves = [_make_mock_move(player=1, x=i, y=0) for i in range(50)]

        # Sample at move_count=0
        ai.move_count = 0
        result_a = ai._sample_moves_for_training(moves)

        # Sample at move_count=1 (should differ due to seeding)
        ai.move_count = 1
        result_b = ai._sample_moves_for_training(moves)

        # Results should be different (high probability with 50 moves, 5 sample)
        # Note: There's a tiny chance they could be equal by coincidence,
        # but with these numbers it's extremely unlikely
        self.assertNotEqual(result_a, result_b)

    def test_sample_moves_returns_all_when_limit_is_zero_or_negative(self):
        """When limit is 0 or negative, all moves are returned (disabled)."""
        for limit in [0, -1, -100]:
            config = AIConfig(
                difficulty=5,
                randomness=None,
                rng_seed=None,
                training_move_sample_limit=limit,
            )
            ai = HeuristicAI(player_number=1, config=config)

            moves = [_make_mock_move(player=1, x=i, y=0) for i in range(20)]

            result = ai._sample_moves_for_training(moves)
            self.assertEqual(len(result), 20, f"Failed for limit={limit}")


class TestHeuristicAISwapEvaluation(unittest.TestCase):
    """Test HeuristicAI swap (pie rule) opening evaluation."""

    def setUp(self):
        self.config = AIConfig(difficulty=5)
        self.ai = HeuristicAI(player_number=2, config=self.config)  # P2 evaluates swap

        # Create a mock game state for 2-player game
        self.game_state = GameState(
            id="swap-test-game",
            boardType=BoardType.SQUARE8,
            rngSeed=None,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={},
                markers={},
                collapsed_spaces={},
                eliminatedRings={},
            ),
            players=[
                Player(
                    id="p1",
                    username="P1",
                    type="human",
                    playerNumber=1,
                    isReady=True,
                    timeRemaining=600,
                    ringsInHand=10,
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
                    ringsInHand=10,
                    eliminatedRings=0,
                    territorySpaces=0,
                    aiDifficulty=None,
                ),
            ],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=2,
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
            victoryThreshold=19,
            territoryVictoryThreshold=33,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
            lpsRoundIndex=0,
            lpsCurrentRoundActorMask={},
            lpsExclusivePlayerForCompletedRound=None,
        )

    def test_swap_bonus_zero_for_three_plus_players(self):
        """Swap bonus should be 0.0 for 3+ player games (swap is 2P only)."""
        # Add a third player
        self.game_state.players.append(
            Player(
                id="p3",
                username="P3",
                type="human",
                playerNumber=3,
                isReady=True,
                timeRemaining=600,
                ringsInHand=10,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            )
        )

        # Even with P1 having a center stack, bonus should be 0
        self.game_state.board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        bonus = self.ai.evaluate_swap_opening_bonus(self.game_state)
        self.assertEqual(bonus, 0.0)

    def test_swap_bonus_zero_when_p1_has_no_stacks(self):
        """Swap bonus should be 0.0 when P1 has no stacks."""
        # No stacks on board
        bonus = self.ai.evaluate_swap_opening_bonus(self.game_state)
        self.assertEqual(bonus, 0.0)

    def test_swap_bonus_positive_for_center_stack(self):
        """Swap bonus should be positive when P1 has a center stack."""
        # Place P1 stack in center (3,3 or 4,4 on 8x8 board)
        self.game_state.board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        bonus = self.ai.evaluate_swap_opening_bonus(self.game_state)
        self.assertGreater(bonus, 0.0)

    def test_swap_bonus_higher_for_taller_stacks(self):
        """Swap bonus should be higher for taller P1 stacks."""
        # Height-1 stack at center
        self.game_state.board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        bonus_height1 = self.ai.evaluate_swap_opening_bonus(self.game_state)

        # Height-3 stack at center
        self.game_state.board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1, 1, 1],
            stackHeight=3,
            capHeight=3,
            controllingPlayer=1,
        )
        bonus_height3 = self.ai.evaluate_swap_opening_bonus(self.game_state)

        self.assertGreater(bonus_height3, bonus_height1)

    def test_swap_bonus_center_vs_edge(self):
        """Center position should provide higher swap bonus than edge."""
        # Edge stack (0,0)
        self.game_state.board.stacks["0,0"] = RingStack(
            position=Position(x=0, y=0),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        bonus_edge = self.ai.evaluate_swap_opening_bonus(self.game_state)

        # Clear and add center stack (3,3)
        self.game_state.board.stacks.clear()
        self.game_state.board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        bonus_center = self.ai.evaluate_swap_opening_bonus(self.game_state)

        # Center should be significantly higher (center weight = 15, adj = 3)
        self.assertGreater(bonus_center, bonus_edge)


class TestOpeningPositionClassifier(unittest.TestCase):
    """Test HeuristicAI Opening Position Classifier (v1.3 enhanced swap evaluation)."""

    def setUp(self):
        self.config = AIConfig(difficulty=5)
        self.ai = HeuristicAI(player_number=2, config=self.config)

        # Create a mock game state for 2-player game
        self.game_state = GameState(
            id="classifier-test-game",
            boardType=BoardType.SQUARE8,
            rngSeed=None,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={},
                markers={},
                collapsed_spaces={},
                eliminatedRings={},
            ),
            players=[
                Player(
                    id="p1",
                    username="P1",
                    type="human",
                    playerNumber=1,
                    isReady=True,
                    timeRemaining=600,
                    ringsInHand=10,
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
                    ringsInHand=10,
                    eliminatedRings=0,
                    territorySpaces=0,
                    aiDifficulty=None,
                ),
            ],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=2,
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
            victoryThreshold=19,
            territoryVictoryThreshold=33,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
            lpsRoundIndex=0,
            lpsCurrentRoundActorMask={},
            lpsExclusivePlayerForCompletedRound=None,
        )

    # --- Position helper tests ---

    def test_is_corner_position(self):
        """Corner positions should be correctly identified."""
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for x, y in corners:
            pos = Position(x=x, y=y)
            self.assertTrue(
                self.ai._is_corner_position(pos, self.game_state),
                f"({x},{y}) should be a corner"
            )

        # Non-corners
        non_corners = [(1, 0), (0, 1), (3, 3), (4, 4)]
        for x, y in non_corners:
            pos = Position(x=x, y=y)
            self.assertFalse(
                self.ai._is_corner_position(pos, self.game_state),
                f"({x},{y}) should not be a corner"
            )

    def test_is_edge_position(self):
        """Edge (non-corner) positions should be correctly identified."""
        edges = [(0, 3), (3, 0), (7, 4), (5, 7)]
        for x, y in edges:
            pos = Position(x=x, y=y)
            self.assertTrue(
                self.ai._is_edge_position(pos, self.game_state),
                f"({x},{y}) should be an edge"
            )

        # Corners are not "edge" positions for our purposes
        pos = Position(x=0, y=0)
        self.assertFalse(self.ai._is_edge_position(pos, self.game_state))

        # Center is not edge
        pos = Position(x=3, y=3)
        self.assertFalse(self.ai._is_edge_position(pos, self.game_state))

    def test_is_strategic_diagonal_position(self):
        """Strategic diagonal positions (one diagonal step from center)."""
        # On an 8x8 board, center positions are 3,3 / 3,4 / 4,3 / 4,4
        # Diagonal from (3,3) includes (2,2), (2,4), (4,2)
        # Diagonal from (4,4) includes (3,5), (5,3), (5,5)
        diagonals = [(2, 2), (2, 5), (5, 2), (5, 5)]
        for x, y in diagonals:
            pos = Position(x=x, y=y)
            self.assertTrue(
                self.ai._is_strategic_diagonal_position(pos, self.game_state),
                f"({x},{y}) should be a strategic diagonal"
            )

        # Center positions themselves are not "diagonal" positions
        pos = Position(x=3, y=3)
        self.assertFalse(
            self.ai._is_strategic_diagonal_position(pos, self.game_state)
        )

    # --- Opening strength classifier tests ---

    def test_opening_strength_center_highest(self):
        """Center positions should have highest opening strength."""
        center_pos = Position(x=3, y=3)
        strength = self.ai.compute_opening_strength(center_pos, self.game_state)
        self.assertGreaterEqual(strength, 0.9)
        self.assertLessEqual(strength, 1.0)

    def test_opening_strength_corner_lowest(self):
        """Corner positions should have lowest opening strength."""
        corner_pos = Position(x=0, y=0)
        strength = self.ai.compute_opening_strength(corner_pos, self.game_state)
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 0.2)

    def test_opening_strength_ordering(self):
        """Opening strength should follow: center > adjacent > interior > edge > corner.

        Note: "Strategic diagonal" positions are a subset of "adjacent to center"
        because diagonal adjacency is included in adjacency. The ordering tested:
        - Center (3,3): 0.95
        - Adjacent to center (3,2): 0.75
        - Interior position not adjacent to center (1,1): 0.45
        - Edge position (0,3): 0.35
        - Corner (0,0): 0.15
        """
        center = self.ai.compute_opening_strength(Position(x=3, y=3), self.game_state)
        adjacent = self.ai.compute_opening_strength(Position(x=3, y=2), self.game_state)
        interior = self.ai.compute_opening_strength(Position(x=1, y=1), self.game_state)
        edge = self.ai.compute_opening_strength(Position(x=0, y=3), self.game_state)
        corner = self.ai.compute_opening_strength(Position(x=0, y=0), self.game_state)

        self.assertGreater(center, adjacent)
        self.assertGreater(adjacent, interior)
        self.assertGreater(interior, edge)
        self.assertGreater(edge, corner)

    def test_opening_strength_range(self):
        """All opening strengths should be in [0, 1] range."""
        test_positions = [
            (0, 0), (0, 4), (3, 3), (4, 4), (7, 7), (2, 2), (5, 5)
        ]
        for x, y in test_positions:
            pos = Position(x=x, y=y)
            strength = self.ai.compute_opening_strength(pos, self.game_state)
            self.assertGreaterEqual(strength, 0.0, f"Strength at ({x},{y}) < 0")
            self.assertLessEqual(strength, 1.0, f"Strength at ({x},{y}) > 1")

    # --- Enhanced swap evaluation tests ---

    def test_classifier_swap_zero_for_three_plus_players(self):
        """Classifier swap should be 0.0 for 3+ player games."""
        self.game_state.players.append(
            Player(
                id="p3",
                username="P3",
                type="human",
                playerNumber=3,
                isReady=True,
                timeRemaining=600,
                ringsInHand=10,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            )
        )

        self.game_state.board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        score = self.ai.evaluate_swap_with_classifier(self.game_state)
        self.assertEqual(score, 0.0)

    def test_classifier_swap_zero_when_no_stacks(self):
        """Classifier swap should be 0.0 when P1 has no stacks."""
        score = self.ai.evaluate_swap_with_classifier(self.game_state)
        self.assertEqual(score, 0.0)

    def test_classifier_swap_high_for_center_stack(self):
        """Classifier swap should be high for center P1 stack."""
        self.game_state.board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        score = self.ai.evaluate_swap_with_classifier(self.game_state)
        # With center stack: strength ~0.95 * 20 = 19 + height bonus
        self.assertGreater(score, 15.0)

    def test_classifier_swap_low_for_corner_stack(self):
        """Classifier swap should be low (possibly negative) for corner P1 stack."""
        self.game_state.board.stacks["0,0"] = RingStack(
            position=Position(x=0, y=0),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        score = self.ai.evaluate_swap_with_classifier(self.game_state)
        # Corner has penalty of 8.0, low strength (~0.15 * 20 = 3)
        # Net should be much lower than center
        self.assertLess(score, 5.0)

    def test_classifier_swap_center_vs_corner_difference(self):
        """Center position should have much higher swap value than corner."""
        # Corner stack
        self.game_state.board.stacks["0,0"] = RingStack(
            position=Position(x=0, y=0),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        score_corner = self.ai.evaluate_swap_with_classifier(self.game_state)

        # Clear and add center stack
        self.game_state.board.stacks.clear()
        self.game_state.board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        score_center = self.ai.evaluate_swap_with_classifier(self.game_state)

        # Center should be significantly higher
        self.assertGreater(score_center - score_corner, 10.0)


if __name__ == "__main__":
    unittest.main()
