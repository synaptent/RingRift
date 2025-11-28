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
)

class TestHeuristicAI(unittest.TestCase):
    def setUp(self):
        self.config = AIConfig(difficulty=5)
        self.ai = HeuristicAI(player_number=1, config=self.config)
        
        # Create a mock game state
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
                collapsedSpaces={},
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
        board.collapsed_spaces.clear()

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
        board.collapsed_spaces.clear()

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

    def test_evaluate_lps_action_advantage_rewards_unique_actions(self):
        """
        LPS action advantage should reward being one of the few players
        with real actions remaining in a 3+ player game.
        """
        board = self.game_state.board
        board.stacks.clear()
        board.collapsed_spaces.clear()

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
        board.collapsed_spaces.clear()

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


if __name__ == '__main__':
    unittest.main()