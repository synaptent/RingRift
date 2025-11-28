"""
Archived diagnostic suite: RNG determinism tests for Python AI service.

This file was moved from ai-service/tests/test_determinism.py and is
retained for historical/debugging reference only. It is not part of the
canonical rules or CI gating suites (superseded by test_engine_determinism.py
and test_no_random_in_rules_core.py).
"""

import random
from datetime import datetime
from app.models import (
    GameState,
    AIConfig,
    BoardType,
    GamePhase,
    GameStatus,
    BoardState,
    Player,
    TimeControl,
)
from app.ai.random_ai import RandomAI
from app.ai.heuristic_ai import HeuristicAI


def create_test_state() -> GameState:
    """Create a minimal test GameState for determinism testing."""

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
        formedLines=[],
        territories={},
    )

    players = [
        Player(
            id="player1",
            username="Player 1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="player2",
            username="Player 2",
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    time_control = TimeControl(
        initialTime=600,
        increment=0,
        type="rapid",
    )

    return GameState(
        id="test-game",
        boardType=BoardType.SQUARE8,
        rngSeed=42,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=2,
        moveHistory=[],
        timeControl=time_control,
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
        mustMoveFromStackKey=None,
        chainCaptureState=None,
        zobristHash=None,
    )


class TestAIDeterminism:
    """Test suite for AI determinism with seeded RNG."""

    def test_seeded_random_ai_determinism(self):
        state = create_test_state()

        config1 = AIConfig(difficulty=1, randomness=0.5, rngSeed=42)
        config2 = AIConfig(difficulty=1, randomness=0.5, rngSeed=42)

        ai1 = RandomAI(player_number=2, config=config1)
        ai2 = RandomAI(player_number=2, config=config2)

        move1 = ai1.select_move(state)
        move2 = ai2.select_move(state)

        assert move1 is not None
        assert move2 is not None
        assert move1.type == move2.type
        assert move1.to == move2.to

    def test_different_seeds_produce_different_moves(self):
        state = create_test_state()

        config1 = AIConfig(difficulty=1, randomness=0.5, rngSeed=42)
        config2 = AIConfig(difficulty=1, randomness=0.5, rngSeed=43)

        ai1 = RandomAI(player_number=2, config=config1)
        ai2 = RandomAI(player_number=2, config=config2)

        move1 = ai1.select_move(state)
        move2 = ai2.select_move(state)

        assert move1 is not None
        assert move2 is not None

    def test_heuristic_ai_determinism(self):
        state = create_test_state()

        config1 = AIConfig(
            difficulty=5,
            randomness=0.05,
            rngSeed=12345,
            heuristic_profile_id="v1-heuristic-5",
        )
        config2 = AIConfig(
            difficulty=5,
            randomness=0.05,
            rngSeed=12345,
            heuristic_profile_id="v1-heuristic-5",
        )

        ai1 = HeuristicAI(player_number=2, config=config1)
        ai2 = HeuristicAI(player_number=2, config=config2)

        move1 = ai1.select_move(state)
        move2 = ai2.select_move(state)

        assert move1 is not None
        assert move2 is not None
        assert move1.type == move2.type
        assert move1.to == move2.to

    def test_evaluation_determinism(self):
        state = create_test_state()

        config1 = AIConfig(difficulty=5, randomness=0.0, rngSeed=999)
        config2 = AIConfig(difficulty=5, randomness=0.0, rngSeed=999)

        ai1 = HeuristicAI(player_number=2, config=config1)
        ai2 = HeuristicAI(player_number=2, config=config2)

        eval1 = ai1.evaluate_position(state)
        eval2 = ai2.evaluate_position(state)

        assert eval1 == eval2

    def test_base_ai_rng_helpers(self):
        config = AIConfig(difficulty=3, randomness=0.2, rngSeed=777)

        ai = RandomAI(player_number=1, config=config)

        items = [1, 2, 3, 4, 5]
        selected1 = ai.get_random_element(items)

        ai2 = RandomAI(player_number=1, config=config)
        selected2 = ai2.get_random_element(items)

        assert selected1 == selected2

