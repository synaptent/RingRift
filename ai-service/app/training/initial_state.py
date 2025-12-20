"""
Lightweight initial state creation for self-play and tournament matches.

This module is intentionally kept free of torch dependencies so it can be
imported without loading neural network libraries. Use this module when
you only need to create game states, not for full training data generation.
"""

import os
from datetime import datetime

from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)


def create_initial_state(
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    rings_per_player_override: int | None = None,
    lps_rounds_required: int = 3,
) -> GameState:
    """Create an initial GameState for self-play.

    This mirrors the core TypeScript `createInitialGameState` logic for
    per-player ring counts and victory thresholds while defaulting to an
    ACTIVE game suitable for self-play training.

    Parameters
    ----------
    board_type:
        Board geometry to use (square8, square19, hexagonal).
    num_players:
        Number of active players in the game (2–4 supported).
    rings_per_player_override:
        Optional override for rings per player (for ablation experiments).
        If None, uses the default from BOARD_CONFIGS.
    lps_rounds_required:
        Number of consecutive exclusive rounds required for LPS victory.
        Default is 2 (traditional rule).
    """
    # Clamp to a sensible range to avoid constructing degenerate states.
    if num_players < 2:
        num_players = 2
    if num_players > 4:
        num_players = 4

    # Use centralized BOARD_CONFIGS from app.rules.core (mirrors TS BOARD_CONFIGS)
    from app.rules.core import (
        BOARD_CONFIGS,
        get_territory_victory_threshold,
        get_victory_threshold,
    )

    if board_type in BOARD_CONFIGS:
        config = BOARD_CONFIGS[board_type]
        size = config.size
        rings_per_player = rings_per_player_override or config.rings_per_player
    else:
        # Fallback to square8-style defaults if an unknown board is passed.
        size = 8
        rings_per_player = rings_per_player_override or 18

    # Victory thresholds per RR-CANON-R061 and RR-CANON-R062
    total_rings = rings_per_player * num_players
    victory_threshold = get_victory_threshold(
        board_type, num_players, rings_per_player_override=rings_per_player_override
    )
    territory_threshold = get_territory_victory_threshold(board_type)

    players = [
        Player(
            id=f"p{idx}",
            username=f"AI {idx}",
            type="ai",
            playerNumber=idx,
            isReady=True,
            timeRemaining=600,
            ringsInHand=rings_per_player,
            eliminatedRings=0,
            territorySpaces=0,
            aiDifficulty=10,
        )
        for idx in range(1, num_players + 1)
    ]

    # Training-time pie rule configuration. The pie rule (swap sides) is
    # opt-in for 2-player games - data shows P2 wins >55% when enabled by default.
    # Multi-player games ignore this and never expose swap_sides.
    #
    # Callers that want to enable the pie rule for experiments may set
    # RINGRIFT_TRAINING_ENABLE_SWAP_RULE=1 (or "true"/"yes"/"on").
    rules_options = None
    enable_flag = os.getenv("RINGRIFT_TRAINING_ENABLE_SWAP_RULE", "").lower()
    if num_players == 2:
        swap_enabled = enable_flag in {"1", "true", "yes", "on"}
        rules_options = {"swapRuleEnabled": swap_enabled}

    return GameState(
        id="self-play",
        boardType=board_type,
        board=BoardState(
            type=board_type,
            size=size,
            stacks={},
            markers={},
            collapsedSpaces={},
            eliminatedRings={},
        ),
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        # For training we start in an ACTIVE state so env loops run.
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=num_players,
        # Training use-cases historically treated this as total rings available
        # in the game rather than "placed on board". Preserve that behaviour
        # but generalise to N players.
        totalRingsInPlay=total_rings,
        totalRingsEliminated=0,
        victoryThreshold=victory_threshold,
        territoryVictoryThreshold=territory_threshold,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        rulesOptions=rules_options,
        lpsRoundsRequired=lps_rounds_required,
    )
