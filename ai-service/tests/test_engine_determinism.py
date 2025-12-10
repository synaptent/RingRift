"""
Engine determinism tests for Python GameEngine.

Verifies that applying a fixed, fully specified sequence of moves starting
from a canonical initial state produces identical final game state snapshots
and canonical hashes on repeated runs (RR‑CANON R190).
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Tuple

# Ensure app package is importable when running tests directly.
# Mirrors the pattern used in other ai-service tests.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.models import (  # noqa: E402
    GameState,
    BoardType,
    GamePhase,
    GameStatus,
    Position,
)
from app.game_engine import GameEngine  # noqa: E402
from app.rules.core import (  # noqa: E402
    hash_game_state,
    compute_progress_snapshot,
    summarize_board,
)


def make_initial_state() -> GameState:
    """Construct a canonical initial GameState for determinism testing.

    This mirrors the structure used in other engine tests but keeps the
    configuration minimal: an empty 8x8 board, two players with equal
    supplies, and an active ring_placement phase for player 1.
    """
    from app.models import BoardState, Player, TimeControl

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
            id="p1",
            username="P1",
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
            id="p2",
            username="P2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=None,
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

    # Use a fixed timestamp so that metadata fields cannot introduce
    # spurious differences between runs.
    fixed_ts = datetime(2020, 1, 1, 0, 0, 0)

    return GameState(
        id="engine-determinism",
        boardType=BoardType.SQUARE8,
        rngSeed=42,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=time_control,
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        createdAt=fixed_ts,
        lastMoveAt=fixed_ts,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=18,  # RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold=33,
        mustMoveFromStackKey=None,
        chainCaptureState=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
    )


def _build_script(initial_timestamp: datetime):
    """Build a fixed script of legal moves from the canonical initial state.

    Script:
      1. P1 places a single ring at (0,0).
      2. P1 moves that stack from (0,0) to (0,1).

    These moves are legal under the shared rules and exercise placement,
    movement, marker placement, and phase transitions.
    """
    from app.models import Move, MoveType

    move1 = Move(  # type: ignore[call-arg]
        id="m1",
        type=MoveType.PLACE_RING,
        player=1,
        to=Position(x=0, y=0),
        timestamp=initial_timestamp,
        thinkTime=0,
        moveNumber=1,
        placementCount=1,
    )

    move2 = Move(  # type: ignore[call-arg]
        **{
            "id": "m2",
            "type": MoveType.MOVE_STACK,
            "player": 1,
            "from": Position(x=0, y=0),
            "to": Position(x=0, y=1),
            "timestamp": initial_timestamp,
            "thinkTime": 0,
            "moveNumber": 2,
        }
    )

    return move1, move2


def _normalise_state(state: GameState) -> Dict[str, Any]:
    """Project a GameState onto a deterministic, hashable snapshot.

    This deliberately ignores non-rules metadata such as timestamps,
    rngSeed, and zobristHash, and focuses only on:
      - current player, phase, and game status,
      - per-player ring/territory counters,
      - global elimination/threshold fields,
      - canonical board summary (stacks/markers/collapsed),
      - canonical S-invariant (markers, collapsed, eliminated).
    """
    progress = compute_progress_snapshot(state)
    board_summary = summarize_board(state.board)

    players_view = sorted(
        [
            {
                "playerNumber": p.player_number,
                "ringsInHand": p.rings_in_hand,
                "eliminatedRings": p.eliminated_rings,
                "territorySpaces": p.territory_spaces,
            }
            for p in state.players
        ],
        key=lambda p: p["playerNumber"],
    )

    snapshot: Dict[str, Any] = {
        "boardType": state.board.type,
        "currentPlayer": state.current_player,
        "currentPhase": state.current_phase,
        "gameStatus": state.game_status,
        "totalRingsInPlay": state.total_rings_in_play,
        "totalRingsEliminated": state.total_rings_eliminated,
        "victoryThreshold": state.victory_threshold,
        "territoryVictoryThreshold": state.territory_victory_threshold,
        "players": players_view,
        "progress": progress,
        "boardSummary": board_summary,
    }

    return snapshot


def run_engine_script() -> Tuple[Dict[str, Any], str]:
    """Run the fixed move script from a fresh initial state.

    Returns:
      - a normalised snapshot of the final state, and
      - the canonical hash from app.rules.core.hash_game_state.
    """
    state = make_initial_state()
    # Use the canonical last_move_at field from the Pydantic model.
    script = _build_script(state.last_move_at)

    current = state
    for mv in script:
        current = GameEngine.apply_move(current, mv)

    snapshot = _normalise_state(current)
    canonical_hash = hash_game_state(current)

    return snapshot, canonical_hash


def test_engine_state_determinism():
    """Running the same script twice yields identical state and hash."""
    snap1, hash1 = run_engine_script()
    snap2, hash2 = run_engine_script()

    assert snap2 == snap1
    assert hash2 == hash1