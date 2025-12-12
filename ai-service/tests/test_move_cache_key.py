from datetime import datetime

from app.ai.move_cache import cache_moves, get_cached_moves
from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)


def _make_state(
    *,
    board_type: BoardType = BoardType.SQUARE8,
    board_size: int = 8,
    max_players: int = 2,
    zobrist_hash: int = 12345,
    must_move_from_stack_key: str | None = None,
    rules_options: dict | None = None,
) -> GameState:
    now = datetime.now()
    board = BoardState(type=board_type, size=board_size)
    players = [
        Player(
            id=f"p{i}",
            username=f"P{i}",
            type="ai",
            playerNumber=i,
            isReady=True,
            timeRemaining=600,
            aiDifficulty=None,
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0,
        )
        for i in range(1, max_players + 1)
    ]

    return GameState(
        id="move-cache-key-test",
        boardType=board_type,
        rngSeed=None,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        winner=None,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=max_players,
        totalRingsInPlay=40,
        totalRingsEliminated=0,
        victoryThreshold=21,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=must_move_from_stack_key,
        zobristHash=zobrist_hash,
        rulesOptions=rules_options,
    )


def test_move_cache_key_includes_must_move_from_stack_key() -> None:
    state_a = _make_state(must_move_from_stack_key="A1", zobrist_hash=999)
    state_b = _make_state(must_move_from_stack_key="B2", zobrist_hash=999)

    cache_moves(state_a, 1, ["m"])
    assert get_cached_moves(state_a, 1) == ["m"]
    assert get_cached_moves(state_b, 1) is None


def test_move_cache_key_includes_rules_options() -> None:
    state_with_swap = _make_state(rules_options={"swapRuleEnabled": True}, zobrist_hash=111)
    state_without_swap = _make_state(rules_options={"swapRuleEnabled": False}, zobrist_hash=111)

    cache_moves(state_with_swap, 1, ["m"])
    assert get_cached_moves(state_with_swap, 1) == ["m"]
    assert get_cached_moves(state_without_swap, 1) is None


def test_move_cache_key_includes_board_geometry_even_with_same_zobrist() -> None:
    square8 = _make_state(board_type=BoardType.SQUARE8, board_size=8, zobrist_hash=777)
    square19 = _make_state(board_type=BoardType.SQUARE19, board_size=19, zobrist_hash=777)

    cache_moves(square8, 1, ["m"])
    assert get_cached_moves(square8, 1) == ["m"]
    assert get_cached_moves(square19, 1) is None


def test_move_cache_key_includes_max_players() -> None:
    two_player = _make_state(max_players=2, zobrist_hash=4242)
    three_player = _make_state(max_players=3, zobrist_hash=4242)

    cache_moves(two_player, 1, ["m"])
    assert get_cached_moves(two_player, 1) == ["m"]
    assert get_cached_moves(three_player, 1) is None


def test_move_cache_key_includes_player_meta() -> None:
    base = _make_state(zobrist_hash=31337)
    p1 = base.players[0].model_copy(update={"rings_in_hand": 0})
    changed = base.model_copy(update={"players": [p1, *base.players[1:]]})

    cache_moves(base, 1, ["m"])
    assert get_cached_moves(base, 1) == ["m"]
    assert get_cached_moves(changed, 1) is None
