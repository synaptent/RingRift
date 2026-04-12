"""Cross-language parity coverage contracts for every supported game config."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

from app.board_manager import BoardManager
from app.game_engine import GameEngine
from app.models import BoardType, GameState
from app.training.initial_state import create_initial_state

AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AI_SERVICE_ROOT.parent
TS_HARNESS = REPO_ROOT / "tests" / "scripts" / "ts_rules_config_trace_parity.ts"
TRACE_LENGTH = 3
SUPPORTED_CONFIGS = [
    (BoardType.HEX8, 2),
    (BoardType.HEX8, 3),
    (BoardType.HEX8, 4),
    (BoardType.HEXAGONAL, 2),
    (BoardType.HEXAGONAL, 3),
    (BoardType.HEXAGONAL, 4),
    (BoardType.SQUARE8, 2),
    (BoardType.SQUARE8, 3),
    (BoardType.SQUARE8, 4),
    (BoardType.SQUARE19, 2),
    (BoardType.SQUARE19, 3),
    (BoardType.SQUARE19, 4),
]


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _stack_snapshot(state: GameState) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for key, stack in state.board.stacks.items():
        entries.append(
            {
                "key": key,
                "controllingPlayer": stack.controlling_player,
                "stackHeight": stack.stack_height,
                "capHeight": stack.cap_height,
                # TS snapshots serialize rings top->bottom while Python stores
                # them bottom->top internally.
                "rings": list(reversed(stack.rings)),
            }
        )
    return sorted(entries, key=lambda item: str(item["key"]))


def _marker_snapshot(state: GameState) -> list[dict[str, object]]:
    entries = [
        {"key": key, "player": marker.player}
        for key, marker in state.board.markers.items()
    ]
    return sorted(entries, key=lambda item: str(item["key"]))


def _collapsed_snapshot(state: GameState) -> list[dict[str, object]]:
    entries = [
        {"key": key, "player": player}
        for key, player in state.board.collapsed_spaces.items()
    ]
    return sorted(entries, key=lambda item: str(item["key"]))


def _player_snapshot(state: GameState) -> list[dict[str, object]]:
    entries = [
        {
            "playerNumber": player.player_number,
            "type": player.type,
            "ringsInHand": player.rings_in_hand,
            "eliminatedRings": player.eliminated_rings,
            "territorySpaces": player.territory_spaces,
        }
        for player in state.players
    ]
    return sorted(entries, key=lambda item: int(item["playerNumber"]))


def _comparable_snapshot(case_id: str, state: GameState) -> dict[str, object]:
    return {
        "label": case_id,
        "boardType": _enum_value(state.board_type),
        "currentPlayer": state.current_player,
        "currentPhase": _enum_value(state.current_phase),
        "gameStatus": _enum_value(state.game_status),
        "totalRingsInPlay": state.total_rings_in_play,
        "totalRingsEliminated": state.total_rings_eliminated,
        "players": _player_snapshot(state),
        "stacks": _stack_snapshot(state),
        "markers": _marker_snapshot(state),
        "collapsedSpaces": _collapsed_snapshot(state),
    }


def _progress_snapshot(state: GameState) -> dict[str, int]:
    snapshot = BoardManager.compute_progress_snapshot(state)
    return {
        "markers": snapshot.markers,
        "collapsed": snapshot.collapsed,
        "eliminated": snapshot.eliminated,
        "S": snapshot.S,
    }


def _case_id(board_type: BoardType, num_players: int) -> str:
    return f"{board_type.value}_{num_players}p_trace{TRACE_LENGTH}"


def _serialize_trace(board_type: BoardType, num_players: int) -> tuple[list[dict[str, object]], GameState]:
    state = create_initial_state(board_type=board_type, num_players=num_players)
    trace: list[dict[str, object]] = []

    for _ in range(TRACE_LENGTH):
        legal_moves = GameEngine.get_valid_moves(state, state.current_player)
        if legal_moves:
            move = legal_moves[0]
        else:
            requirement = GameEngine.get_phase_requirement(state, state.current_player)
            assert requirement is not None, (
                f"{board_type.value}/{num_players}p ran out of interactive moves "
                "without a canonical bookkeeping requirement"
            )
            move = GameEngine.synthesize_bookkeeping_move(requirement, state)
            assert move is not None, (
                f"{board_type.value}/{num_players}p could not synthesize bookkeeping "
                f"move for {requirement.type.value}"
            )

        trace.append(move.model_dump(by_alias=True, exclude_none=True, mode="json"))

        try:
            state = GameEngine.apply_move(state, move, trace_mode=True)
        except TypeError:
            state = GameEngine.apply_move(state, move)

    return trace, state


@lru_cache(maxsize=1)
def _ts_results_by_config() -> dict[tuple[str, int], dict[str, object]]:
    npx_path = os.environ.get("RINGRIFT_NPX_PATH") or shutil.which("npx")
    assert npx_path is not None, "npx is required for TS↔Python parity coverage contracts"
    assert TS_HARNESS.exists(), f"Missing TS parity harness: {TS_HARNESS}"

    cases: list[dict[str, object]] = []
    expected_by_config: dict[tuple[str, int], dict[str, object]] = {}

    for board_type, num_players in SUPPORTED_CONFIGS:
        trace, final_state = _serialize_trace(board_type, num_players)
        case_id = _case_id(board_type, num_players)
        config_key = (board_type.value, num_players)
        cases.append(
            {
                "caseId": case_id,
                "boardType": board_type.value,
                "numPlayers": num_players,
                "moves": trace,
            }
        )
        expected_by_config[config_key] = {
            "snapshot": _comparable_snapshot(case_id, final_state),
            "progress": _progress_snapshot(final_state),
        }

    payload = {"cases": cases}
    env = os.environ.copy()
    env.setdefault("TS_NODE_PROJECT", "tsconfig.server.json")

    with tempfile.TemporaryDirectory(prefix="rules-parity-coverage-") as temp_dir:
        payload_path = Path(temp_dir) / "payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        proc = subprocess.run(
            [npx_path, "ts-node", "-T", str(TS_HARNESS), str(payload_path)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    assert proc.returncode == 0, (
        "TS parity harness failed:\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )

    raw_results = json.loads(proc.stdout)
    results_by_config: dict[tuple[str, int], dict[str, object]] = {}
    for entry in raw_results["results"]:
        config_key = (str(entry["boardType"]), int(entry["numPlayers"]))
        results_by_config[config_key] = {
            "ts": {
                "snapshot": entry["snapshot"],
                "progress": entry["progress"],
            },
            "python": expected_by_config[config_key],
        }
    return results_by_config


def test_rules_parity_trace_contract_covers_all_supported_configs() -> None:
    """Every supported board/player config must have a TS↔Python trace contract."""
    results = _ts_results_by_config()
    assert set(results) == {
        (board_type.value, num_players)
        for board_type, num_players in SUPPORTED_CONFIGS
    }


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "board_type,num_players",
    SUPPORTED_CONFIGS,
    ids=lambda value: value.value if isinstance(value, BoardType) else str(value),
)
def test_supported_config_trace_matches_typescript_snapshot(
    board_type: BoardType,
    num_players: int,
) -> None:
    """Each supported config should match TS after a short canonical trace."""
    result = _ts_results_by_config()[(board_type.value, num_players)]
    assert result["python"]["snapshot"] == result["ts"]["snapshot"]
    assert result["python"]["progress"] == result["ts"]["progress"]
