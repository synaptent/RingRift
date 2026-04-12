"""Contracts for TS↔Python rules surface completeness."""

from __future__ import annotations

import re
from pathlib import Path

from app.models import GamePhase, MoveType
from app.rules.history_contract import ALWAYS_VALID_MOVE_TYPES, phase_move_contract
from app.utils.victory_type import normalize_victory_type

AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AI_SERVICE_ROOT.parent
TS_GAME_TYPES = REPO_ROOT / "src" / "shared" / "types" / "game.ts"
TS_ENGINE_TYPES = REPO_ROOT / "src" / "shared" / "engine" / "types.ts"


def _extract_type_union(source: str, type_name: str) -> list[str]:
    values: list[str] = []
    lines = source.splitlines()
    start_pattern = re.compile(rf"^export type {re.escape(type_name)}\s*=\s*(.*)$")
    collecting = False

    for line in lines:
        if not collecting:
            match = start_pattern.match(line)
            if match is None:
                continue
            collecting = True
            rhs = match.group(1).strip()
            values.extend(re.findall(r"'([^']+)'", rhs))
            if rhs.endswith(";"):
                return values
            continue

        stripped = line.strip()
        if stripped.startswith("|"):
            values.extend(re.findall(r"'([^']+)'", stripped))
        if stripped.endswith(";"):
            return values

    assert collecting, f"Could not find type union {type_name}"
    raise AssertionError(f"Could not find terminating semicolon for type union {type_name}")


def _extract_interface_property_union(
    source: str,
    interface_name: str,
    property_name: str,
) -> list[str]:
    interface_match = re.search(
        rf"export interface {re.escape(interface_name)}\s*\{{(.*?)\n\}}",
        source,
        re.DOTALL,
    )
    assert interface_match is not None, f"Could not find interface {interface_name}"
    interface_body = interface_match.group(1)
    property_match = re.search(
        rf"{re.escape(property_name)}\s*:\s*(.*?);",
        interface_body,
        re.DOTALL,
    )
    assert property_match is not None, (
        f"Could not find property {property_name} on {interface_name}"
    )
    return re.findall(r"'([^']+)'", property_match.group(1))


def test_python_move_types_cover_typescript_move_surface() -> None:
    """Python MoveType must cover the canonical TS move surface."""
    source = TS_GAME_TYPES.read_text(encoding="utf-8")
    ts_move_types = set(_extract_type_union(source, "CanonicalMoveType"))
    ts_move_types.update(_extract_type_union(source, "LegacyMoveType"))

    py_move_types = {move_type.value for move_type in MoveType}

    missing = sorted(ts_move_types - py_move_types)
    unexpected = sorted(py_move_types - ts_move_types)

    assert not missing, f"Python MoveType is missing TS move types: {missing}"
    assert unexpected == ["chain_capture"], (
        "Python MoveType should only carry one intentional compatibility-only "
        f"extra (chain_capture); got extras {unexpected}"
    )


def test_history_contract_covers_all_canonical_typescript_move_types() -> None:
    """The canonical storage contract should cover every canonical TS move."""
    source = TS_GAME_TYPES.read_text(encoding="utf-8")
    ts_canonical_moves = set(_extract_type_union(source, "CanonicalMoveType"))

    contract_moves = set(ALWAYS_VALID_MOVE_TYPES)
    for allowed_moves in phase_move_contract().values():
        contract_moves.update(allowed_moves)

    assert ts_canonical_moves == contract_moves


def test_python_game_phases_match_typescript() -> None:
    """Python GamePhase must stay in lockstep with TS GamePhase."""
    source = TS_GAME_TYPES.read_text(encoding="utf-8")
    ts_phases = set(_extract_type_union(source, "GamePhase"))
    py_phases = {phase.value for phase in GamePhase}
    assert py_phases == ts_phases


def test_typescript_win_conditions_normalize_to_python_victory_types() -> None:
    """Every TS WinCondition type should map into Python's canonical labels."""
    source = TS_GAME_TYPES.read_text(encoding="utf-8")
    win_conditions = _extract_interface_property_union(source, "WinCondition", "type")
    normalized = {normalize_victory_type(value) for value in win_conditions}
    assert normalized == {"ring_elimination", "territory", "lps"}


def test_legacy_engine_action_surface_is_not_used_for_move_completeness() -> None:
    """Document the legacy ActionType surface so audits don't use the wrong file."""
    source = TS_ENGINE_TYPES.read_text(encoding="utf-8")
    action_types = set(_extract_type_union(source, "ActionType"))
    assert action_types == {
        "PLACE_RING",
        "MOVE_STACK",
        "OVERTAKING_CAPTURE",
        "CONTINUE_CHAIN",
        "PROCESS_LINE",
        "CHOOSE_LINE_REWARD",
        "PROCESS_TERRITORY",
        "ELIMINATE_STACK",
        "SKIP_PLACEMENT",
    }
