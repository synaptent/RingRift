from __future__ import annotations

"""
Canonical phase ↔ move contract shared by history validators and DB writers.

This module is intentionally free of GameReplayDB or engine imports so it can
be used safely from both read‑side tooling (parity, offline validation) and
write‑side paths (GameReplayDB._store_move_conn).
"""

from dataclasses import dataclass
from typing import Literal

CanonicalPhase = Literal[
    "ring_placement",
    "movement",
    "capture",
    "chain_capture",
    "line_processing",
    "territory_processing",
    "forced_elimination",
]


def phase_move_contract() -> dict[CanonicalPhase, tuple[str, ...]]:
    """
    Canonical mapping from phase -> allowed move types for recordings.

    This encodes the RR‑CANON‑R070/R075 phase‑to‑MoveType contract at the
    storage level. Moves that do not appear here are treated as
    non‑canonical for parity/recording purposes even if legacy engines
    might still accept them.
    """
    return {
        "ring_placement": (
            "place_ring",
            "skip_placement",
            # RR‑CANON‑R075: forced no‑op when no placement is possible.
            "no_placement_action",
            # Pie rule (2‑player): Player 2 may swap sides after P1's first placement.
            "swap_sides",
        ),
        "movement": (
            "move_stack",
            "move_ring",
            "build_stack",
            "overtaking_capture",
            "continue_capture_segment",
            # RR‑CANON‑R110–R115: recovery action when eligible.
            "recovery_slide",
            # RR-CANON-R115: recovery-eligible players may skip recovery to preserve buried rings.
            "skip_recovery",
            # RR‑CANON‑R075: forced no‑op when no movement/capture is possible.
            "no_movement_action",
        ),
        "capture": (
            "overtaking_capture",
            "continue_capture_segment",
            "skip_capture",
        ),
        "chain_capture": (
            "continue_capture_segment",
        ),
        "line_processing": (
            "process_line",
            "choose_line_option",
            "no_line_action",
        ),
        "territory_processing": (
            "choose_territory_option",
            "eliminate_rings_from_stack",
            "skip_territory_processing",
            "no_territory_action",
        ),
        "forced_elimination": (
            "forced_elimination",
        ),
    }


def _move_type_to_phase_map() -> dict[str, str]:
    """
    Inverted mapping from move_type -> canonical phase.

    Used to infer phase when the stored phase field is empty (legacy
    recordings that didn't track phase at the Move level). For move types
    that appear in multiple phases (e.g. 'overtaking_capture' in both
    movement and capture), we return the most common/primary phase.
    """
    contract = phase_move_contract()
    result: dict[str, str] = {}
    # Process in reverse priority order so earlier phases win for ambiguous types.
    for phase in reversed(list(contract.keys())):
        for move_type in contract[phase]:  # type: ignore[literal-required]
            result[move_type] = phase
    return result


def derive_phase_from_move_type(move_type: str) -> str:
    """
    Derive the canonical phase for a given move_type.

    Returns empty string if the move_type is not in the canonical contract.
    """
    return _move_type_to_phase_map().get(move_type, "")


@dataclass(frozen=True)
class CanonicalMoveCheckResult:
    """Result of validating a single (phase, move_type) pair."""

    ok: bool
    effective_phase: str
    reason: str | None = None


def validate_canonical_move(phase: str | None, move_type: str) -> CanonicalMoveCheckResult:
    """
    Validate a single (phase, move_type) pair against the canonical contract.

    - If phase is empty/None, we infer it from move_type.
    - If move_type is unknown, or the phase is not canonical, or the pair
      is inconsistent, ok will be False and reason will describe why.
    """
    raw_phase = (phase or "").strip()
    eff_phase = raw_phase or derive_phase_from_move_type(move_type)

    contract = phase_move_contract()

    if not eff_phase:
        return CanonicalMoveCheckResult(
            ok=False,
            effective_phase="",
            reason=f"non_canonical_move_type:{move_type}",
        )

    if eff_phase not in contract:
        return CanonicalMoveCheckResult(
            ok=False,
            effective_phase=eff_phase,
            reason=f"non_canonical_phase:{eff_phase}",
        )

    allowed = contract[eff_phase]  # type: ignore[index]
    if move_type not in allowed:
        return CanonicalMoveCheckResult(
            ok=False,
            effective_phase=eff_phase,
            reason=f"phase_move_mismatch:{eff_phase}:{move_type}",
        )

    return CanonicalMoveCheckResult(ok=True, effective_phase=eff_phase, reason=None)
