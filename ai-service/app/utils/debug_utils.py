"""Debug Utilities for RingRift AI.

This module consolidates valuable debugging patterns harvested from archived
debug scripts in scripts/archive/debug/. These utilities help diagnose:

- Python/TypeScript state parity issues
- GPU/CPU computation differences
- Game replay verification
- State comparison and diffing

Usage:
    from app.utils.debug_utils import (
        StateDiffer,
        GameReplayDebugger,
        summarize_game_state,
        compare_states,
    )

    # Compare Python and TypeScript states
    differ = StateDiffer()
    diff_report = differ.diff_py_ts_state(py_state, ts_state_dict)

    # Debug game replay
    debugger = GameReplayDebugger(db_path)
    debugger.trace_game(game_id, max_moves=50)

Harvested from:
    - scripts/archive/debug/debug_ts_python_state_diff.py
    - scripts/archive/debug/debug_gpu_cpu_parity.py
    - scripts/archive/debug/debug_chain_divergence.py

Created: December 2025
Purpose: Consolidation of debug utilities (Phase 14 integration)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# GPU bookkeeping moves that CPU handles implicitly
GPU_BOOKKEEPING_MOVES = frozenset({
    "skip_capture",
    "skip_recovery",
    "no_placement_action",
    "no_movement_action",
    "no_line_action",
    "no_territory_action",
    "process_line",
    "process_territory_region",
    "choose_line_option",
    "choose_territory_option",
})


@dataclass
class StateDiff:
    """Result of comparing two game states."""
    
    are_equal: bool = True
    phase_match: bool = True
    player_match: bool = True
    status_match: bool = True
    
    player_diffs: dict[int, dict[str, tuple[Any, Any]]] = field(default_factory=dict)
    stack_only_in_first: list[str] = field(default_factory=list)
    stack_only_in_second: list[str] = field(default_factory=list)
    stack_value_diffs: dict[str, tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=dict)
    collapsed_only_in_first: list[str] = field(default_factory=list)
    collapsed_only_in_second: list[str] = field(default_factory=list)
    collapsed_value_diffs: dict[str, tuple[int, int]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.are_equal = (
            self.phase_match
            and self.player_match
            and self.status_match
            and not self.player_diffs
            and not self.stack_only_in_first
            and not self.stack_only_in_second
            and not self.stack_value_diffs
            and not self.collapsed_only_in_first
            and not self.collapsed_only_in_second
            and not self.collapsed_value_diffs
        )
    
    def summary(self) -> str:
        """Generate a human-readable summary of the diff."""
        if self.are_equal:
            return "States are identical"
        
        lines = ["State differences found:"]
        
        if not self.phase_match:
            lines.append("  - Phase mismatch")
        if not self.player_match:
            lines.append("  - Current player mismatch")
        if not self.status_match:
            lines.append("  - Game status mismatch")
        
        if self.player_diffs:
            lines.append(f"  - Player attribute differences: {len(self.player_diffs)} players")
        
        if self.stack_only_in_first:
            lines.append(f"  - Stacks only in first: {len(self.stack_only_in_first)}")
        if self.stack_only_in_second:
            lines.append(f"  - Stacks only in second: {len(self.stack_only_in_second)}")
        if self.stack_value_diffs:
            lines.append(f"  - Stack value differences: {len(self.stack_value_diffs)}")
        
        if self.collapsed_only_in_first:
            lines.append(f"  - Collapsed only in first: {len(self.collapsed_only_in_first)}")
        if self.collapsed_only_in_second:
            lines.append(f"  - Collapsed only in second: {len(self.collapsed_only_in_second)}")
        if self.collapsed_value_diffs:
            lines.append(f"  - Collapsed owner differences: {len(self.collapsed_value_diffs)}")
        
        return "\n".join(lines)


class StateDiffer:
    """Compare game states between Python and TypeScript representations.
    
    Harvested from: scripts/archive/debug/debug_ts_python_state_diff.py
    """
    
    @staticmethod
    def summarize_players_py(state: Any) -> dict[int, tuple[int, int, int]]:
        """Summarize player stats from Python GameState.
        
        Returns:
            Dict mapping player number to (eliminated_rings, territory_spaces, rings_in_hand)
        """
        summary: dict[int, tuple[int, int, int]] = {}
        for p in state.players:
            summary[p.player_number] = (
                p.eliminated_rings,
                p.territory_spaces,
                p.rings_in_hand,
            )
        return summary
    
    @staticmethod
    def summarize_players_ts(ts_state: dict[str, Any]) -> dict[int, tuple[int, int, int]]:
        """Summarize player stats from TypeScript state dict.
        
        Returns:
            Dict mapping player number to (eliminated_rings, territory_spaces, rings_in_hand)
        """
        summary: dict[int, tuple[int, int, int]] = {}
        for p in ts_state.get("players", []):
            num = int(p["playerNumber"])
            summary[num] = (
                int(p.get("eliminatedRings", 0)),
                int(p.get("territorySpaces", 0)),
                int(p.get("ringsInHand", 0)),
            )
        return summary
    
    @staticmethod
    def summarize_stacks_py(state: Any) -> dict[str, tuple[int, int]]:
        """Summarize board stacks from Python GameState.
        
        Returns:
            Dict mapping position key to (stack_height, controlling_player)
        """
        board = state.board
        out: dict[str, tuple[int, int]] = {}
        for key, stack in (board.stacks or {}).items():
            out[str(key)] = (stack.stack_height, stack.controlling_player)
        return out
    
    @staticmethod
    def summarize_stacks_ts(ts_state: dict[str, Any]) -> dict[str, tuple[int, int]]:
        """Summarize board stacks from TypeScript state dict.
        
        Returns:
            Dict mapping position key to (stack_height, controlling_player)
        """
        board = ts_state.get("board", {})
        stacks = board.get("stacks", {}) or {}
        out: dict[str, tuple[int, int]] = {}
        for key, stack in stacks.items():
            out[str(key)] = (
                int(stack.get("stackHeight", 0)),
                int(stack.get("controllingPlayer", 0)),
            )
        return out
    
    @staticmethod
    def summarize_collapsed_py(state: Any) -> dict[str, int]:
        """Summarize collapsed territory from Python GameState.
        
        Returns:
            Dict mapping position key to owner player number
        """
        board = state.board
        out: dict[str, int] = {}
        for key, owner in (board.collapsed_spaces or {}).items():
            out[str(key)] = int(owner)
        return out
    
    @staticmethod
    def summarize_collapsed_ts(ts_state: dict[str, Any]) -> dict[str, int]:
        """Summarize collapsed territory from TypeScript state dict.
        
        Returns:
            Dict mapping position key to owner player number
        """
        board = ts_state.get("board", {})
        collapsed = board.get("collapsedSpaces", {}) or {}
        out: dict[str, int] = {}
        for key, owner in collapsed.items():
            out[str(key)] = int(owner)
        return out
    
    def diff_py_ts_state(
        self,
        py_state: Any,
        ts_state: dict[str, Any],
    ) -> StateDiff:
        """Compare Python and TypeScript game states.
        
        Args:
            py_state: Python GameState object
            ts_state: TypeScript state as dict (from JSON)
            
        Returns:
            StateDiff with comparison results
        """
        diff = StateDiff()
        
        # Compare phase, player, status
        py_phase = getattr(py_state, "current_phase", None)
        py_phase_val = py_phase.value if hasattr(py_phase, "value") else str(py_phase)
        ts_phase = ts_state.get("currentPhase")
        diff.phase_match = py_phase_val == ts_phase
        
        diff.player_match = py_state.current_player == ts_state.get("currentPlayer")
        
        py_status = getattr(py_state, "game_status", None)
        py_status_val = py_status.value if hasattr(py_status, "value") else str(py_status)
        ts_status = ts_state.get("gameStatus")
        diff.status_match = py_status_val == ts_status
        
        # Compare players
        py_players = self.summarize_players_py(py_state)
        ts_players = self.summarize_players_ts(ts_state)
        all_players = sorted(set(py_players) | set(ts_players))
        
        for num in all_players:
            py_vals = py_players.get(num)
            ts_vals = ts_players.get(num)
            if py_vals != ts_vals:
                diff.player_diffs[num] = {
                    "py": py_vals,
                    "ts": ts_vals,
                }
        
        # Compare stacks
        py_stacks = self.summarize_stacks_py(py_state)
        ts_stacks = self.summarize_stacks_ts(ts_state)
        py_keys = set(py_stacks)
        ts_keys = set(ts_stacks)
        
        diff.stack_only_in_first = sorted(py_keys - ts_keys)
        diff.stack_only_in_second = sorted(ts_keys - py_keys)
        
        for key in sorted(py_keys & ts_keys):
            if py_stacks[key] != ts_stacks[key]:
                diff.stack_value_diffs[key] = (py_stacks[key], ts_stacks[key])
        
        # Compare collapsed
        py_collapsed = self.summarize_collapsed_py(py_state)
        ts_collapsed = self.summarize_collapsed_ts(ts_state)
        py_c_keys = set(py_collapsed)
        ts_c_keys = set(ts_collapsed)
        
        diff.collapsed_only_in_first = sorted(py_c_keys - ts_c_keys)
        diff.collapsed_only_in_second = sorted(ts_c_keys - py_c_keys)
        
        for key in sorted(py_c_keys & ts_c_keys):
            if py_collapsed[key] != ts_collapsed[key]:
                diff.collapsed_value_diffs[key] = (py_collapsed[key], ts_collapsed[key])
        
        # Recompute are_equal
        diff.are_equal = (
            diff.phase_match
            and diff.player_match
            and diff.status_match
            and not diff.player_diffs
            and not diff.stack_only_in_first
            and not diff.stack_only_in_second
            and not diff.stack_value_diffs
            and not diff.collapsed_only_in_first
            and not diff.collapsed_only_in_second
            and not diff.collapsed_value_diffs
        )
        
        return diff


def summarize_game_state(state: Any) -> dict[str, Any]:
    """Create a summary dict of a game state for debugging.
    
    Args:
        state: Python GameState object
        
    Returns:
        Dict with key state information for logging/debugging
    """
    phase = getattr(state, "current_phase", None)
    phase_val = phase.value if hasattr(phase, "value") else str(phase)
    
    status = getattr(state, "game_status", None)
    status_val = status.value if hasattr(status, "value") else str(status)
    
    return {
        "phase": phase_val,
        "current_player": state.current_player,
        "game_status": status_val,
        "move_number": getattr(state, "move_number", 0),
        "winner": getattr(state, "winner", None),
        "num_stacks": len(getattr(state.board, "stacks", {}) or {}),
        "num_collapsed": len(getattr(state.board, "collapsed_spaces", {}) or {}),
        "total_elims": getattr(state, "total_rings_eliminated", 0),
        "players": {
            p.player_number: {
                "eliminated": p.eliminated_rings,
                "territory": p.territory_spaces,
                "in_hand": p.rings_in_hand,
            }
            for p in state.players
        },
    }


def load_ts_state_dump(path: Path) -> dict[str, Any]:
    """Load a TypeScript state dump from JSON file.
    
    Args:
        path: Path to the JSON dump file
        
    Returns:
        Parsed state dict
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not path.exists():
        raise FileNotFoundError(f"TS state dump not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_bookkeeping_move(move_type: str) -> bool:
    """Check if a move type is a GPU bookkeeping move.
    
    GPU bookkeeping moves are handled implicitly by the CPU game engine
    and should be skipped when replaying GPU-generated games on CPU.
    
    Args:
        move_type: Move type string
        
    Returns:
        True if this is a bookkeeping move
    """
    return move_type in GPU_BOOKKEEPING_MOVES


def compare_states(state1: Any, state2: Any, label1: str = "state1", label2: str = "state2") -> str:
    """Quick comparison of two Python game states.
    
    Args:
        state1: First GameState
        state2: Second GameState
        label1: Label for first state in output
        label2: Label for second state in output
        
    Returns:
        Human-readable comparison string
    """
    s1 = summarize_game_state(state1)
    s2 = summarize_game_state(state2)
    
    lines = [f"Comparing {label1} vs {label2}:"]
    
    for key in ["phase", "current_player", "game_status", "move_number", "winner"]:
        v1, v2 = s1.get(key), s2.get(key)
        match = "✓" if v1 == v2 else "✗"
        lines.append(f"  {key}: {v1} vs {v2} {match}")
    
    for key in ["num_stacks", "num_collapsed", "total_elims"]:
        v1, v2 = s1.get(key), s2.get(key)
        match = "✓" if v1 == v2 else "✗"
        lines.append(f"  {key}: {v1} vs {v2} {match}")
    
    # Compare players
    p1, p2 = s1.get("players", {}), s2.get("players", {})
    all_players = sorted(set(p1.keys()) | set(p2.keys()))
    
    for pnum in all_players:
        pv1 = p1.get(pnum, {})
        pv2 = p2.get(pnum, {})
        if pv1 != pv2:
            lines.append(f"  P{pnum} differs:")
            for attr in ["eliminated", "territory", "in_hand"]:
                v1 = pv1.get(attr, "?")
                v2 = pv2.get(attr, "?")
                if v1 != v2:
                    lines.append(f"    {attr}: {v1} vs {v2}")
    
    return "\n".join(lines)


class GameReplayDebugger:
    """Debug utility for tracing game replays.
    
    Harvested from: scripts/archive/debug/debug_gpu_cpu_parity.py
    """
    
    def __init__(self, db_path: Path | str | None = None):
        """Initialize debugger.
        
        Args:
            db_path: Optional path to GameReplayDB for loading states
        """
        self.db_path = Path(db_path) if db_path else None
        self._db = None
    
    def _get_db(self):
        """Lazy-load the database."""
        if self._db is None and self.db_path:
            from app.db.game_replay import GameReplayDB
            GameReplayDB.DB_FILE_EXTENSION = ".db"
            self._db = GameReplayDB(str(self.db_path))
        return self._db
    
    def get_state_at_move(self, game_id: str, move_index: int) -> Any:
        """Load state at a specific move from the database.
        
        Args:
            game_id: Game identifier
            move_index: Move index (0-based)
            
        Returns:
            GameState at that move
            
        Raises:
            RuntimeError: If state cannot be loaded
        """
        db = self._get_db()
        if db is None:
            raise RuntimeError("No database configured")
        
        state = db.get_state_at_move(game_id, move_index)
        if state is None:
            raise RuntimeError(f"get_state_at_move({game_id}, {move_index}) returned None")
        return state
    
    def trace_moves(
        self,
        moves: list[dict[str, Any]],
        initial_state: Any,
        max_moves: int = 50,
        skip_bookkeeping: bool = True,
    ) -> list[dict[str, Any]]:
        """Trace through moves and report on each step.
        
        Args:
            moves: List of move dicts (from GPU export or game record)
            initial_state: Starting GameState
            max_moves: Maximum moves to trace
            skip_bookkeeping: Whether to skip GPU bookkeeping moves
            
        Returns:
            List of trace entries with success/failure info
        """
        from app.game_engine import GameEngine
        from app.models import MoveType, Position
        
        trace: list[dict[str, Any]] = []
        state = initial_state
        
        for i, m in enumerate(moves[:max_moves]):
            move_type_str = m.get("type", "unknown")
            
            # Skip bookkeeping moves if requested
            if skip_bookkeeping and is_bookkeeping_move(move_type_str):
                trace.append({
                    "index": i,
                    "status": "skipped",
                    "move_type": move_type_str,
                    "reason": "bookkeeping",
                })
                continue
            
            # Parse move
            try:
                move_type = MoveType(move_type_str)
            except ValueError:
                trace.append({
                    "index": i,
                    "status": "error",
                    "move_type": move_type_str,
                    "reason": f"Unknown move type: {move_type_str}",
                })
                continue
            
            from_pos = Position(**m["from"]) if m.get("from") else None
            to_pos = Position(**m["to"]) if m.get("to") else None
            
            # Find matching valid move
            valid = GameEngine.get_valid_moves(state, state.current_player)
            matched = None
            
            for v in valid:
                if v.type != move_type:
                    continue
                v_to = v.to.to_key() if v.to else None
                m_to = to_pos.to_key() if to_pos else None
                v_from = v.from_pos.to_key() if v.from_pos else None
                m_from = from_pos.to_key() if from_pos else None
                
                if v_from == m_from and v_to == m_to:
                    matched = v
                    break
            
            if matched:
                state = GameEngine.apply_move(state, matched)
                trace.append({
                    "index": i,
                    "status": "ok",
                    "move_type": move_type_str,
                    "phase": state.current_phase.value,
                    "player": state.current_player,
                })
            else:
                # Try bookkeeping synthesis
                req = GameEngine.get_phase_requirement(state, state.current_player)
                if req:
                    synth = GameEngine.synthesize_bookkeeping_move(req, state)
                    if synth:
                        state = GameEngine.apply_move(state, synth)
                        trace.append({
                            "index": i,
                            "status": "synthesized",
                            "move_type": move_type_str,
                            "synth_type": synth.type.value,
                        })
                        continue
                
                trace.append({
                    "index": i,
                    "status": "failed",
                    "move_type": move_type_str,
                    "expected_from": str(from_pos),
                    "expected_to": str(to_pos),
                    "valid_count": len(valid),
                    "phase": state.current_phase.value,
                    "player": state.current_player,
                })
        
        return trace


__all__ = [
    "GPU_BOOKKEEPING_MOVES",
    "StateDiff",
    "StateDiffer",
    "GameReplayDebugger",
    "summarize_game_state",
    "load_ts_state_dump",
    "is_bookkeeping_move",
    "compare_states",
]
