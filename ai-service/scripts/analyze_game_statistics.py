#!/usr/bin/env python3
"""Comprehensive game statistics analysis tool for RingRift selfplay data.

This script analyzes selfplay game data across all board types and player counts,
producing detailed statistics on:
- Victory type distribution
- Win rates by player position
- Game length statistics
- Recovery action usage
- Board type comparisons
- Performance metrics

Usage:
    python scripts/analyze_game_statistics.py --data-dir data/selfplay
    python scripts/analyze_game_statistics.py --data-dir data/selfplay --output report.json
    python scripts/analyze_game_statistics.py --data-dir data/selfplay --format markdown

Output formats:
    - json: Machine-readable JSON report
    - markdown: Human-readable markdown report (default)
    - both: Both formats
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
AI_SERVICE_ROOT = Path(PROJECT_ROOT).resolve()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# Schema Normalization - handles old and new JSONL formats
# =============================================================================

# Mapping from board_size to board_type
BOARD_SIZE_TO_TYPE = {
    8: "square8",
    19: "square19",
    25: "square25",
    11: "hexagonal",
    13: "hexagonal",
    15: "hexagonal",
}

# Mapping from old victory_type to normalized victory type
OLD_VICTORY_TYPE_MAP = {
    "ring_elimination": "elimination",
    "elimination": "elimination",
    "territory": "territory",
    "lps": "lps",
    "last_player_standing": "lps",
    "timeout": "timeout",
    "stalemate": "stalemate",
    "draw": "draw",
}

# Mapping from new termination_reason format to normalized victory type
TERMINATION_REASON_MAP = {
    "status:completed:elimination": "elimination",
    "status:completed:territory": "territory",
    "status:completed:lps": "lps",
    "status:completed:timeout": "timeout",
    "status:completed:stalemate": "stalemate",
    "status:completed:draw": "draw",
    "status:completed:unknown": "unknown",
}

# AI type inference from file paths
AI_PATH_PATTERNS = {
    "mcts_nn": "mcts+nn",
    "mcts-nn": "mcts+nn",
    "mcts_only": "mcts",
    "mcts-only": "mcts",
    "descent_nn": "descent+nn",
    "descent-nn": "descent+nn",
    "descent_only": "descent",
    "descent-only": "descent",
    "nn_only": "neural_net",
    "nn-only": "neural_net",
    "heuristic_only": "heuristic",
    "heuristic-only": "heuristic",
    "random_only": "random",
    "random-only": "random",
    "gpu_heuristic": "gpu_heuristic",
    "gpu_": "gpu_heuristic",
    "cpu_canonical": "cpu_heuristic",
    "fresh_cpu": "cpu_heuristic",
    "hybrid": "hybrid_gpu",
}


def infer_ai_type(game: dict[str, Any], file_path: str = "") -> str:
    """Infer AI opponent type from game data or file path."""
    # Check explicit engine_mode first
    engine_mode = game.get("engine_mode")
    if engine_mode:
        mode_map = {
            "heuristic-only": "heuristic",
            "mcts-only": "mcts",
            "descent-only": "descent",
            "random-only": "random",
            "nn-only": "neural_net",
            "mixed": "mixed",
        }
        return mode_map.get(engine_mode, engine_mode)

    # Check ai_config or similar fields
    ai_config = game.get("ai_config") or game.get("ai_type") or game.get("opponent_type")
    if ai_config:
        return str(ai_config)

    # Infer from file path
    if file_path:
        path_lower = file_path.lower()
        for pattern, ai_type in AI_PATH_PATTERNS.items():
            if pattern in path_lower:
                return ai_type

    return "unknown"


def normalize_game(game: dict[str, Any], file_path: str = "") -> dict[str, Any]:
    """Normalize a game record to handle both old and new schema formats.

    Old schema fields:
      - board_size: int (8, 19, 25, etc.)
      - victory_type: "ring_elimination", "timeout", etc.
      - stalemate_tiebreaker: optional string

    New schema fields:
      - board_type: "square8", "square19", "hexagonal", etc.
      - termination_reason: "status:completed:elimination", etc.

    Returns a normalized game dict with consistent field names.
    """
    normalized = game.copy()

    # --- Normalize board_type ---
    if "board_type" not in normalized or normalized.get("board_type") == "unknown":
        if "config" in game and "board_type" in game["config"]:
            normalized["board_type"] = game["config"]["board_type"]
        elif "board_size" in game:
            board_size = game["board_size"]
            normalized["board_type"] = BOARD_SIZE_TO_TYPE.get(board_size, f"square{board_size}")
        elif "moves" in game and game["moves"]:
            max_coord = 0
            for move in game["moves"]:
                if isinstance(move, dict):
                    if "to" in move:
                        max_coord = max(max_coord, move["to"].get("x", 0), move["to"].get("y", 0))
            if max_coord <= 7:
                normalized["board_type"] = "square8"
            elif max_coord <= 18:
                normalized["board_type"] = "square19"
            else:
                normalized["board_type"] = f"square{max_coord + 1}"
        else:
            normalized["board_type"] = "unknown"

    # --- Normalize num_players ---
    if "num_players" not in normalized:
        if "config" in game and "num_players" in game["config"]:
            normalized["num_players"] = game["config"]["num_players"]
        elif "moves" in game and game["moves"]:
            max_player = max(
                (m.get("player", 1) for m in game["moves"] if isinstance(m, dict) and "player" in m),
                default=2
            )
            normalized["num_players"] = max_player
        else:
            normalized["num_players"] = 2

    # --- Normalize victory_type ---
    # Prefer explicit victory_type when present; several generators historically
    # emitted termination_reason values that were too coarse (e.g. mapping all
    # non-threshold endings to "lps"). termination_reason remains a fallback for
    # older schemas that didn't emit victory_type.
    victory_type = None

    if "victory_type" in game and game["victory_type"] is not None:
        old_vtype = (
            game["victory_type"].lower()
            if isinstance(game["victory_type"], str)
            else str(game["victory_type"])
        )
        victory_type = OLD_VICTORY_TYPE_MAP.get(old_vtype, old_vtype)

    if victory_type is None and "termination_reason" in game:
        tr = game["termination_reason"]
        if tr in TERMINATION_REASON_MAP:
            victory_type = TERMINATION_REASON_MAP[tr]
        elif isinstance(tr, str) and tr.startswith("status:completed:"):
            victory_type = tr.split(":")[-1]

    if victory_type == "stalemate" and game.get("stalemate_tiebreaker"):
        normalized["_stalemate_tiebreaker"] = game["stalemate_tiebreaker"].lower()

    if victory_type:
        normalized["victory_type"] = victory_type

    # --- Infer AI type ---
    normalized["_ai_type"] = infer_ai_type(game, file_path)

    return normalized


def is_completed_game(game: dict[str, Any], *, include_winner_only: bool) -> bool:
    """Check if a game record represents a completed game (not an eval pool position)."""
    if game.get("game_status") == "active":
        return False

    has_moves = "moves" in game and len(game.get("moves", [])) > 0
    has_victory = "victory_type" in game or "termination_reason" in game
    has_winner = game.get("winner") is not None

    is_eval_pool = "move_history" in game and "moves" not in game
    if is_eval_pool:
        return False

    return has_moves or has_victory or (include_winner_only and has_winner)


@dataclass
class GameStats:
    """Statistics for a single game configuration (board + player count)."""

    board_type: str
    num_players: int
    total_games: int = 0
    total_moves: int = 0
    total_time_seconds: float = 0.0
    wins_by_player: dict[str, int] = field(default_factory=dict)
    victory_types: dict[str, int] = field(default_factory=dict)
    # Stalemate breakdown by tiebreaker type
    stalemate_by_tiebreaker: dict[str, int] = field(default_factory=dict)
    draws: int = 0
    games_with_recovery: int = 0
    total_recovery_opportunities: int = 0
    games_with_fe: int = 0
    game_lengths: list[int] = field(default_factory=list)  # Individual game lengths
    # Enhanced move analysis
    move_type_counts: dict[str, int] = field(default_factory=dict)
    # Recovery slide analysis
    games_with_recovery_slide: int = 0
    recovery_slides_by_player: dict[str, int] = field(default_factory=dict)  # player -> count
    wins_with_recovery_slide: int = 0  # Games where winner used recovery slide
    wins_without_recovery_slide: int = 0  # Games where winner did not use recovery slide
    # Capture chain analysis
    max_capture_chain_lengths: list[int] = field(default_factory=list)  # Max chain per game
    total_captures: int = 0
    total_chain_captures: int = 0
    # Forced elimination analysis
    fe_counts_per_game: list[int] = field(default_factory=list)
    fe_by_player: dict[str, int] = field(default_factory=dict)
    # Game phase analysis
    ring_placement_moves: int = 0  # How many place_ring moves
    territory_claims: int = 0  # process_territory_region moves
    line_formations: int = 0  # process_line moves
    # Tempo/momentum analysis
    consecutive_skips_by_player: dict[str, list[int]] = field(default_factory=dict)  # player -> list of skip streaks
    longest_skip_streak_per_game: list[int] = field(default_factory=list)
    # Comebacks: player in "losing" position who won
    games_with_late_fe_winner: int = 0  # Winner had forced elimination in game
    # Active game length (non-skip moves)
    active_moves_per_game: list[int] = field(default_factory=list)  # moves that aren't no_*_action
    # First blood analysis: who captures first
    first_capture_by_player: dict[str, int] = field(default_factory=dict)
    first_capturer_wins: int = 0
    first_capturer_loses: int = 0
    # Config/value sanity checks (helps detect rule drift in mixed datasets)
    starting_rings_per_player_counts: dict[str, int] = field(default_factory=dict)
    victory_threshold_counts: dict[str, int] = field(default_factory=dict)
    territory_victory_threshold_counts: dict[str, int] = field(default_factory=dict)
    lps_rounds_required_counts: dict[str, int] = field(default_factory=dict)
    # Recovery mode breakdown (RR-CANON-R112)
    recovery_slides_by_mode: dict[str, int] = field(default_factory=dict)
    games_with_stack_strike: int = 0
    wins_with_stack_strike: int = 0
    # Timeout diagnostics
    timeout_move_count_hist: dict[str, int] = field(default_factory=dict)

    @property
    def moves_per_game(self) -> float:
        return self.total_moves / self.total_games if self.total_games > 0 else 0.0

    @property
    def games_per_second(self) -> float:
        return self.total_games / self.total_time_seconds if self.total_time_seconds > 0 else 0.0

    @property
    def min_moves(self) -> int:
        return min(self.game_lengths) if self.game_lengths else 0

    @property
    def max_moves(self) -> int:
        return max(self.game_lengths) if self.game_lengths else 0

    @property
    def median_moves(self) -> float:
        if not self.game_lengths:
            return 0.0
        sorted_lengths = sorted(self.game_lengths)
        n = len(sorted_lengths)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_lengths[mid - 1] + sorted_lengths[mid]) / 2
        return float(sorted_lengths[mid])

    @property
    def std_moves(self) -> float:
        if len(self.game_lengths) < 2:
            return 0.0
        avg = self.moves_per_game
        variance = sum((x - avg) ** 2 for x in self.game_lengths) / len(self.game_lengths)
        return variance ** 0.5

    def win_rate(self, player: int) -> float:
        wins = self.wins_by_player.get(str(player), 0)
        return wins / self.total_games if self.total_games > 0 else 0.0

    def victory_type_rate(self, vtype: str) -> float:
        count = self.victory_types.get(vtype, 0)
        return count / self.total_games if self.total_games > 0 else 0.0

    def get_aggregated_victory_counts(self) -> dict[str, int]:
        """Get aggregated victory counts.

        Aggregation rules:
        - Territory: territory + stalemate (territory tiebreaker)
        - Elimination: elimination + ring_elimination + stalemate (ring_elimination tiebreaker)
        - LPS: lps (last player standing)
        """
        territory = self.victory_types.get("territory", 0)
        stalemate_territory = self.stalemate_by_tiebreaker.get("territory", 0)
        # If no stalemate breakdown, assume all stalemate are territory
        if not self.stalemate_by_tiebreaker and self.victory_types.get("stalemate", 0) > 0:
            stalemate_territory = self.victory_types.get("stalemate", 0)

        elimination = self.victory_types.get("elimination", 0)
        ring_elim = self.victory_types.get("ring_elimination", 0)
        stalemate_ring_elim = self.stalemate_by_tiebreaker.get("ring_elimination", 0)

        lps = self.victory_types.get("lps", 0)

        return {
            "territory": territory + stalemate_territory,
            "elimination": elimination + ring_elim + stalemate_ring_elim,
            "lps": lps,
        }


@dataclass
class AnalysisReport:
    """Complete analysis report across all configurations."""

    stats_by_config: dict[tuple[str, int], GameStats] = field(default_factory=dict)
    recovery_analysis: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data_sources: list[str] = field(default_factory=list)

    def total_games(self) -> int:
        return sum(s.total_games for s in self.stats_by_config.values())

    def total_moves(self) -> int:
        return sum(s.total_moves for s in self.stats_by_config.values())


def load_stats_json(path: Path) -> dict[str, Any] | None:
    """Load a stats.json file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
        return None


def load_recovery_analysis(path: Path) -> dict[str, Any] | None:
    """Load recovery analysis results if available."""
    candidates = [
        path / "recovery_analysis_results.json",
        path / "actual_recovery_opportunities.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return None


def iter_jsonl_games(path: Path, *, include_winner_only: bool) -> Iterator[dict[str, Any]]:
    """Yield normalized completed games from a JSONL file.

    This is streaming by design to avoid loading large JSONL files into memory.
    """
    file_path_str = str(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not is_completed_game(game, include_winner_only=include_winner_only):
                    continue
                yield normalize_game(game, file_path_str)
    except OSError:
        return


def _get_first(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _extract_starting_rings_per_player(game: dict[str, Any]) -> int | None:
    state = game.get("initial_state") or game.get("initialState") or game.get("initial_state_json")
    if isinstance(state, dict):
        players = state.get("players")
        if isinstance(players, list) and players:
            rings: list[int] = []
            for p in players:
                if not isinstance(p, dict):
                    continue
                value = _get_first(p, ["rings_in_hand", "ringsInHand"])
                if isinstance(value, int):
                    rings.append(value)
            if rings:
                if len(set(rings)) == 1:
                    return rings[0]
                return min(rings)

        # Some traces store a matrix of rings-in-hand by player index.
        matrix = _get_first(state, ["rings_in_hand", "ringsInHand"])
        if isinstance(matrix, list) and matrix:
            ints = [v for v in matrix if isinstance(v, int)]
            if ints:
                if len(set(ints)) == 1:
                    return ints[0]
                return min(ints)

    config = game.get("config")
    if isinstance(config, dict):
        value = _get_first(config, ["rings_per_player", "ringsPerPlayer"])
        if isinstance(value, int):
            return value

    return None


def _extract_victory_threshold(game: dict[str, Any]) -> int | None:
    state = game.get("initial_state") or game.get("initialState") or game.get("initial_state_json")
    if isinstance(state, dict):
        value = _get_first(state, ["victory_threshold", "victoryThreshold"])
        if isinstance(value, int):
            return value
    return None


def _extract_territory_victory_threshold(game: dict[str, Any]) -> int | None:
    state = game.get("initial_state") or game.get("initialState") or game.get("initial_state_json")
    if isinstance(state, dict):
        value = _get_first(state, ["territory_victory_threshold", "territoryVictoryThreshold"])
        if isinstance(value, int):
            return value
    return None


def _extract_lps_rounds_required(game: dict[str, Any]) -> int | None:
    state = game.get("initial_state") or game.get("initialState") or game.get("initial_state_json")
    if isinstance(state, dict):
        rules_options = _get_first(state, ["rules_options", "rulesOptions"])
        if isinstance(rules_options, dict):
            value = _get_first(rules_options, ["lpsRoundsRequired", "lps_rounds_required"])
            if isinstance(value, int):
                return value
    return None


def collect_stats_from_jsonl(
    jsonl_files: list[Path], report: AnalysisReport, *, include_winner_only: bool
) -> None:
    """Collect statistics from JSONL files and add to report."""
    for jsonl_path in jsonl_files:
        any_games = False
        for game in iter_jsonl_games(jsonl_path, include_winner_only=include_winner_only):
            any_games = True
            board_type = game.get("board_type", "unknown")
            num_players = int(game.get("num_players", 2) or 2)
            key = (board_type, num_players)
            if key not in report.stats_by_config:
                report.stats_by_config[key] = GameStats(board_type=board_type, num_players=num_players)

            stats = report.stats_by_config[key]

            stats.total_games += 1
            move_count = game.get("move_count", len(game.get("moves", [])))
            stats.total_moves += move_count
            stats.game_lengths.append(move_count)
            stats.total_time_seconds += game.get("game_time_seconds", 0.0)

            # Initial-state sanity checks / drift detection.
            starting_rings = _extract_starting_rings_per_player(game)
            if starting_rings is not None:
                key_str = str(starting_rings)
                stats.starting_rings_per_player_counts[key_str] = (
                    stats.starting_rings_per_player_counts.get(key_str, 0) + 1
                )

            victory_threshold = _extract_victory_threshold(game)
            if victory_threshold is not None:
                key_str = str(victory_threshold)
                stats.victory_threshold_counts[key_str] = (
                    stats.victory_threshold_counts.get(key_str, 0) + 1
                )

            territory_threshold = _extract_territory_victory_threshold(game)
            if territory_threshold is not None:
                key_str = str(territory_threshold)
                stats.territory_victory_threshold_counts[key_str] = (
                    stats.territory_victory_threshold_counts.get(key_str, 0) + 1
                )

            lps_rounds_required = _extract_lps_rounds_required(game)
            if lps_rounds_required is not None:
                key_str = str(lps_rounds_required)
                stats.lps_rounds_required_counts[key_str] = (
                    stats.lps_rounds_required_counts.get(key_str, 0) + 1
                )

            # Winner
            winner = game.get("winner")
            if winner is not None:
                stats.wins_by_player[str(winner)] = stats.wins_by_player.get(str(winner), 0) + 1

            # Victory type
            victory_type = game.get("victory_type")
            if victory_type:
                stats.victory_types[victory_type] = stats.victory_types.get(victory_type, 0) + 1
                if victory_type == "timeout":
                    move_key = str(int(move_count) if isinstance(move_count, int) else 0)
                    stats.timeout_move_count_hist[move_key] = (
                        stats.timeout_move_count_hist.get(move_key, 0) + 1
                    )

            # Stalemate tiebreaker
            tiebreaker = game.get("stalemate_tiebreaker")
            if tiebreaker and victory_type == "stalemate":
                stats.stalemate_by_tiebreaker[tiebreaker] = (
                    stats.stalemate_by_tiebreaker.get(tiebreaker, 0) + 1
                )

            # Analyze moves in detail
            moves = game.get("moves", [])
            has_fe = False
            fe_count = 0
            has_recovery_slide = False
            has_stack_strike = False
            recovery_slides_by_player_in_game: dict[str, int] = {}
            stack_strikes_by_player_in_game: set[str] = set()
            fe_by_player_in_game: dict[str, int] = {}
            capture_chain_length = 0
            max_chain_in_game = 0
            in_chain = False
            first_capture_player = None
            active_move_count = 0
            # Track consecutive skips per player
            current_skip_streak: dict[str, int] = {}
            max_skip_streak = 0

            for m in moves:
                if not isinstance(m, dict):
                    continue
                move_type = m.get("type", "")
                player = m.get("player")
                player_str = str(player) if player is not None else None

                # Count move types
                stats.move_type_counts[move_type] = stats.move_type_counts.get(move_type, 0) + 1

                # Track phase-specific actions
                if move_type == "place_ring":
                    stats.ring_placement_moves += 1
                elif move_type in ("choose_territory_option", "process_territory_region"):
                    stats.territory_claims += 1
                elif move_type in ("choose_line_option", "choose_line_reward", "process_line"):
                    stats.line_formations += 1

                # Track active vs skip moves
                is_skip = move_type.startswith("no_") or move_type.startswith("skip_")
                if not is_skip:
                    active_move_count += 1
                    # Reset skip streak for this player
                    if player_str:
                        if player_str in current_skip_streak and current_skip_streak[player_str] > 0:
                            if player_str not in stats.consecutive_skips_by_player:
                                stats.consecutive_skips_by_player[player_str] = []
                            stats.consecutive_skips_by_player[player_str].append(current_skip_streak[player_str])
                        current_skip_streak[player_str] = 0
                elif player_str:
                    # Increment skip streak
                    current_skip_streak[player_str] = current_skip_streak.get(player_str, 0) + 1
                    if current_skip_streak[player_str] > max_skip_streak:
                        max_skip_streak = current_skip_streak[player_str]

                # Track forced elimination
                if move_type == "forced_elimination":
                    has_fe = True
                    fe_count += 1
                    if player_str:
                        stats.fe_by_player[player_str] = stats.fe_by_player.get(player_str, 0) + 1
                        fe_by_player_in_game[player_str] = fe_by_player_in_game.get(player_str, 0) + 1

                # Track recovery slide
                if move_type == "recovery_slide":
                    has_recovery_slide = True
                    recovery_mode = m.get("recoveryMode") or m.get("recovery_mode")
                    if isinstance(recovery_mode, str) and recovery_mode:
                        stats.recovery_slides_by_mode[recovery_mode] = (
                            stats.recovery_slides_by_mode.get(recovery_mode, 0) + 1
                        )
                        if recovery_mode == "stack_strike":
                            has_stack_strike = True
                            if player_str:
                                stack_strikes_by_player_in_game.add(player_str)
                    if player_str:
                        recovery_slides_by_player_in_game[player_str] = (
                            recovery_slides_by_player_in_game.get(player_str, 0) + 1
                        )
                        stats.recovery_slides_by_player[player_str] = (
                            stats.recovery_slides_by_player.get(player_str, 0) + 1
                        )

                # Track capture chains
                if move_type == "overtaking_capture":
                    stats.total_captures += 1
                    # Track first capture
                    if first_capture_player is None and player_str:
                        first_capture_player = player_str
                        stats.first_capture_by_player[player_str] = (
                            stats.first_capture_by_player.get(player_str, 0) + 1
                        )
                    if in_chain:
                        capture_chain_length += 1
                    else:
                        in_chain = True
                        capture_chain_length = 1
                elif move_type == "continue_capture_segment":
                    stats.total_chain_captures += 1
                    capture_chain_length += 1
                else:
                    # Chain ended
                    if capture_chain_length > max_chain_in_game:
                        max_chain_in_game = capture_chain_length
                    capture_chain_length = 0
                    in_chain = False

            # Final chain check
            if capture_chain_length > max_chain_in_game:
                max_chain_in_game = capture_chain_length

            # Record active moves
            stats.active_moves_per_game.append(active_move_count)

            # Record max skip streak for this game
            if max_skip_streak > 0:
                stats.longest_skip_streak_per_game.append(max_skip_streak)

            if has_fe:
                stats.games_with_fe += 1
                stats.fe_counts_per_game.append(fe_count)
                # Check if winner had forced elimination (comeback)
                if winner is not None and str(winner) in fe_by_player_in_game:
                    stats.games_with_late_fe_winner += 1

            if has_recovery_slide:
                stats.games_with_recovery_slide += 1
                # Check if winner used recovery slide
                if winner is not None and str(winner) in recovery_slides_by_player_in_game:
                    stats.wins_with_recovery_slide += 1
                elif winner is not None:
                    stats.wins_without_recovery_slide += 1
            elif winner is not None:
                stats.wins_without_recovery_slide += 1

            if has_stack_strike:
                stats.games_with_stack_strike += 1
                if winner is not None and str(winner) in stack_strikes_by_player_in_game:
                    stats.wins_with_stack_strike += 1

            if max_chain_in_game > 0:
                stats.max_capture_chain_lengths.append(max_chain_in_game)

            # First capture correlation with winning
            if first_capture_player is not None and winner is not None:
                if first_capture_player == str(winner):
                    stats.first_capturer_wins += 1
                else:
                    stats.first_capturer_loses += 1

        if any_games:
            report.data_sources.append(str(jsonl_path))


def collect_stats(
    data_dir: Path, jsonl_files: list[Path] | None = None, *, include_winner_only: bool = False
) -> AnalysisReport:
    """Collect statistics from all subdirectories in data_dir and optional JSONL files."""
    report = AnalysisReport()

    # Process JSONL files first if provided
    if jsonl_files:
        collect_stats_from_jsonl(jsonl_files, report, include_winner_only=include_winner_only)

    if data_dir.exists():
        # Find all stats.json files
        for stats_path in data_dir.rglob("stats.json"):
            data = load_stats_json(stats_path)
            if data is None:
                continue

            board_type = data.get("board_type", "unknown")
            num_players = data.get("num_players", 2)
            key = (board_type, num_players)

            if key not in report.stats_by_config:
                report.stats_by_config[key] = GameStats(board_type=board_type, num_players=num_players)

            stats = report.stats_by_config[key]
            stats.total_games += data.get("total_games", 0)
            stats.total_moves += data.get("total_moves", 0)
            stats.total_time_seconds += data.get("total_time_seconds", 0.0)
            stats.draws += data.get("draws", 0)
            stats.games_with_recovery += data.get("games_with_recovery_opportunities", 0)
            stats.total_recovery_opportunities += data.get("total_recovery_opportunities", 0)
            stats.games_with_fe += data.get("games_with_fe", 0)

            # Merge wins by player
            for player, wins in data.get("wins_by_player", {}).items():
                stats.wins_by_player[player] = stats.wins_by_player.get(player, 0) + wins

            # Merge victory types
            victory_types = data.get("victory_type_counts", data.get("victory_types", {}))
            for vtype, count in victory_types.items():
                stats.victory_types[vtype] = stats.victory_types.get(vtype, 0) + count

            # Merge stalemate tiebreaker breakdown
            stalemate_tiebreakers = data.get("stalemate_by_tiebreaker", {})
            for tiebreaker, count in stalemate_tiebreakers.items():
                stats.stalemate_by_tiebreaker[tiebreaker] = (
                    stats.stalemate_by_tiebreaker.get(tiebreaker, 0) + count
                )

            # Collect individual game lengths if available
            game_lengths = data.get("game_lengths", [])
            if game_lengths:
                stats.game_lengths.extend(game_lengths)

            report.data_sources.append(str(stats_path))

        # Load recovery analysis if available
        recovery_data = load_recovery_analysis(data_dir)
        if recovery_data:
            report.recovery_analysis = recovery_data

    return report


def generate_markdown_report(report: AnalysisReport) -> str:
    """Generate a markdown report from the analysis."""
    lines = []
    lines.append("# RingRift Game Statistics Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append(f"**Total Games Analyzed:** {report.total_games():,}")
    lines.append(f"**Total Moves:** {report.total_moves():,}")
    lines.append(f"**Data Sources:** {len(report.data_sources)}")
    lines.append("")

    # Aggregated Victory Categories
    lines.append("## 1. Aggregated Victory Categories")
    lines.append("")
    lines.append("*Categories: Territory (territory + stalemate-territory), ")
    lines.append("Elimination (elimination + ring_elimination + stalemate-ring_elim), LPS (last player standing)*")
    lines.append("")
    lines.append("| Board/Players | Games | Territory | Elimination | LPS |")
    lines.append("|---------------|-------|-----------|-------------|-----|")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        agg = stats.get_aggregated_victory_counts()
        territory = agg["territory"]
        elimination = agg["elimination"]
        lps = agg["lps"]

        territory_pct = f"{100 * territory / stats.total_games:.1f}%" if territory else "0%"
        elimination_pct = f"{100 * elimination / stats.total_games:.1f}%" if elimination else "0%"
        lps_pct = f"{100 * lps / stats.total_games:.1f}%" if lps else "0%"

        lines.append(
            f"| {board_type} {num_players}p | {stats.total_games} | "
            f"{territory_pct} ({territory}) | {elimination_pct} ({elimination}) | {lps_pct} ({lps}) |"
        )

    lines.append("")

    # Detailed Victory Type Distribution (raw)
    lines.append("## 2. Detailed Victory Type Distribution")
    lines.append("")
    lines.append("| Board/Players  | Games | Territory   | Elimination | LPS        | Stalemate   | Timeout    |")
    lines.append("|----------------|-------|-------------|-------------|------------|-------------|------------|")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        territory = stats.victory_types.get("territory", 0)
        ring_elim = stats.victory_types.get("ring_elimination", 0)
        lps = stats.victory_types.get("lps", 0)
        stalemate = stats.victory_types.get("stalemate", 0)
        timeout = stats.victory_types.get("timeout", 0)

        def fmt_pct(count: int, width: int = 11) -> str:
            if count == 0:
                return "-".center(width)
            val = f"{count} ({100 * count / stats.total_games:.0f}%)"
            return val.ljust(width)

        board_col = f"{board_type} {num_players}p".ljust(14)
        games_col = str(stats.total_games).rjust(5)
        lines.append(
            f"| {board_col} | {games_col} | "
            f"{fmt_pct(territory)} | {fmt_pct(ring_elim)} | "
            f"{fmt_pct(lps, 10)} | {fmt_pct(stalemate)} | {fmt_pct(timeout, 10)} |"
        )

    lines.append("")

    # Stalemate Tiebreaker Breakdown
    any_stalemate_data = any(
        stats.stalemate_by_tiebreaker for stats in report.stats_by_config.values()
    )
    if any_stalemate_data:
        lines.append("### Stalemate Tiebreaker Breakdown")
        lines.append("")
        lines.append("| Board/Players | Total Stalemate | Territory | Ring Elim | Other |")
        lines.append("|---------------|-----------------|-----------|-----------|-------|")

        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            total_stalemate = stats.victory_types.get("stalemate", 0)
            if total_stalemate == 0:
                continue
            territory_tb = stats.stalemate_by_tiebreaker.get("territory", 0)
            ring_elim_tb = stats.stalemate_by_tiebreaker.get("ring_elimination", 0)
            other_tb = total_stalemate - territory_tb - ring_elim_tb

            lines.append(
                f"| {board_type} {num_players}p | {total_stalemate} | "
                f"{territory_tb} | {ring_elim_tb} | {other_tb} |"
            )

        lines.append("")

    # Win Distribution by Player Position
    lines.append("## 2. Win Distribution by Player Position")
    lines.append("")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        lines.append(f"### {board_type.title()} {num_players}-player ({stats.total_games} games)")
        lines.append("")
        lines.append("| Player | Wins | Win Rate |")
        lines.append("|--------|------|----------|")

        for p in range(1, num_players + 1):
            wins = stats.wins_by_player.get(str(p), 0)
            rate = 100 * stats.win_rate(p)
            lines.append(f"| Player {p} | {wins} | {rate:.1f}% |")

        # Calculate expected win rate and deviation
        expected = 100 / num_players
        lines.append("")
        lines.append(f"*Expected win rate (uniform): {expected:.1f}%*")
        lines.append("")

    # Game Length Statistics
    lines.append("## 3. Game Length Statistics")
    lines.append("")
    lines.append("| Configuration | Avg Moves | Min | Max | Median | Std Dev | Games/Sec | Total Time |")
    lines.append("|--------------|-----------|-----|-----|--------|---------|-----------|------------|")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        avg_moves = stats.moves_per_game
        min_moves = stats.min_moves
        max_moves = stats.max_moves
        median_moves = stats.median_moves
        std_moves = stats.std_moves
        gps = stats.games_per_second
        total_time = stats.total_time_seconds

        time_str = f"{total_time:.1f}s" if total_time < 60 else f"{total_time / 60:.1f}m"
        # Show detailed stats if game_lengths data is available
        if stats.game_lengths:
            lines.append(
                f"| {board_type} {num_players}p | {avg_moves:.1f} | {min_moves} | {max_moves} | "
                f"{median_moves:.1f} | {std_moves:.1f} | {gps:.3f} | {time_str} |"
            )
        else:
            lines.append(
                f"| {board_type} {num_players}p | {avg_moves:.1f} | - | - | "
                f"- | - | {gps:.3f} | {time_str} |"
            )

    lines.append("")

    # Rules/config drift detection (rings/thresholds).
    any_rings_data = any(
        stats.starting_rings_per_player_counts or stats.victory_threshold_counts
        for stats in report.stats_by_config.values()
    )
    if any_rings_data:
        lines.append("## 4. Rules/Config Sanity Checks")
        lines.append("")
        lines.append("*Detects mixed datasets (e.g., different rings-per-player or LPS thresholds) that can silently skew conclusions.*")
        lines.append("")
        lines.append("| Config | Starting Rings/Player (top) | Victory Threshold (top) | Territory Threshold (top) | LPS Rounds (top) |")
        lines.append("|--------|-----------------------------|-------------------------|---------------------------|-----------------|")

        def fmt_top_counts(counts: dict[str, int]) -> str:
            if not counts:
                return "-"
            items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:2]
            return ", ".join(f"{k}×{v}" for k, v in items)

        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            lines.append(
                f"| {board_type} {num_players}p | "
                f"{fmt_top_counts(stats.starting_rings_per_player_counts)} | "
                f"{fmt_top_counts(stats.victory_threshold_counts)} | "
                f"{fmt_top_counts(stats.territory_victory_threshold_counts)} | "
                f"{fmt_top_counts(stats.lps_rounds_required_counts)} |"
            )
        lines.append("")

    # Recovery Action Analysis
    if report.recovery_analysis:
        lines.append("## 5. Recovery Action Analysis")
        lines.append("")
        ra = report.recovery_analysis

        total = ra.get("total_games", 0)
        with_fe = ra.get("games_with_fe", 0)
        with_recovery = ra.get("games_with_recovery", 0)
        states = ra.get("states_analyzed", 0)

        lines.append(f"- **Games Analyzed:** {total}")
        lines.append(f"- **Games with Forced Elimination:** {with_fe} ({100 * with_fe / total:.1f}%)" if total else "")
        lines.append(f"- **Games with Recovery Used:** {with_recovery}")
        lines.append(f"- **States Analyzed:** {states:,}")
        lines.append("")

        # Condition frequencies
        cond_freq = ra.get("condition_frequencies", {})
        if cond_freq:
            lines.append("### Recovery Condition Frequencies")
            lines.append("")
            lines.append("| Condition | Frequency | % of States |")
            lines.append("|-----------|-----------|-------------|")
            for cond, count in sorted(cond_freq.items(), key=lambda x: -x[1]):
                pct = 100 * count / states if states else 0
                lines.append(f"| {cond} | {count:,} | {pct:.1f}% |")
            lines.append("")

        # Conditions met distribution
        cond_dist = ra.get("conditions_met_distribution", {})
        if cond_dist:
            lines.append("### Conditions Met Distribution")
            lines.append("")
            lines.append("| # Conditions | States | % |")
            lines.append("|--------------|--------|---|")
            for key, count in sorted(cond_dist.items()):
                pct = 100 * count / states if states else 0
                lines.append(f"| {key} | {count:,} | {pct:.1f}% |")
            lines.append("")

    # Enhanced Move Analysis Section
    any_move_data = any(stats.move_type_counts for stats in report.stats_by_config.values())
    if any_move_data:
        lines.append("## 6. Move Type Distribution")
        lines.append("")
        lines.append("*Top 10 move types across all configurations*")
        lines.append("")
        # Aggregate move types
        all_move_types: dict[str, int] = {}
        for stats in report.stats_by_config.values():
            for mtype, count in stats.move_type_counts.items():
                all_move_types[mtype] = all_move_types.get(mtype, 0) + count
        total_moves = sum(all_move_types.values())
        lines.append("| Move Type | Count | % of All Moves |")
        lines.append("|-----------|-------|----------------|")
        for mtype, count in sorted(all_move_types.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / total_moves if total_moves else 0
            lines.append(f"| {mtype} | {count:,} | {pct:.1f}% |")
        lines.append("")

    # Recovery Slide Analysis
    any_recovery_data = any(stats.games_with_recovery_slide > 0 for stats in report.stats_by_config.values())
    lines.append("## 7. Recovery Slide Analysis")
    lines.append("")
    if any_recovery_data:
        lines.append("| Config | Games w/ Recovery | Recovery User Wins | Non-Recovery Wins (in recovery games) |")
        lines.append("|--------|-------------------|--------------------|-----------------------------------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            games_w = stats.games_with_recovery_slide
            wins_w = stats.wins_with_recovery_slide
            # In games where recovery was used, how many were won by non-recovery user
            # This is tracked in wins_without_recovery_slide only for games WITH recovery
            # But our current tracking conflates "games with no recovery" and "recovery games where winner didn't use recovery"
            # Let's show: games_w, wins_w (recovery user won), and games_w - wins_w (non-recovery winner in recovery games)
            pct_games = 100 * games_w / stats.total_games if stats.total_games else 0
            # Out of games with recovery, what % were won by recovery user
            pct_recovery_wins = 100 * wins_w / games_w if games_w else 0
            # Non-recovery wins in recovery games (approximate: games_w had games_w games with winners)
            non_recovery_wins_in_recovery_games = games_w - wins_w if games_w > 0 else 0
            pct_non_recovery = 100 * non_recovery_wins_in_recovery_games / games_w if games_w else 0
            lines.append(
                f"| {board_type} {num_players}p | {games_w} ({pct_games:.1f}%) | "
                f"{wins_w} ({pct_recovery_wins:.1f}%) | {non_recovery_wins_in_recovery_games} ({pct_non_recovery:.1f}%) |"
            )
        lines.append("")
    else:
        lines.append("*No recovery slides detected in the analyzed games.*")
        lines.append("")

    any_recovery_mode_data = any(stats.recovery_slides_by_mode for stats in report.stats_by_config.values())
    if any_recovery_mode_data:
        lines.append("### Recovery Mode Breakdown")
        lines.append("")
        lines.append("| Config | Line | Fallback | Stack-Strike | Games w/ Stack-Strike | Winner Used Stack-Strike |")
        lines.append("|--------|------|----------|--------------|------------------------|--------------------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            line_count = stats.recovery_slides_by_mode.get("line", 0)
            fallback_count = stats.recovery_slides_by_mode.get("fallback", 0)
            strike_count = stats.recovery_slides_by_mode.get("stack_strike", 0)
            lines.append(
                f"| {board_type} {num_players}p | {line_count} | {fallback_count} | {strike_count} | "
                f"{stats.games_with_stack_strike} | {stats.wins_with_stack_strike} |"
            )
        lines.append("")

    # Capture Chain Analysis
    any_capture_data = any(stats.max_capture_chain_lengths for stats in report.stats_by_config.values())
    if any_capture_data:
        lines.append("## 8. Capture Chain Analysis")
        lines.append("")
        lines.append("| Config | Total Captures | Chain Captures | Max Chain | Avg Max Chain |")
        lines.append("|--------|----------------|----------------|-----------|---------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            total_cap = stats.total_captures
            chain_cap = stats.total_chain_captures
            max_chain = max(stats.max_capture_chain_lengths) if stats.max_capture_chain_lengths else 0
            avg_max = sum(stats.max_capture_chain_lengths) / len(stats.max_capture_chain_lengths) if stats.max_capture_chain_lengths else 0
            lines.append(f"| {board_type} {num_players}p | {total_cap:,} | {chain_cap:,} | {max_chain} | {avg_max:.1f} |")
        lines.append("")

    # Forced Elimination Analysis
    any_fe_data = any(stats.games_with_fe > 0 for stats in report.stats_by_config.values())
    if any_fe_data:
        lines.append("## 9. Forced Elimination Analysis")
        lines.append("")
        lines.append("| Config | Games w/ FE | FE Count | Winner Had FE | Comeback Rate |")
        lines.append("|--------|-------------|----------|---------------|---------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0:
                continue
            games_fe = stats.games_with_fe
            pct_fe = 100 * games_fe / stats.total_games if stats.total_games else 0
            total_fe = sum(stats.fe_counts_per_game)
            comebacks = stats.games_with_late_fe_winner
            comeback_rate = 100 * comebacks / games_fe if games_fe else 0
            lines.append(
                f"| {board_type} {num_players}p | {games_fe} ({pct_fe:.1f}%) | "
                f"{total_fe} | {comebacks} | {comeback_rate:.1f}% |"
            )
        lines.append("")

    # First Blood Analysis
    any_first_capture = any(stats.first_capture_by_player for stats in report.stats_by_config.values())
    if any_first_capture:
        lines.append("## 10. First Blood Analysis")
        lines.append("")
        lines.append("*Does making the first capture correlate with winning?*")
        lines.append("")
        lines.append("| Config | First Capturer Wins | First Capturer Loses | Win Rate |")
        lines.append("|--------|---------------------|----------------------|----------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0 or not stats.first_capture_by_player:
                continue
            fc_wins = stats.first_capturer_wins
            fc_loses = stats.first_capturer_loses
            total_fc = fc_wins + fc_loses
            win_rate = 100 * fc_wins / total_fc if total_fc else 0
            expected = 100 / num_players
            delta = win_rate - expected
            delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
            lines.append(
                f"| {board_type} {num_players}p | {fc_wins} | {fc_loses} | "
                f"{win_rate:.1f}% ({delta_str}% vs expected) |"
            )
        lines.append("")

    # Game Tempo/Activity Analysis
    any_activity_data = any(stats.active_moves_per_game for stats in report.stats_by_config.values())
    if any_activity_data:
        lines.append("## 11. Game Activity Analysis")
        lines.append("")
        lines.append("*Active moves (excluding skip/no-action moves)*")
        lines.append("")
        lines.append("| Config | Avg Total Moves | Avg Active Moves | Activity Rate |")
        lines.append("|--------|-----------------|------------------|---------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            if stats.total_games == 0 or not stats.active_moves_per_game:
                continue
            avg_total = stats.moves_per_game
            avg_active = sum(stats.active_moves_per_game) / len(stats.active_moves_per_game)
            activity_rate = 100 * avg_active / avg_total if avg_total else 0
            lines.append(f"| {board_type} {num_players}p | {avg_total:.1f} | {avg_active:.1f} | {activity_rate:.1f}% |")
        lines.append("")

    any_timeout_diag = any(stats.timeout_move_count_hist for stats in report.stats_by_config.values())
    if any_timeout_diag:
        lines.append("## 12. Timeout Diagnostics")
        lines.append("")
        lines.append("*Flags suspiciously consistent timeouts (often a runner/config pathology rather than real gameplay).*")
        lines.append("")
        lines.append("| Config | Timeout Games | Unique Move Counts | Most Common Move Count |")
        lines.append("|--------|--------------:|------------------:|------------------------|")
        for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
            timeout_total = stats.victory_types.get("timeout", 0)
            if timeout_total == 0:
                continue
            hist = stats.timeout_move_count_hist
            unique = len(hist)
            most_common = "-"
            if hist:
                k, v = sorted(hist.items(), key=lambda x: (-x[1], x[0]))[0]
                most_common = f"{k} ({v})"
            lines.append(f"| {board_type} {num_players}p | {timeout_total} | {unique} | {most_common} |")
        lines.append("")

    # Key Findings
    lines.append("## 13. Key Findings")
    lines.append("")

    # Position advantage analysis
    position_advantages = []
    for (board_type, num_players), stats in report.stats_by_config.items():
        if stats.total_games < 10:
            continue
        expected = 1.0 / num_players
        max_player = None
        max_rate = 0.0
        for p in range(1, num_players + 1):
            rate = stats.win_rate(p)
            if rate > max_rate:
                max_rate = rate
                max_player = p
        if max_player and max_rate > expected + 0.05:
            position_advantages.append((board_type, num_players, max_player, max_rate))

    if position_advantages:
        lines.append("### Position Advantages")
        for board, players, player, rate in position_advantages:
            lines.append(f"- **{board} {players}p:** Player {player} has advantage ({rate * 100:.1f}% win rate)")
        lines.append("")

    # Victory type patterns
    lines.append("### Victory Type Patterns")
    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games < 10:
            continue
        dominant = max(stats.victory_types.items(), key=lambda x: x[1], default=(None, 0))
        if dominant[0] and dominant[1] > 0:
            rate = 100 * dominant[1] / stats.total_games
            lines.append(f"- **{board_type} {num_players}p:** {dominant[0]} dominant ({rate:.0f}%)")
    lines.append("")

    # Recovery status
    if report.recovery_analysis:
        with_recovery = report.recovery_analysis.get("games_with_recovery", 0)
        dist = report.recovery_analysis.get("conditions_met_distribution", {}) or {}
        # Canonical RR-CANON-R110 eligibility is 3 conditions. Keep backward
        # compatibility with older analysis artifacts that used a (wrong) 4th
        # "zero rings in hand" condition.
        conditions_met = dist.get("3_conditions", dist.get("4_conditions", 0))
        lines.append("### Recovery Mechanic Status")
        if with_recovery == 0 and conditions_met > 0:
            lines.append(
                f"- **Recovery is NOT being used** despite {conditions_met} states meeting all eligibility conditions"
            )
            lines.append("- This suggests turn-skipping prevents recovery-eligible players from taking turns")
        elif with_recovery > 0:
            lines.append(f"- Recovery was used in {with_recovery} games")
        lines.append("")

    return "\n".join(lines)


def generate_json_report(report: AnalysisReport) -> dict[str, Any]:
    """Generate a JSON-serializable report."""
    result: dict[str, Any] = {
        "timestamp": report.timestamp,
        "summary": {
            "total_games": report.total_games(),
            "total_moves": report.total_moves(),
            "data_sources": len(report.data_sources),
        },
        "configurations": {},
        "recovery_analysis": report.recovery_analysis,
    }

    for (board_type, num_players), stats in report.stats_by_config.items():
        key = f"{board_type}_{num_players}p"
        config_data: dict[str, Any] = {
            "board_type": board_type,
            "num_players": num_players,
            "total_games": stats.total_games,
            "total_moves": stats.total_moves,
            "moves_per_game": stats.moves_per_game,
            "total_time_seconds": stats.total_time_seconds,
            "games_per_second": stats.games_per_second,
            "wins_by_player": stats.wins_by_player,
            "win_rates": {str(p): stats.win_rate(p) for p in range(1, num_players + 1)},
            "victory_types": stats.victory_types,
            "victory_type_rates": {
                vtype: stats.victory_type_rate(vtype) for vtype in stats.victory_types
            },
            "stalemate_by_tiebreaker": stats.stalemate_by_tiebreaker,
            "aggregated_victory_types": stats.get_aggregated_victory_counts(),
            "draws": stats.draws,
            "games_with_recovery": stats.games_with_recovery,
            "games_with_fe": stats.games_with_fe,
            "games_with_recovery_slide": stats.games_with_recovery_slide,
            "recovery_slides_by_player": stats.recovery_slides_by_player,
            "wins_with_recovery_slide": stats.wins_with_recovery_slide,
            "wins_without_recovery_slide": stats.wins_without_recovery_slide,
            "games_with_late_fe_winner": stats.games_with_late_fe_winner,
            "fe_by_player": stats.fe_by_player,
            "move_type_counts": stats.move_type_counts,
            "total_captures": stats.total_captures,
            "total_chain_captures": stats.total_chain_captures,
            "ring_placement_moves": stats.ring_placement_moves,
            "territory_claims": stats.territory_claims,
            "line_formations": stats.line_formations,
            "first_capture_by_player": stats.first_capture_by_player,
            "first_capturer_wins": stats.first_capturer_wins,
            "first_capturer_loses": stats.first_capturer_loses,
            "starting_rings_per_player_counts": stats.starting_rings_per_player_counts,
            "victory_threshold_counts": stats.victory_threshold_counts,
            "territory_victory_threshold_counts": stats.territory_victory_threshold_counts,
            "lps_rounds_required_counts": stats.lps_rounds_required_counts,
            "recovery_slides_by_mode": stats.recovery_slides_by_mode,
            "games_with_stack_strike": stats.games_with_stack_strike,
            "wins_with_stack_strike": stats.wins_with_stack_strike,
            "timeout_move_count_hist": stats.timeout_move_count_hist,
        }
        # Add detailed game length statistics if available
        if stats.game_lengths:
            config_data["game_length_stats"] = {
                "min": stats.min_moves,
                "max": stats.max_moves,
                "median": stats.median_moves,
                "std_dev": stats.std_moves,
            }
        if stats.active_moves_per_game:
            config_data["active_moves_stats"] = {
                "avg": sum(stats.active_moves_per_game) / len(stats.active_moves_per_game),
                "min": min(stats.active_moves_per_game),
                "max": max(stats.active_moves_per_game),
            }
        if stats.longest_skip_streak_per_game:
            config_data["skip_streak_stats"] = {
                "avg_longest": sum(stats.longest_skip_streak_per_game) / len(stats.longest_skip_streak_per_game),
                "max_longest": max(stats.longest_skip_streak_per_game),
            }
        if stats.max_capture_chain_lengths:
            config_data["capture_chain_stats"] = {
                "max_chain": max(stats.max_capture_chain_lengths),
                "avg_max_chain": sum(stats.max_capture_chain_lengths) / len(stats.max_capture_chain_lengths),
            }
        result["configurations"][key] = config_data

    return result


def _read_jsonl_filelist(filelist_path: Path) -> list[Path]:
    """Read a list of JSONL paths from a text/TSV file.

    Supports:
    - One path per line
    - TSV manifests with the path as the last column (e.g. mtime\\tsize\\tpath)
    """
    try:
        lines = filelist_path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        print(f"Warning: Failed to read jsonl file list {filelist_path}: {e}", file=sys.stderr)
        return []

    candidates: list[Path] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if not parts:
            continue
        raw = os.path.expanduser(parts[-1])
        path = Path(raw)
        if not path.is_absolute():
            local_candidate = filelist_path.parent / path
            if local_candidate.exists():
                path = local_candidate
            else:
                path = AI_SERVICE_ROOT / path
        candidates.append(path)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    result: list[Path] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        result.append(p)
    return result


def _expand_jsonl_globs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for p in paths:
        if p.exists():
            expanded.append(p)
            continue
        s = str(p)
        if any(ch in s for ch in ["*", "?", "["]):
            expanded.extend(Path(match) for match in glob.glob(s))
        else:
            expanded.append(p)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    result: list[Path] = []
    for p in expanded:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        result.append(p)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze RingRift selfplay game statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/selfplay"),
        help="Directory containing selfplay data (default: data/selfplay)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: stdout for markdown, auto-named for json)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="If no completed games are found, emit an empty report instead of exiting non-zero.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        help="JSONL files to include in analysis (can specify multiple or use glob patterns)",
    )
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        help="Directory to scan for JSONL files (*.jsonl)",
    )
    parser.add_argument(
        "--jsonl-filelist",
        type=Path,
        help="Text/TSV file listing JSONL files to include (path in last column for TSV).",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=None,
        help="Only include JSONL files modified within the last N hours.",
    )
    parser.add_argument(
        "--include-winner-only",
        action="store_true",
        help="Include records that have a winner but no moves/termination fields (often non-game logs).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Collect JSONL files from various sources
    jsonl_files: list[Path] = []
    if args.jsonl:
        jsonl_files.extend(args.jsonl)
    if args.jsonl_dir and args.jsonl_dir.exists():
        jsonl_files.extend(args.jsonl_dir.glob("*.jsonl"))
    if args.jsonl_filelist:
        jsonl_files.extend(_read_jsonl_filelist(args.jsonl_filelist))

    jsonl_files = _expand_jsonl_globs(jsonl_files)
    missing_files = [p for p in jsonl_files if not p.exists()]
    if missing_files and not args.quiet:
        for p in missing_files[:10]:
            print(f"Warning: JSONL file not found: {p}", file=sys.stderr)
        if len(missing_files) > 10:
            print(f"Warning: ... and {len(missing_files) - 10} more missing JSONL files", file=sys.stderr)
    jsonl_files = [p for p in jsonl_files if p.exists()]
    if args.max_age_hours is not None:
        cutoff = datetime.now(timezone.utc).timestamp() - (args.max_age_hours * 3600)
        jsonl_files = [p for p in jsonl_files if p.stat().st_mtime >= cutoff]

    # Check if we have any data sources
    has_data_dir = args.data_dir.exists()
    has_jsonl = len(jsonl_files) > 0

    if not has_data_dir and not has_jsonl:
        if args.allow_empty:
            report = AnalysisReport()
        else:
            print(f"Error: No data sources found. Data directory does not exist: {args.data_dir}", file=sys.stderr)
            return 1
    else:
        if not args.quiet:
            if has_data_dir:
                print(f"Analyzing data in {args.data_dir}...", file=sys.stderr)
            if has_jsonl:
                print(f"Including {len(jsonl_files)} JSONL file(s)...", file=sys.stderr)

        report = collect_stats(
            args.data_dir,
            jsonl_files if has_jsonl else None,
            include_winner_only=bool(args.include_winner_only),
        )

        if report.total_games() == 0 and not args.allow_empty:
            print("Error: No game data found", file=sys.stderr)
            return 1

        if not args.quiet:
            print(
                f"Found {report.total_games()} games across {len(report.stats_by_config)} configurations",
                file=sys.stderr,
            )

    # Generate outputs
    if args.format in ("markdown", "both"):
        md_report = generate_markdown_report(report)
        if args.output and args.format == "markdown":
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md_report)
            if not args.quiet:
                print(f"Markdown report written to {args.output}", file=sys.stderr)
        elif args.format == "both" and args.output:
            md_path = args.output.with_suffix(".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_report)
            if not args.quiet:
                print(f"Markdown report written to {md_path}", file=sys.stderr)
        else:
            print(md_report)

    if args.format in ("json", "both"):
        json_report = generate_json_report(report)
        if args.output:
            json_path = args.output if args.format == "json" else args.output.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_report, f, indent=2)
            if not args.quiet:
                print(f"JSON report written to {json_path}", file=sys.stderr)
        elif args.format == "json":
            print(json.dumps(json_report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
