#!/usr/bin/env python3
"""
Analyze selfplay games for recovery opportunities AND forced elimination moves.

This script replays games from JSONL files and detects:
1. Recovery eligibility windows (RR-CANON-R110)
2. Forced elimination moves (RR-CANON-R100)
3. Move type distribution

Usage:
    python scripts/analyze_game_mechanics.py [--dir PATH] [--limit N] [--verbose]
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter


@dataclass
class BoardState:
    """Simplified board state for tracking game mechanics."""
    # Stack data: {position_key: {"rings": [player_numbers], "controller": player}}
    stacks: dict = field(default_factory=dict)
    # Markers: {position_key: player_number}
    markers: dict = field(default_factory=dict)
    # Players with rings in hand
    rings_in_hand: dict = field(default_factory=dict)

    def player_controls_any_stack(self, player: int) -> bool:
        """Check if player controls at least one stack."""
        for stack in self.stacks.values():
            if stack.get("controller") == player:
                return True
        return False

    def count_controlled_stacks(self, player: int) -> int:
        """Count stacks controlled by player."""
        return sum(1 for s in self.stacks.values() if s.get("controller") == player)

    def player_has_marker(self, player: int) -> bool:
        """Check if player has at least one marker."""
        return player in self.markers.values()

    def count_buried_rings(self, player: int) -> int:
        """Count rings owned by player that are buried (not controlling)."""
        count = 0
        for stack in self.stacks.values():
            controller = stack.get("controller")
            rings = stack.get("rings", [])
            for ring in rings:
                if ring == player and controller != player:
                    count += 1
        return count

    def is_eligible_for_recovery(self, player: int) -> bool:
        """Check if player meets all recovery eligibility conditions (RR-CANON-R110)."""
        # Condition 1: Controls no stacks
        if self.player_controls_any_stack(player):
            return False

        # Condition 2: Has at least one marker
        if not self.player_has_marker(player):
            return False

        # Condition 3: Has at least one buried ring
        if self.count_buried_rings(player) < 1:
            return False

        return True

    def has_turn_material(self, player: int) -> bool:
        """Check if player has material to take a turn (old logic)."""
        has_stacks = self.player_controls_any_stack(player)
        has_rings_in_hand = self.rings_in_hand.get(player, 0) > 0
        return has_stacks or has_rings_in_hand


@dataclass
class GameStats:
    """Statistics for a single game."""
    game_index: int
    num_moves: int
    move_types: Counter = field(default_factory=Counter)
    recovery_opportunities: int = 0
    recovery_moves_used: int = 0
    forced_elimination_moves: int = 0
    forced_elimination_opportunities: int = 0
    overtakes: int = 0
    winner: Optional[int] = None
    victory_type: Optional[str] = None


@dataclass
class AnalysisStats:
    """Aggregated statistics from analysis."""
    games_analyzed: int = 0
    total_moves: int = 0
    move_type_counts: Counter = field(default_factory=Counter)

    # Recovery stats
    games_with_recovery_opportunities: int = 0
    total_recovery_opportunities: int = 0
    total_recovery_moves_used: int = 0

    # Forced elimination stats
    games_with_forced_elimination: int = 0
    total_forced_elimination_moves: int = 0
    total_forced_elimination_opportunities: int = 0

    # Overtake stats
    total_overtakes: int = 0
    games_with_overtakes: int = 0

    # Victory stats
    victory_types: Counter = field(default_factory=Counter)
    winners: Counter = field(default_factory=Counter)


def parse_position_key(pos: dict) -> str:
    """Convert position dict to string key."""
    if pos is None:
        return ""
    x = pos.get("x", pos.get("col", 0))
    y = pos.get("y", pos.get("row", 0))
    return f"{x},{y}"


def apply_move_to_board(board: BoardState, move: dict, num_players: int) -> dict:
    """Apply a move to update the board state. Returns move metadata."""
    move_type = move.get("type", "")
    player = move.get("player", 0)
    metadata = {"overtake": False}

    if move_type == "place_ring":
        to_pos = move.get("to")
        if to_pos:
            key = parse_position_key(to_pos)
            placement_count = move.get("placement_count", 2)
            if key not in board.stacks:
                board.stacks[key] = {"rings": [], "controller": player}
            board.stacks[key]["rings"] = [player] * placement_count
            board.stacks[key]["controller"] = player
            board.rings_in_hand[player] = max(0, board.rings_in_hand.get(player, 18) - placement_count)
            board.markers[key] = player

    elif move_type == "move_stack":
        from_pos = move.get("from_pos") or move.get("from")
        to_pos = move.get("to")

        if from_pos and to_pos:
            from_key = parse_position_key(from_pos)
            to_key = parse_position_key(to_pos)

            moving_stack = board.stacks.get(from_key, {"rings": [], "controller": 0})
            board.markers[from_key] = player

            if from_key in board.stacks:
                del board.stacks[from_key]

            if to_key in board.stacks:
                # Overtaking!
                metadata["overtake"] = True
                dest_stack = board.stacks[to_key]
                combined_rings = dest_stack["rings"] + moving_stack["rings"]
                board.stacks[to_key] = {
                    "rings": combined_rings,
                    "controller": player
                }
            else:
                board.stacks[to_key] = {
                    "rings": moving_stack["rings"],
                    "controller": player
                }

    elif move_type == "recovery_slide":
        # Recovery move - similar to move_stack
        from_pos = move.get("from_pos") or move.get("from")
        to_pos = move.get("to")
        if from_pos and to_pos:
            from_key = parse_position_key(from_pos)
            to_key = parse_position_key(to_pos)
            moving_stack = board.stacks.get(from_key, {"rings": [], "controller": 0})
            board.markers[from_key] = player
            if from_key in board.stacks:
                del board.stacks[from_key]
            board.stacks[to_key] = {
                "rings": moving_stack["rings"],
                "controller": player
            }

    elif move_type == "forced_elimination":
        # Forced elimination - player loses a ring
        pass

    return metadata


def analyze_game(game: dict, game_index: int) -> GameStats:
    """Analyze a single game for various mechanics."""
    stats = GameStats(game_index=game_index, num_moves=0)

    moves = game.get("moves", [])
    num_players = game.get("num_players", 2)
    initial_state = game.get("initial_state", {})

    stats.num_moves = len(moves)
    stats.winner = game.get("winner")
    stats.victory_type = game.get("victory_type") or game.get("termination_reason")

    # Initialize board state
    board = BoardState()

    # Set initial rings in hand
    for p in range(1, num_players + 1):
        rings_per_player = 36 // num_players
        board.rings_in_hand[p] = rings_per_player

    # Load initial state if provided
    if initial_state:
        init_stacks = initial_state.get("board", {}).get("stacks", {})
        for key, stack_data in init_stacks.items():
            if isinstance(stack_data, dict):
                board.stacks[key] = {
                    "rings": stack_data.get("rings", []),
                    "controller": stack_data.get("controlling_player", 0)
                }
        init_markers = initial_state.get("board", {}).get("markers", {})
        for key, marker_data in init_markers.items():
            if isinstance(marker_data, dict):
                board.markers[key] = marker_data.get("player", 0)
            else:
                board.markers[key] = marker_data
        init_players = initial_state.get("players", [])
        for pdata in init_players:
            pnum = pdata.get("player_number", 0)
            board.rings_in_hand[pnum] = pdata.get("rings_in_hand", 18)

    # Track recovery eligibility windows
    recovery_eligible: dict[int, bool] = {}

    for i, move in enumerate(moves):
        move_type = move.get("type", "")
        move_player = move.get("player", 0)

        # Count move types
        stats.move_types[move_type] += 1

        # Check for recovery eligibility BEFORE the move
        for p in range(1, num_players + 1):
            was_eligible = recovery_eligible.get(p, False)
            is_eligible = board.is_eligible_for_recovery(p)

            if is_eligible and not was_eligible:
                stats.recovery_opportunities += 1
                recovery_eligible[p] = True
            elif not is_eligible and was_eligible:
                recovery_eligible[p] = False

        # Check for forced elimination conditions
        # Forced elimination: player controls stacks but has NO legal moves
        # This is detected by looking for forced_elimination move type
        if move_type == "forced_elimination":
            stats.forced_elimination_moves += 1

        # Check if player with stacks might be in forced elimination situation
        # (this is an approximation - real check requires legal move generation)
        if move_player > 0 and board.count_controlled_stacks(move_player) > 0:
            # Player has stacks - could potentially be in forced elim situation
            # We'll count this as an "opportunity" for forced elimination to occur
            if move_type in ("move_stack", "place_ring"):
                # Normal move - not forced elimination
                pass
            elif move_type == "no_movement_action":
                # No movement action while having stacks could indicate forced elim
                stats.forced_elimination_opportunities += 1

        # Track recovery moves used
        if move_type == "recovery_slide":
            stats.recovery_moves_used += 1

        # Apply the move
        metadata = apply_move_to_board(board, move, num_players)
        if metadata.get("overtake"):
            stats.overtakes += 1

    return stats


def analyze_jsonl_file(filepath: Path, limit: Optional[int] = None, verbose: bool = False) -> AnalysisStats:
    """Analyze a single JSONL file."""
    stats = AnalysisStats()

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if limit and line_num >= limit:
                    break

                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    continue

                game_stats = analyze_game(game, line_num)

                stats.games_analyzed += 1
                stats.total_moves += game_stats.num_moves
                stats.move_type_counts.update(game_stats.move_types)

                # Recovery stats
                if game_stats.recovery_opportunities > 0:
                    stats.games_with_recovery_opportunities += 1
                stats.total_recovery_opportunities += game_stats.recovery_opportunities
                stats.total_recovery_moves_used += game_stats.recovery_moves_used

                # Forced elimination stats
                if game_stats.forced_elimination_moves > 0:
                    stats.games_with_forced_elimination += 1
                stats.total_forced_elimination_moves += game_stats.forced_elimination_moves
                stats.total_forced_elimination_opportunities += game_stats.forced_elimination_opportunities

                # Overtake stats
                if game_stats.overtakes > 0:
                    stats.games_with_overtakes += 1
                stats.total_overtakes += game_stats.overtakes

                # Victory stats
                if game_stats.victory_type:
                    stats.victory_types[game_stats.victory_type] += 1
                if game_stats.winner is not None:
                    stats.winners[game_stats.winner] += 1

                if verbose and line_num < 5:
                    print(f"  Game {line_num}: {game_stats.num_moves} moves, "
                          f"overtakes={game_stats.overtakes}, "
                          f"recovery_opp={game_stats.recovery_opportunities}, "
                          f"forced_elim={game_stats.forced_elimination_moves}")

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return stats


def merge_stats(total: AnalysisStats, partial: AnalysisStats) -> None:
    """Merge partial stats into total."""
    total.games_analyzed += partial.games_analyzed
    total.total_moves += partial.total_moves
    total.move_type_counts.update(partial.move_type_counts)

    total.games_with_recovery_opportunities += partial.games_with_recovery_opportunities
    total.total_recovery_opportunities += partial.total_recovery_opportunities
    total.total_recovery_moves_used += partial.total_recovery_moves_used

    total.games_with_forced_elimination += partial.games_with_forced_elimination
    total.total_forced_elimination_moves += partial.total_forced_elimination_moves
    total.total_forced_elimination_opportunities += partial.total_forced_elimination_opportunities

    total.games_with_overtakes += partial.games_with_overtakes
    total.total_overtakes += partial.total_overtakes

    total.victory_types.update(partial.victory_types)
    total.winners.update(partial.winners)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze selfplay games for recovery and forced elimination mechanics"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/selfplay/comprehensive",
        help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum games to analyze per file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )

    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    jsonl_files = list(data_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL files in {data_dir}")
    print()

    # Aggregate stats
    total_stats = AnalysisStats()

    for jf in sorted(jsonl_files):
        rel_path = jf.relative_to(data_dir) if data_dir in jf.parents or jf.parent == data_dir else jf.name
        print(f"Analyzing: {rel_path}")
        stats = analyze_jsonl_file(jf, args.limit, args.verbose)
        merge_stats(total_stats, stats)

        if args.verbose:
            print(f"  Games: {stats.games_analyzed}, Moves: {stats.total_moves}")
            print(f"  Overtakes: {stats.total_overtakes}, Recovery opps: {stats.total_recovery_opportunities}")
            print(f"  Forced elim moves: {stats.total_forced_elimination_moves}")
            print()

    # Print summary
    print()
    print("=" * 70)
    print("GAME MECHANICS ANALYSIS SUMMARY")
    print("=" * 70)
    print()

    print(f"Total games analyzed: {total_stats.games_analyzed}")
    print(f"Total moves: {total_stats.total_moves}")
    if total_stats.games_analyzed > 0:
        print(f"Average moves per game: {total_stats.total_moves / total_stats.games_analyzed:.1f}")
    print()

    # Move type breakdown
    print("-" * 40)
    print("MOVE TYPE DISTRIBUTION")
    print("-" * 40)
    for move_type, count in total_stats.move_type_counts.most_common():
        pct = count / total_stats.total_moves * 100 if total_stats.total_moves > 0 else 0
        print(f"  {move_type}: {count} ({pct:.1f}%)")
    print()

    # Overtake stats
    print("-" * 40)
    print("OVERTAKE STATISTICS")
    print("-" * 40)
    print(f"Total overtakes: {total_stats.total_overtakes}")
    print(f"Games with overtakes: {total_stats.games_with_overtakes}")
    if total_stats.games_analyzed > 0:
        pct = total_stats.games_with_overtakes / total_stats.games_analyzed * 100
        print(f"Percentage of games with overtakes: {pct:.1f}%")
    if total_stats.games_with_overtakes > 0:
        avg = total_stats.total_overtakes / total_stats.games_with_overtakes
        print(f"Average overtakes per game (when present): {avg:.1f}")
    print()

    # Recovery stats
    print("-" * 40)
    print("RECOVERY MECHANIC (RR-CANON-R110)")
    print("-" * 40)
    print(f"Games with recovery opportunities: {total_stats.games_with_recovery_opportunities}")
    print(f"Total recovery opportunities: {total_stats.total_recovery_opportunities}")
    print(f"Recovery moves actually used: {total_stats.total_recovery_moves_used}")

    if total_stats.total_recovery_opportunities > 0:
        usage_rate = total_stats.total_recovery_moves_used / total_stats.total_recovery_opportunities * 100
        print(f"Recovery usage rate: {usage_rate:.1f}%")

    if total_stats.total_recovery_opportunities == 0:
        print()
        print("*** WARNING: No recovery opportunities detected! ***")
        print("This could indicate:")
        print("  1. Games don't reach states where players lose all stacks")
        print("  2. Players losing stacks also lose all markers")
        print("  3. Players don't have buried rings when they lose stacks")
    elif total_stats.total_recovery_moves_used == 0:
        print()
        print("*** CRITICAL: Recovery opportunities exist but NO recovery moves used! ***")
        print("This confirms the turn-skipping bug prevents recovery from working.")
    print()

    # Forced elimination stats
    print("-" * 40)
    print("FORCED ELIMINATION (RR-CANON-R100)")
    print("-" * 40)
    print(f"Games with forced elimination: {total_stats.games_with_forced_elimination}")
    print(f"Total forced elimination moves: {total_stats.total_forced_elimination_moves}")
    print(f"Potential forced elim opportunities (no_movement while having stacks): {total_stats.total_forced_elimination_opportunities}")

    if total_stats.total_forced_elimination_moves == 0:
        print()
        print("*** WARNING: No forced elimination moves detected! ***")
        print("This could indicate:")
        print("  1. AI players always have legal moves available")
        print("  2. Games end before players get boxed in")
        print("  3. Forced elimination move generation is broken")
        print()
        print("Expected forced elimination conditions:")
        print("  - Player controls at least one stack")
        print("  - Player has no legal move_stack moves (all directions blocked)")
        print("  - Player has no rings_in_hand to place")
    print()

    # Victory stats
    print("-" * 40)
    print("VICTORY TYPES")
    print("-" * 40)
    for vtype, count in total_stats.victory_types.most_common():
        pct = count / total_stats.games_analyzed * 100 if total_stats.games_analyzed > 0 else 0
        print(f"  {vtype}: {count} ({pct:.1f}%)")
    print()

    # Winner distribution
    print("-" * 40)
    print("WINNER DISTRIBUTION")
    print("-" * 40)
    for winner, count in sorted(total_stats.winners.items()):
        pct = count / total_stats.games_analyzed * 100 if total_stats.games_analyzed > 0 else 0
        label = f"Player {winner}" if winner > 0 else "Draw"
        print(f"  {label}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
