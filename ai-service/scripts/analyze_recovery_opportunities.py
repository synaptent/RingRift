#!/usr/bin/env python3
"""
Analyze selfplay games to detect recovery opportunity windows.

This script replays games from JSONL files and detects when players meet
the recovery eligibility requirements (RR-CANON-R110):
1. Controls no stacks (controlling_player != player_number for all stacks)
2. Has at least one marker on the board
3. Has at least one buried ring (ring in a stack where cap != player)

It then checks whether those players ever got a turn where recovery moves
would have been available.

Usage:
    python scripts/analyze_recovery_opportunities.py [--dir PATH] [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoardState:
    """Simplified board state for tracking recovery eligibility."""
    # Stack data: {position_key: {"rings": [player_numbers], "controller": player}}
    stacks: dict = field(default_factory=dict)
    # Markers: {position_key: player_number}
    markers: dict = field(default_factory=dict)
    # Players with rings in hand
    rings_in_hand: dict = field(default_factory=dict)
    # Canonical starting rings per player for this board type.
    # Used as a fallback when initial_state omits rings_in_hand.
    default_rings_per_player: int = 18

    def player_controls_any_stack(self, player: int) -> bool:
        """Check if player controls at least one stack."""
        for stack in self.stacks.values():
            if stack.get("controller") == player:
                return True
        return False

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
        """Check if player meets all recovery eligibility conditions."""
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

@dataclass
class RecoveryOpportunity:
    """Record of a missed recovery opportunity."""
    game_index: int
    move_number: int
    player: int
    controlled_stacks: int
    markers: int
    buried_rings: int
    rings_in_hand: int
    got_turn_after: bool = False
    recovery_move_offered: bool = False


@dataclass
class AnalysisStats:
    """Aggregated statistics from analysis."""
    games_analyzed: int = 0
    games_with_recovery_opportunities: int = 0
    total_recovery_opportunities: int = 0
    opportunities_where_player_got_turn: int = 0
    opportunities_where_recovery_offered: int = 0
    opportunities_by_player: dict = field(default_factory=dict)


def parse_position_key(pos: dict) -> str:
    """Convert position dict to string key."""
    x = pos.get("x", pos.get("col", 0))
    y = pos.get("y", pos.get("row", 0))
    return f"{x},{y}"


def apply_move_to_board(board: BoardState, move: dict, num_players: int):
    """Apply a move to update the board state."""
    move_type = move.get("type", "")
    player = move.get("player", 0)

    if move_type == "place_ring":
        to_pos = move.get("to")
        if to_pos:
            key = parse_position_key(to_pos)
            # Canonical placement defaults to 1 ring but may include a placement_count.
            placement_count = move.get(
                "placement_count",
                move.get("placementCount", 1),
            )
            if key not in board.stacks:
                board.stacks[key] = {"rings": [], "controller": player}
            board.stacks[key]["rings"] = [player] * placement_count
            board.stacks[key]["controller"] = player

            # Reduce rings in hand
            board.rings_in_hand[player] = max(
                0,
                board.rings_in_hand.get(player, board.default_rings_per_player) - placement_count,
            )

            # Leave marker at placement position
            board.markers[key] = player

    elif move_type == "move_stack":
        from_pos = move.get("from_pos") or move.get("from")
        to_pos = move.get("to")

        if from_pos and to_pos:
            from_key = parse_position_key(from_pos)
            to_key = parse_position_key(to_pos)

            # Get the moving stack
            moving_stack = board.stacks.get(from_key, {"rings": [], "controller": 0})

            # Leave marker at from position
            board.markers[from_key] = player

            # Remove stack from source
            if from_key in board.stacks:
                del board.stacks[from_key]

            # Handle destination
            if to_key in board.stacks:
                # Overtaking - moving stack goes on top
                dest_stack = board.stacks[to_key]
                combined_rings = dest_stack["rings"] + moving_stack["rings"]
                board.stacks[to_key] = {
                    "rings": combined_rings,
                    "controller": player  # Mover controls the combined stack
                }
            else:
                # Moving to empty space
                board.stacks[to_key] = {
                    "rings": moving_stack["rings"],
                    "controller": player
                }

    elif move_type == "recovery_slide":
        # This is what we're looking for - if this exists, recovery worked!
        pass  # The actual move application is similar to move_stack

    # Handle other move types as needed (no_line_action, no_territory_action, etc.)


def replay_game_and_find_recovery_opportunities(
    game: dict, game_index: int
) -> list[RecoveryOpportunity]:
    """Replay a game and find all recovery opportunity windows."""
    opportunities = []

    moves = game.get("moves", [])
    board_type_str = game.get("board_type", "square8")
    num_players = game.get("num_players", 2)
    initial_state = game.get("initial_state", {})

    # Initialize board state
    board = BoardState()

    # Best-effort canonical rings-per-player fallback for this game.
    try:
        from app.models import BoardType
        from app.rules.core import get_rings_per_player

        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex": BoardType.HEXAGONAL,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(board_type_str, BoardType.SQUARE8)
        board.default_rings_per_player = get_rings_per_player(board_type)
    except Exception:
        # Keep the dataclass default when imports fail (e.g., minimal env).
        pass

    # Set initial rings in hand
    for p in range(1, num_players + 1):
        # Default to canonical starting rings; prefer initial_state player payload
        # when available (it reflects any non-standard start positions).
        board.rings_in_hand[p] = board.default_rings_per_player

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
            board.rings_in_hand[pnum] = pdata.get("rings_in_hand", board.default_rings_per_player)

    # Track when each player becomes recovery-eligible
    recovery_eligible_since: dict[int, int] = {}  # player -> move_number

    for i, move in enumerate(moves):
        move_type = move.get("type", "")
        move_player = move.get("player", 0)

        # Check all players for recovery eligibility BEFORE the move
        for p in range(1, num_players + 1):
            was_eligible = p in recovery_eligible_since
            is_eligible = board.is_eligible_for_recovery(p)

            if is_eligible and not was_eligible:
                # Player just became eligible
                recovery_eligible_since[p] = i

                # Record the opportunity
                opp = RecoveryOpportunity(
                    game_index=game_index,
                    move_number=i,
                    player=p,
                    controlled_stacks=sum(1 for s in board.stacks.values() if s.get("controller") == p),
                    markers=sum(1 for m in board.markers.values() if m == p),
                    buried_rings=board.count_buried_rings(p),
                    rings_in_hand=board.rings_in_hand.get(p, 0)
                )
                opportunities.append(opp)

            elif was_eligible and not is_eligible:
                # Player is no longer eligible
                del recovery_eligible_since[p]

        # Check if the current move is by a recovery-eligible player
        if move_player in recovery_eligible_since:
            # Find the matching opportunity and mark it
            for opp in opportunities:
                if opp.player == move_player and opp.move_number == recovery_eligible_since[move_player]:
                    opp.got_turn_after = True
                    if move_type == "recovery_slide":
                        opp.recovery_move_offered = True

        # Apply the move
        apply_move_to_board(board, move, num_players)

    return opportunities


def analyze_jsonl_file(filepath: Path, limit: int | None = None) -> AnalysisStats:
    """Analyze a single JSONL file."""
    stats = AnalysisStats()

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            if limit and line_num >= limit:
                break

            try:
                game = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats.games_analyzed += 1

            opportunities = replay_game_and_find_recovery_opportunities(
                game, line_num
            )

            if opportunities:
                stats.games_with_recovery_opportunities += 1
                stats.total_recovery_opportunities += len(opportunities)

                for opp in opportunities:
                    if opp.got_turn_after:
                        stats.opportunities_where_player_got_turn += 1
                    if opp.recovery_move_offered:
                        stats.opportunities_where_recovery_offered += 1

                    # Track by player
                    p = opp.player
                    if p not in stats.opportunities_by_player:
                        stats.opportunities_by_player[p] = {
                            "total": 0,
                            "got_turn": 0,
                            "recovery_offered": 0
                        }
                    stats.opportunities_by_player[p]["total"] += 1
                    if opp.got_turn_after:
                        stats.opportunities_by_player[p]["got_turn"] += 1
                    if opp.recovery_move_offered:
                        stats.opportunities_by_player[p]["recovery_offered"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze selfplay games for recovery opportunities"
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

    for jf in jsonl_files:
        print(f"Analyzing: {jf.relative_to(data_dir)}")
        stats = analyze_jsonl_file(jf, args.limit)

        # Merge stats
        total_stats.games_analyzed += stats.games_analyzed
        total_stats.games_with_recovery_opportunities += stats.games_with_recovery_opportunities
        total_stats.total_recovery_opportunities += stats.total_recovery_opportunities
        total_stats.opportunities_where_player_got_turn += stats.opportunities_where_player_got_turn
        total_stats.opportunities_where_recovery_offered += stats.opportunities_where_recovery_offered

        for p, pstats in stats.opportunities_by_player.items():
            if p not in total_stats.opportunities_by_player:
                total_stats.opportunities_by_player[p] = {
                    "total": 0, "got_turn": 0, "recovery_offered": 0
                }
            total_stats.opportunities_by_player[p]["total"] += pstats["total"]
            total_stats.opportunities_by_player[p]["got_turn"] += pstats["got_turn"]
            total_stats.opportunities_by_player[p]["recovery_offered"] += pstats["recovery_offered"]

        if args.verbose:
            print(f"  Games: {stats.games_analyzed}")
            print(f"  With recovery opportunities: {stats.games_with_recovery_opportunities}")
            print(f"  Total opportunities: {stats.total_recovery_opportunities}")
            print()

    # Print summary
    print()
    print("=" * 60)
    print("RECOVERY OPPORTUNITY ANALYSIS SUMMARY")
    print("=" * 60)
    print()
    print(f"Total games analyzed: {total_stats.games_analyzed}")
    print(f"Games with recovery opportunities: {total_stats.games_with_recovery_opportunities}")
    print(f"Total recovery opportunities detected: {total_stats.total_recovery_opportunities}")
    print()

    if total_stats.total_recovery_opportunities > 0:
        pct_got_turn = (
            total_stats.opportunities_where_player_got_turn /
            total_stats.total_recovery_opportunities * 100
        )
        pct_recovery = (
            total_stats.opportunities_where_recovery_offered /
            total_stats.total_recovery_opportunities * 100
        )

        print(f"Opportunities where player got a turn: {total_stats.opportunities_where_player_got_turn} ({pct_got_turn:.1f}%)")
        print(f"Opportunities where recovery move offered: {total_stats.opportunities_where_recovery_offered} ({pct_recovery:.1f}%)")
        print()

        if total_stats.opportunities_where_player_got_turn == 0:
            print("*** CRITICAL: Players meeting recovery conditions NEVER got turns! ***")
            print("This confirms the turn-skipping bug documented in the plan.")
        elif total_stats.opportunities_where_recovery_offered == 0:
            print("*** WARNING: Recovery-eligible players got turns but NO recovery moves offered ***")
            print("This suggests a bug in recovery move generation.")

        print()
        print("Breakdown by player:")
        for p, pstats in sorted(total_stats.opportunities_by_player.items()):
            print(f"  Player {p}:")
            print(f"    Total opportunities: {pstats['total']}")
            print(f"    Got turn: {pstats['got_turn']}")
            print(f"    Recovery offered: {pstats['recovery_offered']}")
    else:
        print("No recovery opportunities detected in any games.")
        print()
        print("Possible reasons:")
        print("1. Games end before players lose all their stacks")
        print("2. Captures (overtaking) are rare in the data")
        print("3. When players lose stacks, they also lose all markers")


if __name__ == "__main__":
    main()
