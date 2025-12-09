#!/usr/bin/env python
"""Analyze game replays from the database using algebraic notation.

This tool provides various analysis functions for game replays:
- Export games to PGN format for human review
- Analyze opening patterns (first N moves)
- Compute move type statistics
- Identify common position patterns
- Find games by criteria (winner, length, termination, etc.)

Usage:
    # Export games to PGN
    python scripts/analyze_games.py export-pgn --db data/games/selfplay.db --output games.pgn

    # Analyze opening patterns
    python scripts/analyze_games.py openings --db data/games/selfplay.db --depth 10

    # Show move statistics
    python scripts/analyze_games.py stats --db data/games/selfplay.db

    # List games matching criteria
    python scripts/analyze_games.py list --db data/games/selfplay.db --winner 1 --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import GameReplayDB
from app.models import BoardType, GameState, Move, MoveType
from app.notation import (
    game_to_pgn,
    move_to_algebraic,
    moves_to_notation_list,
    MOVE_TYPE_TO_CODE,
)


BOARD_TYPE_MAP = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hex": BoardType.HEXAGONAL,
    "hexagonal": BoardType.HEXAGONAL,
}

BOARD_TYPE_REVERSE = {
    BoardType.SQUARE8: "square8",
    BoardType.SQUARE19: "square19",
    BoardType.HEXAGONAL: "hexagonal",
}


@dataclass
class GameSummary:
    """Summary of a single game."""

    game_id: str
    board_type: str
    winner: int
    total_moves: int
    termination_reason: str
    p1_rings_eliminated: int
    p2_rings_eliminated: int
    opening_sequence: List[str] = field(default_factory=list)


@dataclass
class OpeningPattern:
    """A common opening pattern across games."""

    sequence: Tuple[str, ...]
    count: int
    wins_for_player1: int
    wins_for_player2: int

    @property
    def win_rate_p1(self) -> float:
        if self.count == 0:
            return 0.0
        return self.wins_for_player1 / self.count

    @property
    def win_rate_p2(self) -> float:
        if self.count == 0:
            return 0.0
        return self.wins_for_player2 / self.count


def load_games_from_db(
    db_path: str,
    board_type: Optional[str] = None,
    winner: Optional[int] = None,
    limit: int = 1000,
) -> Iterator[Tuple[dict, GameState, List[Move]]]:
    """Load games from database with optional filtering."""
    db = GameReplayDB(db_path)

    filters: Dict[str, Any] = {}
    if board_type:
        filters["board_type"] = BOARD_TYPE_MAP.get(board_type)
    if winner is not None:
        filters["winner"] = winner

    count = 0
    for game_meta, initial_state, moves in db.iterate_games(**filters):
        if count >= limit:
            break
        yield game_meta, initial_state, moves
        count += 1


def get_board_type_from_meta(game_meta: dict) -> BoardType:
    """Extract BoardType from game metadata."""
    bt_str = game_meta.get("board_type", "square8")
    if isinstance(bt_str, BoardType):
        return bt_str
    return BOARD_TYPE_MAP.get(bt_str, BoardType.SQUARE8)


# =============================================================================
# Export to PGN
# =============================================================================


def export_games_to_pgn(
    db_path: str,
    output_path: str,
    board_type: Optional[str] = None,
    limit: int = 100,
) -> int:
    """Export games from database to PGN format.

    Returns:
        Number of games exported
    """
    exported = 0

    with open(output_path, "w") as f:
        for game_meta, initial_state, moves in load_games_from_db(db_path, board_type, limit=limit):
            bt = get_board_type_from_meta(game_meta)

            metadata = {
                "game_id": game_meta.get("game_id", "unknown"),
                "board": BOARD_TYPE_REVERSE.get(bt, "square8"),
                "date": game_meta.get("created_at", datetime.now().isoformat())[:10],
                "player1": "AI-1",
                "player2": "AI-2",
                "winner": game_meta.get("winner"),
                "termination": game_meta.get("termination_reason", "unknown"),
                "rng_seed": game_meta.get("rng_seed"),
            }

            pgn = game_to_pgn(moves, metadata, bt)
            f.write(pgn)
            f.write("\n\n")
            exported += 1

    return exported


# =============================================================================
# Opening Analysis
# =============================================================================


def analyze_openings(
    db_path: str,
    depth: int = 10,
    board_type: Optional[str] = None,
    min_frequency: int = 2,
    limit: int = 1000,
) -> List[OpeningPattern]:
    """Analyze common opening sequences.

    Args:
        db_path: Path to game database
        depth: Number of moves to consider as "opening"
        board_type: Optional board type filter
        min_frequency: Minimum occurrences to report
        limit: Maximum games to analyze

    Returns:
        List of OpeningPattern objects sorted by frequency
    """
    # Track opening sequences with outcomes
    openings: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: {"count": 0, "p1_wins": 0, "p2_wins": 0})

    for game_meta, initial_state, moves in load_games_from_db(db_path, board_type, limit=limit):
        bt = get_board_type_from_meta(game_meta)
        winner = game_meta.get("winner", 0)

        # Get opening moves in notation
        opening_moves = moves[:depth]
        notation = tuple(moves_to_notation_list(opening_moves, bt))

        openings[notation]["count"] += 1
        if winner == 1:
            openings[notation]["p1_wins"] += 1
        elif winner == 2:
            openings[notation]["p2_wins"] += 1

    # Convert to OpeningPattern objects
    patterns = []
    for seq, stats in openings.items():
        if stats["count"] >= min_frequency:
            patterns.append(
                OpeningPattern(
                    sequence=seq,
                    count=stats["count"],
                    wins_for_player1=stats["p1_wins"],
                    wins_for_player2=stats["p2_wins"],
                )
            )

    # Sort by frequency
    patterns.sort(key=lambda p: -p.count)
    return patterns


# =============================================================================
# Move Statistics
# =============================================================================


@dataclass
class MoveStats:
    """Statistics about moves in analyzed games."""

    total_games: int
    total_moves: int
    avg_game_length: float
    move_type_counts: Dict[str, int]
    move_type_by_phase: Dict[str, Dict[str, int]]  # phase -> move_type -> count
    win_rate_by_first_move: Dict[str, Tuple[int, int, int]]  # move -> (total, p1_wins, p2_wins)
    position_frequency: Dict[str, int]  # position -> count (for placements)


def compute_move_stats(
    db_path: str,
    board_type: Optional[str] = None,
    limit: int = 1000,
) -> MoveStats:
    """Compute statistics about moves across games."""
    total_games = 0
    total_moves = 0
    move_type_counts: Counter = Counter()
    position_freq: Counter = Counter()
    first_moves: Dict[str, List[int]] = defaultdict(list)  # notation -> [winners]

    for game_meta, initial_state, moves in load_games_from_db(db_path, board_type, limit=limit):
        bt = get_board_type_from_meta(game_meta)
        winner = game_meta.get("winner", 0)
        total_games += 1
        total_moves += len(moves)

        for i, move in enumerate(moves):
            # Count move types
            code = MOVE_TYPE_TO_CODE.get(move.type, "?")
            move_type_counts[code] += 1

            # Track position frequency for placements
            if move.type == MoveType.PLACE_RING:
                pos_key = f"{move.to.x},{move.to.y}"
                position_freq[pos_key] += 1

            # Track first move outcomes
            if i == 0:
                notation = move_to_algebraic(move, bt)
                first_moves[notation].append(winner)

    # Compute first move win rates
    win_rate_by_first_move = {}
    for notation, winners in first_moves.items():
        total = len(winners)
        p1_wins = sum(1 for w in winners if w == 1)
        p2_wins = sum(1 for w in winners if w == 2)
        win_rate_by_first_move[notation] = (total, p1_wins, p2_wins)

    avg_length = total_moves / total_games if total_games > 0 else 0

    return MoveStats(
        total_games=total_games,
        total_moves=total_moves,
        avg_game_length=avg_length,
        move_type_counts=dict(move_type_counts),
        move_type_by_phase={},  # Could extend with phase tracking
        win_rate_by_first_move=win_rate_by_first_move,
        position_frequency=dict(position_freq.most_common(20)),
    )


# =============================================================================
# Game Listing
# =============================================================================


def list_games(
    db_path: str,
    board_type: Optional[str] = None,
    winner: Optional[int] = None,
    min_moves: Optional[int] = None,
    max_moves: Optional[int] = None,
    limit: int = 20,
) -> List[GameSummary]:
    """List games matching criteria with summary info."""
    db = GameReplayDB(db_path)

    filters: Dict[str, Any] = {}
    if board_type:
        filters["board_type"] = BOARD_TYPE_MAP.get(board_type)
    if winner is not None:
        filters["winner"] = winner
    if min_moves is not None:
        filters["min_moves"] = min_moves
    if max_moves is not None:
        filters["max_moves"] = max_moves

    summaries = []
    count = 0

    for game_meta, initial_state, moves in db.iterate_games(**filters):
        if count >= limit:
            break

        bt = get_board_type_from_meta(game_meta)

        # Get opening sequence in notation
        opening = moves_to_notation_list(moves[:6], bt)

        summaries.append(
            GameSummary(
                game_id=game_meta.get("game_id", "unknown"),
                board_type=BOARD_TYPE_REVERSE.get(bt, "square8"),
                winner=game_meta.get("winner", 0),
                total_moves=len(moves),
                termination_reason=game_meta.get("termination_reason", "unknown"),
                p1_rings_eliminated=game_meta.get("p1_eliminated_rings", 0) or 0,
                p2_rings_eliminated=game_meta.get("p2_eliminated_rings", 0) or 0,
                opening_sequence=opening,
            )
        )
        count += 1

    return summaries


# =============================================================================
# Pattern Detection
# =============================================================================


def find_pattern_games(
    db_path: str,
    pattern: List[str],
    board_type: Optional[str] = None,
    limit: int = 100,
) -> List[str]:
    """Find games containing a specific move pattern (subsequence).

    Args:
        db_path: Path to game database
        pattern: List of notation strings to match (e.g., ["P d4", "P e5"])
        board_type: Optional board type filter
        limit: Maximum games to search

    Returns:
        List of game IDs containing the pattern
    """
    matching_games = []
    pattern_len = len(pattern)

    for game_meta, initial_state, moves in load_games_from_db(db_path, board_type, limit=limit):
        bt = get_board_type_from_meta(game_meta)
        notation_list = moves_to_notation_list(moves, bt)

        # Search for pattern as subsequence
        for i in range(len(notation_list) - pattern_len + 1):
            if notation_list[i : i + pattern_len] == pattern:
                matching_games.append(game_meta.get("game_id", "unknown"))
                break

    return matching_games


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_export_pgn(args):
    """Export games to PGN format."""
    count = export_games_to_pgn(
        args.db,
        args.output,
        board_type=args.board,
        limit=args.limit,
    )
    print(f"Exported {count} games to {args.output}")


def cmd_openings(args):
    """Analyze opening patterns."""
    patterns = analyze_openings(
        args.db,
        depth=args.depth,
        board_type=args.board,
        min_frequency=args.min_freq,
        limit=args.limit,
    )

    print(f"\nOpening Analysis (depth={args.depth}, min_freq={args.min_freq})")
    print("=" * 80)

    for i, p in enumerate(patterns[: args.top], 1):
        seq_str = " ".join(p.sequence[:5])
        if len(p.sequence) > 5:
            seq_str += " ..."
        print(f"\n{i}. [{p.count} games] {seq_str}")
        print(f"   P1 wins: {p.wins_for_player1} ({p.win_rate_p1*100:.1f}%)")
        print(f"   P2 wins: {p.wins_for_player2} ({p.win_rate_p2*100:.1f}%)")


def cmd_stats(args):
    """Show move statistics."""
    stats = compute_move_stats(
        args.db,
        board_type=args.board,
        limit=args.limit,
    )

    print(f"\nMove Statistics")
    print("=" * 60)
    print(f"Total games analyzed: {stats.total_games}")
    print(f"Total moves: {stats.total_moves}")
    print(f"Average game length: {stats.avg_game_length:.1f} moves")

    print(f"\nMove Type Distribution:")
    for code, count in sorted(stats.move_type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / stats.total_moves if stats.total_moves > 0 else 0
        print(f"  {code:4s}: {count:6d} ({pct:5.1f}%)")

    print(f"\nTop Placement Positions:")
    for pos, count in list(stats.position_frequency.items())[:10]:
        print(f"  {pos}: {count}")

    if stats.win_rate_by_first_move:
        print(f"\nFirst Move Win Rates (top 10 by frequency):")
        sorted_first = sorted(stats.win_rate_by_first_move.items(), key=lambda x: -x[1][0])[:10]
        for notation, (total, p1, p2) in sorted_first:
            p1_rate = 100 * p1 / total if total > 0 else 0
            p2_rate = 100 * p2 / total if total > 0 else 0
            print(f"  {notation:12s}: {total:3d} games, P1 wins {p1_rate:.0f}%, P2 wins {p2_rate:.0f}%")


def cmd_list(args):
    """List games matching criteria."""
    games = list_games(
        args.db,
        board_type=args.board,
        winner=args.winner,
        min_moves=args.min_moves,
        max_moves=args.max_moves,
        limit=args.limit,
    )

    print(f"\nGames Found: {len(games)}")
    print("=" * 100)

    for g in games:
        winner_str = f"P{g.winner}" if g.winner else "draw"
        opening = " ".join(g.opening_sequence[:4])
        print(
            f"{g.game_id[:8]}... | {g.board_type:8s} | {g.total_moves:3d} moves | "
            f"{winner_str:5s} | {g.termination_reason:20s} | {opening}"
        )


def cmd_export_game(args):
    """Export a single game to PGN by ID."""
    db = GameReplayDB(args.db)

    initial_state = db.get_initial_state(args.game_id)
    if initial_state is None:
        print(f"Game not found: {args.game_id}")
        return

    moves = db.get_moves(args.game_id)

    # Get metadata - query for this specific game
    games = db.query_games(limit=10000)
    game_meta = None
    for g in games:
        if g["game_id"] == args.game_id:
            game_meta = g
            break

    if not game_meta:
        print(f"Game metadata not found: {args.game_id}")
        return

    bt = get_board_type_from_meta(game_meta)

    metadata = {
        "game_id": args.game_id,
        "board": BOARD_TYPE_REVERSE.get(bt, "square8"),
        "date": game_meta.get("created_at", datetime.now().isoformat())[:10],
        "player1": "AI-1",
        "player2": "AI-2",
        "winner": game_meta.get("winner"),
        "termination": game_meta.get("termination_reason", "unknown"),
    }

    pgn = game_to_pgn(moves, metadata, bt)

    if args.output:
        with open(args.output, "w") as f:
            f.write(pgn)
        print(f"Exported to {args.output}")
    else:
        print(pgn)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze game replays from database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # export-pgn command
    p_export = subparsers.add_parser("export-pgn", help="Export games to PGN format")
    p_export.add_argument("--db", required=True, help="Path to game database")
    p_export.add_argument("--output", "-o", required=True, help="Output PGN file path")
    p_export.add_argument("--board", choices=["square8", "square19", "hex"], help="Filter by board type")
    p_export.add_argument("--limit", type=int, default=100, help="Max games to export")
    p_export.set_defaults(func=cmd_export_pgn)

    # openings command
    p_open = subparsers.add_parser("openings", help="Analyze opening patterns")
    p_open.add_argument("--db", required=True, help="Path to game database")
    p_open.add_argument("--depth", type=int, default=10, help="Opening depth (moves)")
    p_open.add_argument("--board", choices=["square8", "square19", "hex"], help="Filter by board type")
    p_open.add_argument("--min-freq", type=int, default=2, help="Min pattern frequency")
    p_open.add_argument("--top", type=int, default=20, help="Number of patterns to show")
    p_open.add_argument("--limit", type=int, default=1000, help="Max games to analyze")
    p_open.set_defaults(func=cmd_openings)

    # stats command
    p_stats = subparsers.add_parser("stats", help="Show move statistics")
    p_stats.add_argument("--db", required=True, help="Path to game database")
    p_stats.add_argument("--board", choices=["square8", "square19", "hex"], help="Filter by board type")
    p_stats.add_argument("--limit", type=int, default=1000, help="Max games to analyze")
    p_stats.set_defaults(func=cmd_stats)

    # list command
    p_list = subparsers.add_parser("list", help="List games matching criteria")
    p_list.add_argument("--db", required=True, help="Path to game database")
    p_list.add_argument("--board", choices=["square8", "square19", "hex"], help="Filter by board type")
    p_list.add_argument("--winner", type=int, choices=[1, 2], help="Filter by winner")
    p_list.add_argument("--min-moves", type=int, help="Minimum game length")
    p_list.add_argument("--max-moves", type=int, help="Maximum game length")
    p_list.add_argument("--limit", type=int, default=20, help="Max games to list")
    p_list.set_defaults(func=cmd_list)

    # export-game command
    p_game = subparsers.add_parser("export-game", help="Export single game by ID")
    p_game.add_argument("--db", required=True, help="Path to game database")
    p_game.add_argument("--game-id", required=True, help="Game ID to export")
    p_game.add_argument("--output", "-o", help="Output file (stdout if not specified)")
    p_game.set_defaults(func=cmd_export_game)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
