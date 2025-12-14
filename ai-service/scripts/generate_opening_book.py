#!/usr/bin/env python3
"""Generate opening book from selfplay games.

Analyzes strong games to extract common opening sequences that lead to
favorable positions. The opening book can be used to reduce early-game
variance in selfplay and training.

Usage:
    python scripts/generate_opening_book.py --games 10000
    python scripts/generate_opening_book.py --db data/games/selfplay.db --min-games 50
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def extract_opening_sequences(
    db_path: str,
    board_type: str = "square8",
    num_players: int = 2,
    max_depth: int = 10,
    min_games: int = 50,
    min_win_rate: float = 0.52,
) -> Dict[str, dict]:
    """Extract opening sequences from game database.

    Args:
        db_path: Path to game database
        board_type: Filter by board type
        num_players: Filter by number of players
        max_depth: Maximum opening depth (moves)
        min_games: Minimum games for a sequence to be included
        min_win_rate: Minimum win rate for inclusion

    Returns:
        Dict mapping move sequences to statistics
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get completed games with winners
    cursor = conn.execute("""
        SELECT game_id, winner, moves_json
        FROM games
        WHERE board_type = ?
        AND num_players = ?
        AND status = 'completed'
        AND winner IS NOT NULL
    """, (board_type, num_players))

    # Track sequences: key = move sequence as tuple, value = {wins_p0, wins_p1, total}
    sequences: Dict[Tuple, Dict] = defaultdict(lambda: {"wins": [0] * num_players, "total": 0})

    games_processed = 0
    for row in cursor:
        try:
            moves = json.loads(row["moves_json"]) if row["moves_json"] else []
            winner = row["winner"]

            # Extract opening sequences at each depth
            for depth in range(1, min(len(moves) + 1, max_depth + 1)):
                seq = tuple(str(m) for m in moves[:depth])
                sequences[seq]["total"] += 1
                if winner is not None and 0 <= winner < num_players:
                    sequences[seq]["wins"][winner] += 1

            games_processed += 1
        except (json.JSONDecodeError, TypeError):
            continue

    conn.close()

    # Filter and format results
    opening_book = {}
    for seq, stats in sequences.items():
        if stats["total"] < min_games:
            continue

        # Calculate win rate for player 0 (first player)
        p0_wins = stats["wins"][0]
        win_rate = p0_wins / stats["total"]

        # Include if win rate is notable (significantly above or below 50%)
        if abs(win_rate - 0.5) < (min_win_rate - 0.5):
            continue

        opening_book["|".join(seq)] = {
            "moves": list(seq),
            "games": stats["total"],
            "p0_wins": p0_wins,
            "p0_win_rate": round(win_rate, 4),
            "depth": len(seq),
        }

    print(f"Processed {games_processed} games, found {len(opening_book)} opening sequences")
    return opening_book


def build_opening_tree(opening_book: Dict[str, dict]) -> dict:
    """Convert flat opening book to tree structure for efficient lookup.

    Returns:
        Tree where each node has 'stats' and 'children' keys
    """
    tree = {"stats": None, "children": {}}

    for key, data in opening_book.items():
        moves = data["moves"]
        node = tree

        for move in moves:
            if move not in node["children"]:
                node["children"][move] = {"stats": None, "children": {}}
            node = node["children"][move]

        node["stats"] = {
            "games": data["games"],
            "p0_win_rate": data["p0_win_rate"],
        }

    return tree


def get_best_moves(
    tree: dict,
    current_moves: List[str],
    player: int = 0,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Get best moves from current position according to opening book.

    Args:
        tree: Opening book tree
        current_moves: Moves played so far
        player: Current player (0 or 1)
        top_k: Number of moves to return

    Returns:
        List of (move, win_rate) tuples, sorted by expected value for player
    """
    # Navigate to current position
    node = tree
    for move in current_moves:
        if move not in node["children"]:
            return []  # Position not in book
        node = node["children"][move]

    # Get children with stats
    candidates = []
    for move, child in node["children"].items():
        if child["stats"]:
            win_rate = child["stats"]["p0_win_rate"]
            # For player 1, invert win rate
            expected = win_rate if player == 0 else (1 - win_rate)
            candidates.append((move, expected, child["stats"]["games"]))

    # Sort by expected value, then by games for tiebreaker
    candidates.sort(key=lambda x: (-x[1], -x[2]))

    return [(m, rate) for m, rate, _ in candidates[:top_k]]


def main():
    parser = argparse.ArgumentParser(description="Generate opening book from selfplay")
    parser.add_argument("--db", help="Game database path")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--max-depth", type=int, default=10, help="Max opening depth")
    parser.add_argument("--min-games", type=int, default=50, help="Min games per sequence")
    parser.add_argument("--min-win-rate", type=float, default=0.52, help="Min notable win rate")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--output-tree", help="Output tree structure JSON")

    args = parser.parse_args()

    # Find database
    if args.db:
        db_path = args.db
    else:
        # Look for selfplay database
        candidates = [
            AI_SERVICE_ROOT / "data" / "games" / "selfplay.db",
            AI_SERVICE_ROOT / "data" / "games" / f"selfplay_{args.board}_{args.players}p.db",
            AI_SERVICE_ROOT / "data" / "games" / "aggregated" / "merged_selfplay.db",
        ]
        db_path = None
        for c in candidates:
            if c.exists():
                db_path = str(c)
                break

        if not db_path:
            print("No database found. Specify --db path.")
            return 1

    print(f"Generating opening book from: {db_path}")
    print(f"Board: {args.board}, Players: {args.players}")
    print(f"Max depth: {args.max_depth}, Min games: {args.min_games}")
    print()

    # Extract sequences
    opening_book = extract_opening_sequences(
        db_path=db_path,
        board_type=args.board,
        num_players=args.players,
        max_depth=args.max_depth,
        min_games=args.min_games,
        min_win_rate=args.min_win_rate,
    )

    if not opening_book:
        print("No opening sequences found matching criteria.")
        return 1

    # Print top openings
    print("\nTop 20 Opening Sequences (by games):")
    print("-" * 60)
    sorted_openings = sorted(opening_book.values(), key=lambda x: -x["games"])[:20]
    for i, data in enumerate(sorted_openings, 1):
        moves_str = " -> ".join(data["moves"][:5])
        if len(data["moves"]) > 5:
            moves_str += " ..."
        print(f"{i:2}. {moves_str}")
        print(f"    Games: {data['games']}, P0 win rate: {data['p0_win_rate']:.1%}")

    # Save flat book
    output_path = args.output or AI_SERVICE_ROOT / "data" / f"opening_book_{args.board}_{args.players}p.json"
    with open(output_path, "w") as f:
        json.dump({
            "board_type": args.board,
            "num_players": args.players,
            "max_depth": args.max_depth,
            "min_games": args.min_games,
            "total_sequences": len(opening_book),
            "sequences": opening_book,
        }, f, indent=2)
    print(f"\nOpening book saved to: {output_path}")

    # Build and save tree structure
    if args.output_tree:
        tree = build_opening_tree(opening_book)
        with open(args.output_tree, "w") as f:
            json.dump(tree, f, indent=2)
        print(f"Opening tree saved to: {args.output_tree}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
