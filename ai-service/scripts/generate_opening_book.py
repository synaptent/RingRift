#!/usr/bin/env python3
"""Generate opening book from selfplay games.

Analyzes strong games to extract common opening sequences that lead to
favorable positions. The opening book can be used to reduce early-game
variance in selfplay and training.

Features:
    - Extract openings from high-Elo games
    - Weight moves by Elo and win rate
    - Multi-database support (glob patterns)
    - Export in selfplay-optimized format

Usage:
    # Generate from single database
    python scripts/generate_opening_book.py --db data/games/selfplay.db --min-games 50

    # Generate from multiple databases with Elo filter
    python scripts/generate_opening_book.py --db "data/games/*.db" --min-elo 1600

    # View stats for existing book
    python scripts/generate_opening_book.py --book data/opening_book.json --stats

    # Export for selfplay
    python scripts/generate_opening_book.py --book data/opening_book.json --export-selfplay data/openings_flat.json
"""

import argparse
import glob as glob_module
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OpeningNode:
    """Node in the opening book tree with Elo-weighted statistics."""
    move: str
    count: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    total_elo: float = 0.0
    children: dict[str, "OpeningNode"] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.draws + self.losses
        if total == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / total

    @property
    def avg_elo(self) -> float:
        return self.total_elo / self.count if self.count > 0 else 1500.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "move": self.move,
            "count": self.count,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "avg_elo": round(self.avg_elo, 1),
            "win_rate": round(self.win_rate, 4),
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpeningNode":
        node = cls(
            move=data["move"],
            count=data["count"],
            wins=data["wins"],
            draws=data.get("draws", 0),
            losses=data["losses"],
            total_elo=data.get("avg_elo", 1500.0) * data["count"],
        )
        for k, v in data.get("children", {}).items():
            node.children[k] = cls.from_dict(v)
        return node


@dataclass
class OpeningBook:
    """Complete opening book with tree structure and metadata."""
    board_type: str
    num_players: int
    max_depth: int
    min_games: int
    root: OpeningNode = field(default_factory=lambda: OpeningNode(move="root"))
    total_games: int = 0
    source_databases: list[str] = field(default_factory=list)

    def add_game(
        self,
        moves: list[str],
        winner: int | None,
        player_elo: float = 1500.0,
    ):
        """Add a game's opening to the book."""
        self.total_games += 1
        node = self.root
        current_player = 0

        for _i, move in enumerate(moves[:self.max_depth]):
            move_str = str(move)
            if move_str not in node.children:
                node.children[move_str] = OpeningNode(move=move_str)

            child = node.children[move_str]
            child.count += 1
            child.total_elo += player_elo

            if winner is None:
                child.draws += 1
            elif winner == current_player:
                child.wins += 1
            else:
                child.losses += 1

            node = child
            current_player = (current_player + 1) % self.num_players

    def prune(self, min_count: int = None):
        """Remove moves played fewer than min_count times."""
        min_count = min_count or self.min_games

        def prune_node(node: OpeningNode) -> bool:
            to_remove = [m for m, c in node.children.items() if c.count < min_count]
            for move in to_remove:
                del node.children[move]
            for child in node.children.values():
                prune_node(child)
            return len(node.children) > 0 or node.count >= min_count

        prune_node(self.root)

    def get_moves_for_position(
        self,
        position_moves: list[str],
        temperature: float = 1.0,
        min_win_rate: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Get weighted candidate moves for a position."""
        node = self.root
        for move in position_moves:
            if str(move) in node.children:
                node = node.children[str(move)]
            else:
                return []

        candidates = []
        for move, child in node.children.items():
            if child.win_rate >= min_win_rate:
                weight = (child.count ** 0.5) * (child.win_rate ** 2) * (child.avg_elo / 1500.0) ** 0.5
                if temperature != 1.0:
                    weight = weight ** (1.0 / temperature)
                candidates.append((move, weight))

        total = sum(w for _, w in candidates)
        if total > 0:
            candidates = [(m, w / total) for m, w in candidates]
        return sorted(candidates, key=lambda x: -x[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_type": self.board_type,
            "num_players": self.num_players,
            "max_depth": self.max_depth,
            "min_games": self.min_games,
            "total_games": self.total_games,
            "source_databases": self.source_databases,
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpeningBook":
        book = cls(
            board_type=data["board_type"],
            num_players=data["num_players"],
            max_depth=data["max_depth"],
            min_games=data["min_games"],
            total_games=data["total_games"],
            source_databases=data.get("source_databases", []),
        )
        book.root = OpeningNode.from_dict(data["root"])
        return book

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved opening book to {path}")

    @classmethod
    def load(cls, path: str) -> "OpeningBook":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def extract_games_from_db(
    db_path: str,
    board_type: str,
    num_players: int,
    min_elo: float = 0.0,
) -> list[dict[str, Any]]:
    """Extract games from various database schemas."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    games = []

    # Try standard games table
    try:
        cursor = conn.execute("""
            SELECT game_id, winner, moves_json, player_elos
            FROM games
            WHERE board_type = ? AND num_players = ?
            AND status = 'completed'
        """, (board_type, num_players))
        for row in cursor:
            elos = json.loads(row["player_elos"]) if row["player_elos"] else [1500.0]
            avg_elo = sum(elos) / len(elos) if elos else 1500.0
            if avg_elo >= min_elo:
                games.append({
                    "moves": json.loads(row["moves_json"]) if row["moves_json"] else [],
                    "winner": row["winner"],
                    "avg_elo": avg_elo,
                })
    except (sqlite3.OperationalError, json.JSONDecodeError):
        pass

    # Try selfplay_games table
    try:
        cursor = conn.execute("""
            SELECT game_id, winner, moves, model_elo
            FROM selfplay_games
            WHERE board_type = ? AND num_players = ? AND model_elo >= ?
        """, (board_type, num_players, min_elo))
        for row in cursor:
            moves = row["moves"]
            if isinstance(moves, str):
                moves = json.loads(moves)
            games.append({
                "moves": moves or [],
                "winner": row["winner"],
                "avg_elo": row["model_elo"] or 1500.0,
            })
    except (sqlite3.OperationalError, json.JSONDecodeError):
        pass

    # Try game_results table
    try:
        cursor = conn.execute("""
            SELECT game_id, winner, action_history, elo
            FROM game_results
            WHERE config LIKE ? AND elo >= ?
        """, (f"{board_type}_{num_players}p%", min_elo))
        for row in cursor:
            history = row["action_history"]
            if isinstance(history, str):
                history = json.loads(history)
            moves = [str(a.get("move", a)) for a in history] if history else []
            games.append({
                "moves": moves,
                "winner": row["winner"],
                "avg_elo": row["elo"] or 1500.0,
            })
    except (sqlite3.OperationalError, json.JSONDecodeError):
        pass

    conn.close()
    return games


def generate_book_from_databases(
    db_paths: list[str],
    board_type: str,
    num_players: int,
    max_depth: int = 15,
    min_games: int = 10,
    min_elo: float = 1500.0,
) -> OpeningBook:
    """Generate opening book from multiple databases."""
    book = OpeningBook(
        board_type=board_type,
        num_players=num_players,
        max_depth=max_depth,
        min_games=min_games,
        source_databases=db_paths,
    )

    total_processed = 0
    total_skipped = 0

    for db_path in db_paths:
        if not Path(db_path).exists():
            print(f"Warning: Database not found: {db_path}")
            continue

        print(f"Processing {db_path}...")
        games = extract_games_from_db(db_path, board_type, num_players, min_elo)

        for game in games:
            moves = game.get("moves", [])
            if not moves or len(moves) < 3:
                total_skipped += 1
                continue

            book.add_game(moves, game.get("winner"), game.get("avg_elo", 1500.0))
            total_processed += 1

    print(f"Processed {total_processed} games, skipped {total_skipped}")
    book.prune(min_games)
    return book


def export_for_selfplay(book: OpeningBook, output_path: str):
    """Export opening book as flat list for selfplay sampling."""
    openings = []

    def collect_sequences(node: OpeningNode, current_seq: list[str], depth: int):
        if depth >= book.max_depth or not node.children:
            if current_seq:
                openings.append({
                    "moves": current_seq.copy(),
                    "weight": node.count * node.win_rate * (node.avg_elo / 1500.0),
                    "count": node.count,
                    "win_rate": node.win_rate,
                })
            return
        for move, child in node.children.items():
            current_seq.append(move)
            collect_sequences(child, current_seq, depth + 1)
            current_seq.pop()

    collect_sequences(book.root, [], 0)

    total_weight = sum(o["weight"] for o in openings)
    if total_weight > 0:
        for o in openings:
            o["weight"] /= total_weight

    openings.sort(key=lambda x: -x["weight"])

    output = {
        "board_type": book.board_type,
        "num_players": book.num_players,
        "total_openings": len(openings),
        "openings": openings,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Exported {len(openings)} openings to {output_path}")


def print_book_stats(book: OpeningBook, max_depth: int = 5):
    """Print statistics about the opening book."""
    print(f"\n{'='*50}")
    print(f"Opening Book Statistics")
    print(f"{'='*50}")
    print(f"Board Type: {book.board_type}")
    print(f"Players: {book.num_players}")
    print(f"Total Games: {book.total_games}")
    print(f"Max Depth: {book.max_depth}")
    print(f"Min Games: {book.min_games}")
    print(f"Sources: {len(book.source_databases)} database(s)")

    def count_at_depth(node: OpeningNode, depth: int, counts: dict[int, int]):
        if depth > max_depth:
            return
        counts[depth] = counts.get(depth, 0) + len(node.children)
        for child in node.children.values():
            count_at_depth(child, depth + 1, counts)

    depth_counts = {}
    count_at_depth(book.root, 0, depth_counts)

    print(f"\nMoves by Depth:")
    total_moves = 0
    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        total_moves += count
        print(f"  Depth {depth}: {count} unique moves")
    print(f"  Total: {total_moves} unique positions")

    print(f"\nTop First Moves:")
    first_moves = sorted(book.root.children.items(), key=lambda x: x[1].count, reverse=True)[:10]
    for move, node in first_moves:
        print(f"  {move}: {node.count} games, {node.win_rate:.1%} win rate, Elo {node.avg_elo:.0f}")


# Legacy function for backward compatibility
def extract_opening_sequences(
    db_path: str,
    board_type: str = "square8",
    num_players: int = 2,
    max_depth: int = 10,
    min_games: int = 50,
    min_win_rate: float = 0.52,
) -> dict[str, dict]:
    """Extract opening sequences from game database (legacy format)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT game_id, winner, moves_json
        FROM games
        WHERE board_type = ?
        AND num_players = ?
        AND status = 'completed'
        AND winner IS NOT NULL
    """, (board_type, num_players))

    sequences: dict[tuple, dict] = defaultdict(lambda: {"wins": [0] * num_players, "total": 0})

    games_processed = 0
    for row in cursor:
        try:
            moves = json.loads(row["moves_json"]) if row["moves_json"] else []
            winner = row["winner"]

            for depth in range(1, min(len(moves) + 1, max_depth + 1)):
                seq = tuple(str(m) for m in moves[:depth])
                sequences[seq]["total"] += 1
                if winner is not None and 0 <= winner < num_players:
                    sequences[seq]["wins"][winner] += 1

            games_processed += 1
        except (json.JSONDecodeError, TypeError):
            continue

    conn.close()

    opening_book = {}
    for seq, stats in sequences.items():
        if stats["total"] < min_games:
            continue

        p0_wins = stats["wins"][0]
        win_rate = p0_wins / stats["total"]

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


def build_opening_tree(opening_book: dict[str, dict]) -> dict:
    """Convert flat opening book to tree structure for efficient lookup.

    Returns:
        Tree where each node has 'stats' and 'children' keys
    """
    tree = {"stats": None, "children": {}}

    for _key, data in opening_book.items():
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
    current_moves: list[str],
    player: int = 0,
    top_k: int = 3,
) -> list[tuple[str, float]]:
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
    parser.add_argument("--db", nargs="+", help="Game database path(s), supports glob patterns")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--max-depth", type=int, default=15, help="Max opening depth")
    parser.add_argument("--min-games", type=int, default=10, help="Min games per sequence")
    parser.add_argument("--min-elo", type=float, default=1500.0, help="Min Elo for games to include")
    parser.add_argument("--min-win-rate", type=float, default=0.52, help="Min notable win rate (legacy)")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--output-tree", help="Output tree structure JSON (legacy)")
    parser.add_argument("--book", help="Existing book to load (for --stats or --export-selfplay)")
    parser.add_argument("--stats", action="store_true", help="Print book statistics")
    parser.add_argument("--export-selfplay", help="Export book in selfplay-optimized format")
    parser.add_argument("--use-legacy", action="store_true", help="Use legacy extraction format")

    args = parser.parse_args()

    # Load existing book
    if args.book:
        print(f"Loading opening book from {args.book}...")
        book = OpeningBook.load(args.book)
        if args.stats:
            print_book_stats(book)
        if args.export_selfplay:
            export_for_selfplay(book, args.export_selfplay)
        return 0

    # Expand database paths (glob patterns)
    db_paths = []
    if args.db:
        for pattern in args.db:
            expanded = glob_module.glob(pattern)
            if expanded:
                db_paths.extend(expanded)
            else:
                db_paths.append(pattern)
    else:
        # Look for selfplay databases
        candidates = [
            AI_SERVICE_ROOT / "data" / "games" / "selfplay.db",
            AI_SERVICE_ROOT / "data" / "games" / f"selfplay_{args.board}_{args.players}p.db",
            AI_SERVICE_ROOT / "data" / "games" / "aggregated" / "merged_selfplay.db",
        ]
        for c in candidates:
            if c.exists():
                db_paths.append(str(c))
                break

    if not db_paths:
        print("No database found. Specify --db path.")
        return 1

    print(f"Generating opening book from {len(db_paths)} database(s)")
    print(f"Board: {args.board}, Players: {args.players}")
    print(f"Max depth: {args.max_depth}, Min games: {args.min_games}, Min Elo: {args.min_elo}")
    print()

    # Use new or legacy extraction
    if args.use_legacy and len(db_paths) == 1:
        # Legacy flat format
        opening_book = extract_opening_sequences(
            db_path=db_paths[0],
            board_type=args.board,
            num_players=args.players,
            max_depth=args.max_depth,
            min_games=args.min_games,
            min_win_rate=args.min_win_rate,
        )

        if not opening_book:
            print("No opening sequences found matching criteria.")
            return 1

        print("\nTop 20 Opening Sequences (by games):")
        print("-" * 60)
        sorted_openings = sorted(opening_book.values(), key=lambda x: -x["games"])[:20]
        for i, data in enumerate(sorted_openings, 1):
            moves_str = " -> ".join(data["moves"][:5])
            if len(data["moves"]) > 5:
                moves_str += " ..."
            print(f"{i:2}. {moves_str}")
            print(f"    Games: {data['games']}, P0 win rate: {data['p0_win_rate']:.1%}")

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

        if args.output_tree:
            tree = build_opening_tree(opening_book)
            with open(args.output_tree, "w") as f:
                json.dump(tree, f, indent=2)
            print(f"Opening tree saved to: {args.output_tree}")
    else:
        # New Elo-weighted format
        book = generate_book_from_databases(
            db_paths=db_paths,
            board_type=args.board,
            num_players=args.players,
            max_depth=args.max_depth,
            min_games=args.min_games,
            min_elo=args.min_elo,
        )

        if book.total_games == 0:
            print("No games found matching criteria.")
            return 1

        output_path = args.output or str(AI_SERVICE_ROOT / "data" / f"opening_book_{args.board}_{args.players}p.json")
        book.save(output_path)
        print_book_stats(book)

        if args.export_selfplay:
            export_for_selfplay(book, args.export_selfplay)

    return 0


if __name__ == "__main__":
    sys.exit(main())
