#!/usr/bin/env python3
"""Filter and weight training data by participant Elo ratings.

Higher quality games (between high-Elo models) should contribute more
to training than lower quality games. This script:
1. Reads JSONL game files
2. Looks up participant Elo ratings
3. Filters out low-quality games
4. Assigns quality weights based on participant Elo

Usage:
    # Filter games - keep only games with avg Elo > 1400
    python scripts/filter_training_data.py --min-avg-elo 1400

    # Create weighted NPZ with Elo-based sample weights
    python scripts/filter_training_data.py --create-weighted-npz

    # Show statistics about training data quality
    python scripts/filter_training_data.py --stats

    # Filter specific board config
    python scripts/filter_training_data.py --board square8 --players 2
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

SELFPLAY_DIR = AI_SERVICE_ROOT / "data" / "selfplay"
TRAINING_DIR = AI_SERVICE_ROOT / "data" / "training"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"

# Quality thresholds
DEFAULT_MIN_AVG_ELO = 1350  # Minimum average Elo of participants
DEFAULT_MIN_WINNER_ELO = 1300  # Minimum Elo of winning player
HIGH_QUALITY_ELO = 1600  # Games above this get bonus weight


@dataclass
class GameQuality:
    """Quality metrics for a game."""
    game_id: str
    file_path: str
    participants: list[str]
    participant_elos: list[float]
    avg_elo: float
    min_elo: float
    max_elo: float
    winner_elo: float | None
    num_moves: int
    quality_weight: float
    ai_type: str = "unknown"
    victory_type: str = "unknown"
    is_timeout: bool = False


# NN-guided AI types for filtering
NN_GUIDED_AI_TYPES = {
    "gumbel_mcts", "gumbel-mcts",
    "policy_only", "policy-only",
    "nnue_guided", "nnue-guided",
    "neural_net", "neural-net",
    "nn_vs_nn_tournament",
    "nn-minimax", "nn_minimax",
    "nn-descent", "nn_descent",
    "descent",  # Also NN-guided
}


def get_elo_ratings() -> dict[str, float]:
    """Load all Elo ratings from unified database."""
    ratings = {}

    if not ELO_DB_PATH.exists():
        print(f"Warning: Elo database not found at {ELO_DB_PATH}")
        return ratings

    conn = sqlite3.connect(ELO_DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT participant_id, rating, games_played
            FROM elo_ratings
            WHERE games_played >= 5
        """)

        for row in cursor.fetchall():
            participant_id, rating, games = row
            ratings[participant_id] = rating

            # Also add without prefix
            if participant_id.startswith("nn:"):
                clean_id = participant_id[3:]
                ratings[clean_id] = rating
                ratings[Path(clean_id).name] = rating

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

    return ratings


def calculate_quality_weight(avg_elo: float, min_elo: float) -> float:
    """Calculate quality weight based on participant Elo.

    Returns weight in range [0.1, 2.0]:
    - Low quality (avg < 1300): 0.1
    - Below average (1300-1400): 0.5
    - Average (1400-1500): 1.0
    - Above average (1500-1600): 1.3
    - High quality (1600+): 1.5-2.0
    """
    if avg_elo < 1300:
        return 0.1
    elif avg_elo < 1400:
        return 0.5
    elif avg_elo < 1500:
        return 1.0
    elif avg_elo < 1600:
        return 1.3
    else:
        # Scale from 1.5 to 2.0 based on how far above 1600
        bonus = min((avg_elo - 1600) / 200, 0.5)  # Max 0.5 bonus
        return 1.5 + bonus


def analyze_game(
    game_data: dict[str, Any],
    file_path: str,
    elo_ratings: dict[str, float],
) -> GameQuality | None:
    """Analyze a single game and compute quality metrics."""
    try:
        game_id = game_data.get("game_id", "unknown")

        # Extract participants
        participants = []
        if "players" in game_data:
            for player in game_data["players"]:
                if isinstance(player, dict):
                    participants.append(player.get("model", player.get("id", "unknown")))
                else:
                    participants.append(str(player))
        elif "player_configs" in game_data:
            for config in game_data["player_configs"]:
                if isinstance(config, dict):
                    participants.append(config.get("model_path", config.get("id", "unknown")))

        if not participants:
            return None

        # Look up Elo ratings
        participant_elos = []
        for p in participants:
            elo = elo_ratings.get(p, elo_ratings.get(f"nn:{p}", 1500))
            participant_elos.append(elo)

        avg_elo = sum(participant_elos) / len(participant_elos)
        min_elo = min(participant_elos)
        max_elo = max(participant_elos)

        # Get winner Elo if available
        winner_elo = None
        winner_idx = game_data.get("winner")
        if winner_idx is not None and 0 <= winner_idx < len(participant_elos):
            winner_elo = participant_elos[winner_idx]

        # Count moves
        num_moves = len(game_data.get("moves", []))

        # Calculate quality weight
        quality_weight = calculate_quality_weight(avg_elo, min_elo)

        # Extract AI type
        ai_type = game_data.get("ai_type", game_data.get("_ai_type", "unknown"))
        if ai_type == "unknown":
            # Try to infer from source or participants
            source = game_data.get("source", "")
            if "gumbel" in source.lower():
                ai_type = "gumbel_mcts"
            elif "policy" in source.lower():
                ai_type = "policy_only"
            elif "nnue" in source.lower():
                ai_type = "nnue_guided"

        # Extract victory type and timeout status
        victory_type = game_data.get("victory_type", game_data.get("termination_reason", "unknown"))
        is_timeout = "timeout" in str(victory_type).lower()

        return GameQuality(
            game_id=game_id,
            file_path=file_path,
            participants=participants,
            participant_elos=participant_elos,
            avg_elo=avg_elo,
            min_elo=min_elo,
            max_elo=max_elo,
            winner_elo=winner_elo,
            num_moves=num_moves,
            quality_weight=quality_weight,
            ai_type=ai_type,
            victory_type=victory_type,
            is_timeout=is_timeout,
        )

    except Exception as e:
        return None


def analyze_jsonl_file(
    file_path: Path,
    elo_ratings: dict[str, float],
) -> list[GameQuality]:
    """Analyze all games in a JSONL file."""
    games = []

    try:
        # Detect gzip by magic bytes or extension
        is_gzip = str(file_path).endswith('.gz')
        if not is_gzip:
            try:
                with open(file_path, 'rb') as check_f:
                    magic = check_f.read(2)
                    is_gzip = magic == b'\x1f\x8b'
            except Exception:
                pass

        opener = gzip.open if is_gzip else open
        with opener(file_path, "rt", encoding="utf-8") as f:
            try:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        game_data = json.loads(line)
                        quality = analyze_game(game_data, str(file_path), elo_ratings)
                        if quality:
                            games.append(quality)
                    except json.JSONDecodeError:
                        continue
            except (EOFError, OSError):
                # Handle truncated gzip files
                pass
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return games


def filter_games(
    games: list[GameQuality],
    min_avg_elo: float = DEFAULT_MIN_AVG_ELO,
    min_winner_elo: float = DEFAULT_MIN_WINNER_ELO,
    min_moves: int = 10,
    ai_types_filter: set | None = None,
    exclude_timeout: bool = False,
    exclude_random: bool = False,
) -> list[GameQuality]:
    """Filter games based on quality criteria."""
    filtered = []

    for game in games:
        # Filter by average Elo
        if game.avg_elo < min_avg_elo:
            continue

        # Filter by winner Elo (if available)
        if game.winner_elo is not None and game.winner_elo < min_winner_elo:
            continue

        # Filter by minimum moves (avoid trivial games)
        if game.num_moves < min_moves:
            continue

        # Filter by AI type if specified
        if ai_types_filter and game.ai_type not in ai_types_filter:
            continue

        # Exclude timeout games
        if exclude_timeout and game.is_timeout:
            continue

        # Exclude games with random opponent
        if exclude_random and "random" in game.ai_type.lower():
            continue

        filtered.append(game)

    return filtered


def create_weighted_training_data(
    games: list[GameQuality],
    board_type: str,
    num_players: int,
    output_dir: Path,
) -> Path | None:
    """Create NPZ training file with quality-weighted samples.

    This creates sample_weights that can be used by the training script
    to weight samples during training.
    """
    if not games:
        return None

    # Group games by quality weight
    weight_groups = defaultdict(list)
    for game in games:
        weight_groups[game.quality_weight].append(game)

    # Create weights array
    weights = np.array([g.quality_weight for g in games], dtype=np.float32)

    # Normalize weights to sum to len(games)
    weights = weights / weights.mean()

    # Save weights file
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_file = output_dir / f"{board_type}_{num_players}p_quality_weights.npz"

    np.savez_compressed(
        weights_file,
        weights=weights,
        game_ids=np.array([g.game_id for g in games]),
        avg_elos=np.array([g.avg_elo for g in games]),
    )

    print(f"Saved quality weights to {weights_file}")
    return weights_file


def print_quality_stats(games: list[GameQuality], title: str = "Training Data Quality"):
    """Print statistics about training data quality."""
    if not games:
        print(f"\n{title}: No games to analyze")
        return

    avg_elos = [g.avg_elo for g in games]
    weights = [g.quality_weight for g in games]
    moves = [g.num_moves for g in games]

    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(f"\nTotal games: {len(games)}")
    print(f"\nElo statistics:")
    print(f"  Average Elo: {sum(avg_elos)/len(avg_elos):.0f}")
    print(f"  Min avg Elo: {min(avg_elos):.0f}")
    print(f"  Max avg Elo: {max(avg_elos):.0f}")

    # Distribution by quality tier
    tiers = [
        (0, 1300, "Low quality (< 1300)"),
        (1300, 1400, "Below avg (1300-1400)"),
        (1400, 1500, "Average (1400-1500)"),
        (1500, 1600, "Above avg (1500-1600)"),
        (1600, 9999, "High quality (1600+)"),
    ]

    print("\nQuality distribution:")
    for low, high, label in tiers:
        count = sum(1 for e in avg_elos if low <= e < high)
        pct = 100 * count / len(games)
        print(f"  {label}: {count} games ({pct:.1f}%)")

    print(f"\nQuality weights:")
    print(f"  Mean weight: {sum(weights)/len(weights):.2f}")
    print(f"  Min weight: {min(weights):.2f}")
    print(f"  Max weight: {max(weights):.2f}")

    print(f"\nGame length:")
    print(f"  Avg moves: {sum(moves)/len(moves):.0f}")
    print(f"  Min moves: {min(moves)}")
    print(f"  Max moves: {max(moves)}")


def main():
    parser = argparse.ArgumentParser(description="Filter training data by quality")
    parser.add_argument("--board", type=str, help="Filter specific board type")
    parser.add_argument("--players", type=int, help="Filter specific player count")
    parser.add_argument("--data-dir", type=str, help="Data directory to scan (default: data/selfplay)")
    parser.add_argument("--min-avg-elo", type=float, default=DEFAULT_MIN_AVG_ELO, help=f"Minimum average Elo (default: {DEFAULT_MIN_AVG_ELO})")
    parser.add_argument("--min-winner-elo", type=float, default=DEFAULT_MIN_WINNER_ELO, help=f"Minimum winner Elo (default: {DEFAULT_MIN_WINNER_ELO})")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--create-weighted-npz", action="store_true", help="Create weighted training data")
    parser.add_argument("--output-dir", type=str, help="Output directory for filtered data")
    parser.add_argument(
        "--ai-types",
        type=str,
        nargs="+",
        help="Only include games with these AI types (e.g., --ai-types gumbel_mcts policy_only nnue-guided)",
    )
    parser.add_argument(
        "--exclude-timeout",
        action="store_true",
        help="Exclude games that ended in timeout",
    )
    parser.add_argument(
        "--exclude-random",
        action="store_true",
        help="Exclude games with random AI opponent",
    )

    args = parser.parse_args()

    print("Loading Elo ratings...")
    elo_ratings = get_elo_ratings()
    print(f"Loaded {len(elo_ratings)} Elo ratings")

    # Find JSONL files
    data_dir = Path(args.data_dir) if args.data_dir else SELFPLAY_DIR
    print(f"\nScanning training data in {data_dir}...")
    all_games = []

    patterns = []
    if args.board and args.players:
        patterns.append(f"*{args.board}*{args.players}p*.jsonl*")
    elif args.board:
        patterns.append(f"*{args.board}*.jsonl*")
    else:
        patterns.append("*.jsonl*")

    for pattern in patterns:
        for jsonl_file in data_dir.rglob(pattern):
            games = analyze_jsonl_file(jsonl_file, elo_ratings)
            all_games.extend(games)
            if games:
                print(f"  {jsonl_file.name}: {len(games)} games")

    print(f"\nTotal games found: {len(all_games)}")

    # Build AI types filter set
    ai_types_filter = set(args.ai_types) if args.ai_types else None
    if ai_types_filter:
        print(f"Filtering to AI types: {ai_types_filter}")

    if args.stats:
        print_quality_stats(all_games, "All Training Data")

        # Also show filtered stats
        filtered = filter_games(
            all_games,
            args.min_avg_elo,
            args.min_winner_elo,
            ai_types_filter=ai_types_filter,
            exclude_timeout=args.exclude_timeout,
            exclude_random=args.exclude_random,
        )
        filter_desc = f"Filtered (avg Elo >= {args.min_avg_elo}"
        if ai_types_filter:
            filter_desc += f", AI types: {len(ai_types_filter)}"
        if args.exclude_timeout:
            filter_desc += ", no timeout"
        if args.exclude_random:
            filter_desc += ", no random"
        filter_desc += ")"
        print_quality_stats(filtered, filter_desc)
        return

    # Filter games
    filtered = filter_games(
        all_games,
        args.min_avg_elo,
        args.min_winner_elo,
        ai_types_filter=ai_types_filter,
        exclude_timeout=args.exclude_timeout,
        exclude_random=args.exclude_random,
    )
    print(f"\nFiltered to {len(filtered)} high-quality games")

    print_quality_stats(filtered)

    # Create weighted training data if requested
    if args.create_weighted_npz:
        output_dir = Path(args.output_dir) if args.output_dir else TRAINING_DIR / "weighted"
        board_type = args.board or "all"
        num_players = args.players or 0

        weights_file = create_weighted_training_data(filtered, board_type, num_players, output_dir)
        if weights_file:
            print(f"\nCreated weighted training data: {weights_file}")


if __name__ == "__main__":
    main()
