#!/usr/bin/env python3
"""Cross-AI Self-Play Generator for Training Data Variety.

This script generates training games with diverse AI matchups:
- Random/Heuristic vs Minimax/MCTS/Descent (cross-skill matchups)
- NNUE-based Minimax vs pure heuristic Minimax
- MCTS vs Descent (search algorithm variety)

Goals:
1. Balanced game generation across ALL 9 board/player configurations
2. CPU-heavy games using Minimax (D3-4), MCTS (D5-8), Descent (D9-10)
3. Cross-AI matchups to improve training diversity

Usage:
    # Run balanced selfplay across all configs
    python scripts/run_cross_ai_selfplay.py --games-per-config 100

    # Focus on underrepresented configs
    python scripts/run_cross_ai_selfplay.py --prioritize-underrepresented --target-games 1000

    # Specific matchup types
    python scripts/run_cross_ai_selfplay.py --matchup-type cross-skill --games-per-config 50
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

# NOTE: Shadow contracts are now enabled to validate training data against TS rules.
# Requires Node.js and compiled TypeScript (npm install && npx tsc -p tsconfig.server.json)

# Ensure app.* imports resolve
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import _create_ai_instance, _get_difficulty_profile
from app.models import (
    AIConfig,
    AIType,
    BoardType,
    GameState,
    GameStatus,
)
from app.game_engine import GameEngine
from app.training.generate_data import create_initial_state
from app.training.selfplay_config import SelfplayConfig, create_argument_parser

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_cross_ai_selfplay")


# All 9 board/player configurations
ALL_CONFIGS = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]

# Matchup definitions for cross-AI games
# Format: (player1_difficulties, player2_difficulties, description)
MATCHUP_TYPES = {
    # Fast: only random/heuristic (D1-2), no search - fastest for testing
    "fast": [
        ([1], [1], "random vs random"),
        ([1], [2], "random vs heuristic"),
        ([2], [2], "heuristic vs heuristic"),
    ],
    # CPU-light: no neural nets, fast games (D1=random, D2=heuristic, D3-4=minimax)
    "cpu-light": [
        ([1, 2], [3, 4], "random/heuristic vs minimax"),
        ([2], [2], "heuristic vs heuristic"),
        ([3, 4], [3, 4], "minimax vs minimax"),
        ([1], [3, 4], "random vs minimax"),
    ],
    # Cross-skill: weak vs strong
    "cross-skill": [
        ([1, 2], [3, 4, 5, 6], "random/heuristic vs minimax/mcts"),
        ([1, 2], [7, 8, 9, 10], "random/heuristic vs strong mcts/descent"),
        ([3, 4], [7, 8, 9, 10], "minimax vs strong mcts/descent"),
    ],
    # Search variety: different search algorithms
    "search-variety": [
        ([3, 4], [5, 6], "minimax vs mcts-weak"),
        ([5, 6], [9, 10], "mcts vs descent"),
        ([3, 4], [9, 10], "minimax vs descent"),
    ],
    # CPU-heavy: all search-based AIs
    "cpu-heavy": [
        ([3, 4, 5, 6], [5, 6, 7, 8], "minimax/mcts vs mcts"),
        ([5, 6, 7, 8], [7, 8, 9, 10], "mcts vs mcts/descent"),
        ([3, 4], [3, 4], "minimax vs minimax"),
        ([9, 10], [9, 10], "descent vs descent"),
    ],
    # Balanced: mix of all types
    "balanced": [
        ([1, 2], [3, 4, 5], "weak vs mid"),
        ([3, 4, 5], [6, 7, 8], "mid vs strong"),
        ([5, 6, 7], [8, 9, 10], "mid-strong vs strongest"),
        ([1, 2, 3], [7, 8, 9, 10], "weak-mid vs strongest"),
    ],
}


@dataclass
class MatchupConfig:
    """Configuration for a specific matchup."""
    p1_difficulties: List[int]
    p2_difficulties: List[int]
    description: str


@dataclass
class GameResult:
    """Result of a single game."""
    game_id: str
    board_type: str
    num_players: int
    winner: int
    moves: int
    p1_ai_type: str
    p1_difficulty: int
    p2_ai_type: str
    p2_difficulty: int
    duration_ms: float


def get_board_type_enum(board_type: str) -> BoardType:
    """Convert string to BoardType enum."""
    mapping = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }
    return mapping.get(board_type, BoardType.SQUARE8)


def create_ai_for_player(
    player_num: int,
    difficulty: int,
    board_type: BoardType,
    seed: Optional[int] = None,
) -> Tuple[Any, str]:
    """Create an AI instance for a player.

    Returns (ai_instance, ai_type_name)
    """
    profile = _get_difficulty_profile(difficulty)
    ai_type = profile["ai_type"]

    config = AIConfig(
        difficulty=difficulty,
        randomness=profile["randomness"],
        think_time_ms=profile["think_time_ms"],
        use_opening_book=False,
        use_neural_net=profile.get("use_neural_net", False),
    )

    ai = _create_ai_instance(ai_type, player_num, config)
    ai_type_name = ai_type.value if hasattr(ai_type, 'value') else str(ai_type)

    return ai, ai_type_name


def play_game(
    board_type: str,
    num_players: int,
    matchup: MatchupConfig,
    max_moves: int = 2000,
    seed: Optional[int] = None,
) -> Optional[GameResult]:
    """Play a single game with the specified matchup."""
    rng = random.Random(seed)
    game_id = str(uuid.uuid4())

    # Select difficulties for each player
    bt_enum = get_board_type_enum(board_type)

    ais = {}
    ai_info = {}

    for pnum in range(1, num_players + 1):
        if pnum == 1:
            diff = rng.choice(matchup.p1_difficulties)
        elif pnum == 2:
            diff = rng.choice(matchup.p2_difficulties)
        else:
            # For 3+ player games, alternate between p1 and p2 difficulty pools
            pool = matchup.p1_difficulties if pnum % 2 == 1 else matchup.p2_difficulties
            diff = rng.choice(pool)

        ai, ai_type_name = create_ai_for_player(pnum, diff, bt_enum, seed)
        ais[pnum] = ai
        ai_info[pnum] = {"type": ai_type_name, "difficulty": diff}

    # Initialize game state
    state = create_initial_state(
        board_type=bt_enum,
        num_players=num_players,
    )

    engine = GameEngine()
    start_time = time.time()
    moves = 0

    try:
        # Handle both enum and string status values for compatibility
        def is_active(status):
            if isinstance(status, GameStatus):
                return status == GameStatus.ACTIVE
            return str(status).lower() == "active"

        while is_active(state.game_status) and moves < max_moves:
            current_player = state.current_player
            ai = ais.get(current_player)

            if ai is None:
                logger.error(f"No AI for player {current_player}")
                return None

            # Get AI move
            move = ai.select_move(state)
            if move is None:
                logger.warning(f"AI returned None move at move {moves}")
                break

            # Apply move
            state = engine.apply_move(state, move)
            moves += 1

        duration_ms = (time.time() - start_time) * 1000

        # Determine winner
        winner = 0
        if state.game_status == GameStatus.COMPLETED:
            winner = state.winner if state.winner else 0

        return GameResult(
            game_id=game_id,
            board_type=board_type,
            num_players=num_players,
            winner=winner,
            moves=moves,
            p1_ai_type=ai_info[1]["type"],
            p1_difficulty=ai_info[1]["difficulty"],
            p2_ai_type=ai_info[2]["type"],
            p2_difficulty=ai_info[2]["difficulty"],
            duration_ms=duration_ms,
        )

    except Exception as e:
        import traceback
        logger.error(f"Error playing game: {e}\n{traceback.format_exc()}")
        return None


def get_config_game_counts(data_dir: Path) -> Dict[str, int]:
    """Get current game counts per config from database files."""
    import sqlite3

    counts = {}
    for board, players in ALL_CONFIGS:
        config_key = f"{board}_{players}p"
        counts[config_key] = 0

        # Check various database locations
        db_patterns = [
            f"jsonl_converted_{board}_{players}p.db",
            f"selfplay_{board}_{players}p.db",
            f"games_{board}_{players}p.db",
        ]

        for pattern in db_patterns:
            db_path = data_dir / "games" / pattern
            if db_path.exists():
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM games")
                    count = cursor.fetchone()[0]
                    counts[config_key] = max(counts[config_key], count)
                    conn.close()
                except Exception:
                    pass

    return counts


def prioritize_configs(counts: Dict[str, int], target_per_config: int) -> List[Tuple[str, int]]:
    """Return configs prioritized by how far below target they are.

    Returns list of (config_key, games_needed) sorted by most needed first.
    """
    needed = []
    for config_key, current in counts.items():
        games_needed = max(0, target_per_config - current)
        if games_needed > 0:
            needed.append((config_key, games_needed))

    # Sort by most needed first
    needed.sort(key=lambda x: -x[1])
    return needed


def _play_game_worker(args: Tuple) -> Optional[Dict]:
    """Worker function for parallel game execution."""
    board_type, num_players, matchup_def, max_moves, seed = args

    matchup = MatchupConfig(
        p1_difficulties=matchup_def[0],
        p2_difficulties=matchup_def[1],
        description=matchup_def[2],
    )

    result = play_game(
        board_type=board_type,
        num_players=num_players,
        matchup=matchup,
        max_moves=max_moves,
        seed=seed,
    )

    if result:
        return asdict(result)
    return None


def run_balanced_selfplay(
    games_per_config: int,
    matchup_type: str,
    output_dir: Path,
    max_moves: int = 2000,
    prioritize_underrepresented: bool = False,
    target_games: Optional[int] = None,
    num_workers: int = 1,
):
    """Run balanced selfplay across all configurations.

    Args:
        num_workers: Number of parallel workers (default: 1 for sequential)
    """

    matchups = MATCHUP_TYPES.get(matchup_type, MATCHUP_TYPES["balanced"])

    # Get current game counts if prioritizing
    if prioritize_underrepresented:
        data_dir = ROOT / "data"
        counts = get_config_game_counts(data_dir)

        if target_games:
            prioritized = prioritize_configs(counts, target_games)
            logger.info(f"Prioritized configs (by games needed): {prioritized[:5]}")
        else:
            # Default: target equal distribution
            max_count = max(counts.values()) if counts else 10000
            prioritized = prioritize_configs(counts, max_count)

        # Build weighted config list
        weighted_configs = []
        for config_key, needed in prioritized:
            parts = config_key.rsplit("_", 1)
            board = parts[0]
            players = int(parts[1].replace("p", ""))
            # Add config with weight proportional to games needed
            weight = min(needed // 100 + 1, 10)  # Cap weight at 10
            for _ in range(weight):
                weighted_configs.append((board, players))

        if not weighted_configs:
            weighted_configs = ALL_CONFIGS.copy()
    else:
        weighted_configs = ALL_CONFIGS.copy()

    # Track results
    results_by_config = {f"{b}_{p}p": [] for b, p in ALL_CONFIGS}
    total_games = len(ALL_CONFIGS) * games_per_config

    logger.info(f"Starting balanced selfplay: {total_games} games, matchup type: {matchup_type}")
    logger.info(f"Matchups: {[m[2] for m in matchups]}")
    logger.info(f"Workers: {num_workers}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of all game tasks
    game_tasks = []
    for config_idx, (board_type, num_players) in enumerate(weighted_configs * games_per_config):
        if len(game_tasks) >= total_games:
            break
        matchup_def = random.choice(matchups)
        seed = random.randint(0, 2**31)
        game_tasks.append((board_type, num_players, matchup_def, max_moves, seed))

    game_num = 0
    start_time = time.time()

    if num_workers > 1:
        # Parallel execution
        logger.info(f"Using {num_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_play_game_worker, task): task for task in game_tasks}

            for future in as_completed(futures):
                task = futures[future]
                board_type, num_players, _, _, _ = task
                config_key = f"{board_type}_{num_players}p"

                try:
                    result_dict = future.result()
                    if result_dict:
                        results_by_config[config_key].append(result_dict)

                        # Write result to JSONL (thread-safe with append)
                        jsonl_path = output_dir / f"cross_ai_{config_key}.jsonl"
                        with open(jsonl_path, "a") as f:
                            f.write(json.dumps(result_dict) + "\n")

                except Exception as e:
                    logger.error(f"Game failed: {e}")

                game_num += 1

                # Enhanced progress logging with ETA
                log_interval = max(10, total_games // 20)
                if game_num % log_interval == 0 or game_num == 1 or game_num == total_games:
                    elapsed = time.time() - start_time
                    rate = game_num / elapsed if elapsed > 0 else 0
                    remaining = total_games - game_num
                    eta_seconds = remaining / rate if rate > 0 else 0
                    pct = game_num / total_games * 100
                    logger.info(
                        "[cross-ai-selfplay] Game %d/%d (%.1f%%) | %.2f games/s | ETA: %.0fs",
                        game_num,
                        total_games,
                        pct,
                        rate,
                        eta_seconds,
                    )
    else:
        # Sequential execution (original behavior)
        for task in game_tasks:
            board_type, num_players, matchup_def, max_moves_t, seed = task
            config_key = f"{board_type}_{num_players}p"

            matchup = MatchupConfig(
                p1_difficulties=matchup_def[0],
                p2_difficulties=matchup_def[1],
                description=matchup_def[2],
            )

            result = play_game(
                board_type=board_type,
                num_players=num_players,
                matchup=matchup,
                max_moves=max_moves_t,
                seed=seed,
            )

            if result:
                results_by_config[config_key].append(result)

                # Write result to JSONL
                jsonl_path = output_dir / f"cross_ai_{config_key}.jsonl"
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(asdict(result)) + "\n")

            game_num += 1

            # Enhanced progress logging with ETA
            log_interval = 10 if total_games >= 20 else 1
            if game_num % log_interval == 0 or game_num == 1 or game_num == total_games:
                elapsed = time.time() - start_time
                rate = game_num / elapsed if elapsed > 0 else 0
                remaining = total_games - game_num
                eta_seconds = remaining / rate if rate > 0 else 0
                pct = game_num / total_games * 100
                logger.info(
                    "[cross-ai-selfplay] Game %d/%d (%.1f%%) | %.2f games/s | ETA: %.0fs",
                    game_num,
                    total_games,
                    pct,
                    rate,
                    eta_seconds,
                )

    # Summary
    logger.info("=" * 50)
    logger.info("SELFPLAY SUMMARY")
    logger.info("=" * 50)

    for config_key, results in sorted(results_by_config.items()):
        if results:
            wins = sum(1 for r in results if r.winner == 1)
            avg_moves = sum(r.moves for r in results) / len(results)
            avg_duration = sum(r.duration_ms for r in results) / len(results)
            logger.info(
                f"  {config_key}: {len(results)} games, "
                f"P1 wins: {100*wins/len(results):.1f}%, "
                f"avg moves: {avg_moves:.1f}, "
                f"avg duration: {avg_duration:.0f}ms"
            )

    total_elapsed = time.time() - start_time
    logger.info(f"Total time: {total_elapsed:.1f}s, {game_num/total_elapsed:.2f} games/sec")


def main():
    # Use unified argument parser from SelfplayConfig
    parser = create_argument_parser(
        description="Cross-AI Self-Play Generator",
        include_gpu=False,  # Cross-AI is CPU-based
        include_ramdrive=False,
    )
    # Add script-specific arguments
    parser.add_argument(
        "--games-per-config",
        type=int,
        default=100,
        help="Number of games per board/player config",
    )
    parser.add_argument(
        "--matchup-type",
        choices=list(MATCHUP_TYPES.keys()),
        default="balanced",
        help="Type of AI matchups to generate",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Maximum moves per game",
    )
    parser.add_argument(
        "--prioritize-underrepresented",
        action="store_true",
        help="Generate more games for underrepresented configs",
    )
    parser.add_argument(
        "--target-games",
        type=int,
        help="Target game count per config (for prioritization)",
    )

    parsed = parser.parse_args()

    # Create SelfplayConfig from parsed args
    config = SelfplayConfig(
        board_type=parsed.board,
        num_players=parsed.num_players,
        num_games=parsed.games_per_config * len(ALL_CONFIGS),  # Total games
        output_dir=parsed.output_dir or str(ROOT / "data" / "games" / "cross_ai"),
        num_workers=parsed.num_workers,
        seed=parsed.seed,
        source="run_cross_ai_selfplay.py",
        # Store script-specific options
        extra_options={
            "matchup_type": parsed.matchup_type,
            "prioritize_underrepresented": parsed.prioritize_underrepresented,
            "target_games": parsed.target_games,
            "max_moves": parsed.max_moves,
            "games_per_config": parsed.games_per_config,
        },
    )

    # Limit workers to avoid resource exhaustion
    cpu_count = mp.cpu_count()
    max_workers = min(config.num_workers, cpu_count - 2, 32)  # Leave 2 CPUs free, cap at 32
    if max_workers < 1:
        max_workers = 1

    run_balanced_selfplay(
        games_per_config=config.extra_options["games_per_config"],
        matchup_type=config.extra_options["matchup_type"],
        output_dir=Path(config.output_dir),
        max_moves=config.extra_options["max_moves"],
        prioritize_underrepresented=config.extra_options["prioritize_underrepresented"],
        target_games=config.extra_options["target_games"],
        num_workers=max_workers,
    )


if __name__ == "__main__":
    main()
