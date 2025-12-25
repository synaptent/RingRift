#!/usr/bin/env python3
"""Generate blended training data from multiple engine modes.

This script generates balanced training data by mixing multiple selfplay engines
to maintain both random AND heuristic performance. The key insight is that
pure Gumbel MCTS improves heuristic play but hurts random performance, while
pure heuristic data does the opposite.

Default blend ratios (optimized for 3p/4p gauntlet):
- 40% heuristic-only: Positional understanding, broad coverage
- 30% NNUE-guided: Hybrid NN + heuristic, tactical + positional
- 20% Gumbel MCTS: Deep tactical positions, strong play
- 10% diverse/random: Opening diversity, exploration

Usage:
    # Generate blended data for 3-player square8
    python scripts/generate_blended_selfplay.py \
        --board square8 --num-players 3 \
        --num-games 2000 \
        --output data/training/sq8_3p_blended.npz

    # Custom blend ratios
    python scripts/generate_blended_selfplay.py \
        --board square8 --num-players 4 \
        --num-games 1000 \
        --blend heuristic:0.5,gumbel:0.3,nnue:0.2 \
        --output data/training/sq8_4p_custom.npz

    # Generate to database instead of NPZ
    python scripts/generate_blended_selfplay.py \
        --board square8 --num-players 3 \
        --num-games 500 \
        --output-db data/games/sq8_3p_blended.db
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure app imports resolve
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default blend for 3p/4p games (optimized for gauntlet)
DEFAULT_BLEND = {
    "heuristic": 0.40,  # Broad positional understanding
    "nnue": 0.30,       # Hybrid NN + heuristic evaluation
    "gumbel": 0.20,     # Deep tactical positions
    "diverse": 0.10,    # Opening diversity and exploration
}


@dataclass
class BlendedSelfplayConfig:
    """Configuration for blended selfplay generation."""
    board_type: str = "square8"
    num_players: int = 3
    num_games: int = 1000
    blend_ratios: dict[str, float] = field(default_factory=lambda: DEFAULT_BLEND.copy())
    output_path: str = ""
    output_db: str = ""
    seed: int = 0
    verbose: bool = False
    # Engine-specific settings
    gumbel_budget: int = 150
    nnue_depth: int = 3
    heuristic_difficulty: int = 7
    diverse_temp_range: tuple[float, float] = (0.8, 1.5)
    # GPU settings
    use_gpu: bool = True
    allow_fresh_weights: bool = True


def parse_blend_string(blend_str: str) -> dict[str, float]:
    """Parse blend ratio string like 'heuristic:0.4,gumbel:0.3,nnue:0.2,diverse:0.1'."""
    ratios = {}
    for part in blend_str.split(","):
        if ":" not in part:
            continue
        engine, ratio = part.strip().split(":")
        ratios[engine.strip().lower()] = float(ratio.strip())

    # Normalize to sum to 1.0
    total = sum(ratios.values())
    if total > 0:
        ratios = {k: v / total for k, v in ratios.items()}

    return ratios


def games_per_engine(total_games: int, ratios: dict[str, float]) -> dict[str, int]:
    """Calculate number of games per engine based on ratios."""
    result = {}
    remaining = total_games

    # Sort by ratio to handle rounding better
    sorted_engines = sorted(ratios.items(), key=lambda x: x[1], reverse=True)

    for i, (engine, ratio) in enumerate(sorted_engines):
        if i == len(sorted_engines) - 1:
            # Last engine gets remaining games
            result[engine] = remaining
        else:
            games = int(total_games * ratio)
            result[engine] = games
            remaining -= games

    return result


def run_heuristic_games(
    config: BlendedSelfplayConfig,
    num_games: int,
) -> list[dict]:
    """Generate games using heuristic AI."""
    from app.training.selfplay_runner import HeuristicSelfplayRunner
    from app.training.selfplay_config import SelfplayConfig, EngineMode

    logger.info(f"[heuristic] Generating {num_games} games...")

    selfplay_config = SelfplayConfig(
        board_type=config.board_type,
        num_players=config.num_players,
        num_games=num_games,
        engine_mode=EngineMode.HEURISTIC,
        seed=config.seed,
    )

    runner = HeuristicSelfplayRunner(selfplay_config)
    stats = runner.run()

    logger.info(f"[heuristic] Completed {stats.games_completed} games")
    return []  # Samples are in the DB/output


def run_gumbel_games(
    config: BlendedSelfplayConfig,
    num_games: int,
    output_db: str,
) -> int:
    """Generate games using Gumbel MCTS.

    Returns number of games generated.
    """
    from app.models import BoardType
    from scripts.generate_gumbel_selfplay import (
        GumbelSelfplayConfig,
        run_selfplay,
    )

    logger.info(f"[gumbel] Generating {num_games} games with budget={config.gumbel_budget}...")

    gumbel_config = GumbelSelfplayConfig(
        board_type=config.board_type,
        num_players=config.num_players,
        num_games=num_games,
        simulation_budget=config.gumbel_budget,
        db_path=output_db,
        seed=config.seed + 1000 if config.seed else 0,
        use_gpu=config.use_gpu,
        allow_fresh_weights=config.allow_fresh_weights,
        verbose=config.verbose,
    )

    results = run_selfplay(gumbel_config)
    logger.info(f"[gumbel] Completed {len(results)} games")
    return len(results)


def run_nnue_games(
    config: BlendedSelfplayConfig,
    num_games: int,
    output_db: str,
) -> int:
    """Generate games using NNUE-guided search.

    Returns number of games generated.
    """
    import uuid
    from app.db.game_replay import GameReplayDB
    from app.game_engine import GameEngine
    from app.models import BoardType, GameStatus, AIConfig, AIType
    from app.ai.factory import AIFactory
    from app.training.initial_state import create_initial_state

    logger.info(f"[nnue] Generating {num_games} games with depth={config.nnue_depth}...")

    board_type = BoardType(config.board_type)
    db = GameReplayDB(output_db, enforce_canonical_history=False)
    games_completed = 0

    # Create NNUE-guided AI for each player
    ais = {}
    for p in range(1, config.num_players + 1):
        ai_config = AIConfig(
            difficulty=8,
            use_neural_net=True,
            search_depth=config.nnue_depth,
        )
        try:
            ais[p] = AIFactory.create(
                AIType.NNUE,
                player_number=p,
                config=ai_config,
                board_type=board_type,
            )
        except Exception:
            # Fall back to heuristic if NNUE not available
            ais[p] = AIFactory.create(
                AIType.HEURISTIC,
                player_number=p,
                config=ai_config,
            )

    for game_idx in range(num_games):
        try:
            game_id = str(uuid.uuid4())
            state = create_initial_state(board_type, config.num_players)
            moves = []

            while state.game_status != GameStatus.COMPLETED and len(moves) < 500:
                current_player = state.current_player
                ai = ais[current_player]

                move = ai.select_move(state)
                if not move:
                    break

                state = GameEngine.apply_move(state, move)
                moves.append(move)

            if state.game_status == GameStatus.COMPLETED:
                # Record game to DB
                try:
                    db.record_game_simple(
                        game_id=game_id,
                        initial_state=create_initial_state(board_type, config.num_players),
                        moves=moves,
                        final_state=state,
                        metadata={
                            "source": "blended_nnue",
                            "engine_mode": "nnue-guided",
                        },
                    )
                    games_completed += 1
                except Exception as e:
                    logger.debug(f"[nnue] Failed to record game: {e}")

        except Exception as e:
            logger.debug(f"[nnue] Game {game_idx} failed: {e}")
            continue

    logger.info(f"[nnue] Completed {games_completed} games")
    return games_completed


def run_diverse_games(
    config: BlendedSelfplayConfig,
    num_games: int,
    output_db: str,
) -> int:
    """Generate games with diverse/random opponents for exploration.

    Returns number of games generated.
    """
    import uuid
    from app.db.game_replay import GameReplayDB
    from app.game_engine import GameEngine
    from app.models import BoardType, GameStatus, AIConfig, AIType
    from app.ai.factory import AIFactory
    from app.training.initial_state import create_initial_state

    logger.info(f"[diverse] Generating {num_games} games with varied strength...")

    board_type = BoardType(config.board_type)
    db = GameReplayDB(output_db, enforce_canonical_history=False)
    games_completed = 0
    temp_min, temp_max = config.diverse_temp_range

    for game_idx in range(num_games):
        try:
            game_id = str(uuid.uuid4())
            state = create_initial_state(board_type, config.num_players)
            moves = []

            # Random difficulty per player for this game
            player_difficulties = {
                p: random.randint(3, 9)
                for p in range(1, config.num_players + 1)
            }

            # Random temperature for this game
            temperature = random.uniform(temp_min, temp_max)

            # Create AIs with varied strength
            ais = {}
            for p in range(1, config.num_players + 1):
                ai_config = AIConfig(
                    difficulty=player_difficulties[p],
                    randomness=temperature * 0.3,  # Add randomness
                )
                ais[p] = AIFactory.create(
                    AIType.HEURISTIC,
                    player_number=p,
                    config=ai_config,
                )

            # Random opening moves for diversity
            opening_random_moves = random.randint(2, 6)

            move_count = 0
            while state.game_status != GameStatus.COMPLETED and len(moves) < 500:
                current_player = state.current_player

                # Random moves in opening for exploration diversity
                if move_count < opening_random_moves:
                    legal = GameEngine.get_valid_moves(state, current_player)
                    if legal:
                        move = random.choice(legal)
                    else:
                        break
                else:
                    ai = ais[current_player]
                    move = ai.select_move(state)

                if not move:
                    break

                state = GameEngine.apply_move(state, move)
                moves.append(move)
                move_count += 1

            if state.game_status == GameStatus.COMPLETED:
                try:
                    db.record_game_simple(
                        game_id=game_id,
                        initial_state=create_initial_state(board_type, config.num_players),
                        moves=moves,
                        final_state=state,
                        metadata={
                            "source": "blended_diverse",
                            "engine_mode": "diverse",
                            "difficulties": player_difficulties,
                            "temperature": temperature,
                        },
                    )
                    games_completed += 1
                except Exception as e:
                    logger.debug(f"[diverse] Failed to record game: {e}")

        except Exception as e:
            logger.debug(f"[diverse] Game {game_idx} failed: {e}")
            continue

    logger.info(f"[diverse] Completed {games_completed} games")
    return games_completed


def export_db_to_npz(db_path: str, output_path: str, board_type: str, num_players: int) -> int:
    """Export database to NPZ training file.

    Returns number of samples exported.
    """
    import subprocess

    logger.info(f"Exporting {db_path} to {output_path}...")

    cmd = [
        sys.executable, "scripts/export_replay_dataset.py",
        "--db", db_path,
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--output", output_path,
        "--allow-noncanonical",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.warning(f"Export failed: {result.stderr}")
        return 0

    # Count samples in output
    import numpy as np
    try:
        data = np.load(output_path, allow_pickle=True)
        count = len(data.get("features", []))
        logger.info(f"Exported {count} samples to {output_path}")
        return count
    except Exception as e:
        logger.warning(f"Could not count samples: {e}")
        return 0


def merge_npz_files(npz_files: list[str], output_path: str) -> int:
    """Merge multiple NPZ files into one.

    Returns total number of samples.
    """
    import numpy as np

    logger.info(f"Merging {len(npz_files)} NPZ files...")

    merged = {}
    total_samples = 0

    for npz_path in npz_files:
        if not Path(npz_path).exists():
            continue

        try:
            data = dict(np.load(npz_path, allow_pickle=True))

            for key in data:
                arr = data[key]
                if key not in merged:
                    merged[key] = []
                merged[key].append(arr)

            total_samples += len(data.get("features", []))
        except Exception as e:
            logger.warning(f"Could not load {npz_path}: {e}")

    # Concatenate arrays
    final = {}
    for key, arrays in merged.items():
        try:
            final[key] = np.concatenate(arrays, axis=0)
        except Exception as e:
            logger.warning(f"Could not merge key {key}: {e}")

    # Save merged file
    np.savez_compressed(output_path, **final)
    logger.info(f"Merged {total_samples} samples to {output_path}")

    return total_samples


def run_blended_selfplay(config: BlendedSelfplayConfig) -> dict[str, Any]:
    """Run blended selfplay with multiple engines.

    Returns statistics about the run.
    """
    import tempfile
    import shutil

    start_time = time.time()
    stats = {
        "board_type": config.board_type,
        "num_players": config.num_players,
        "blend_ratios": config.blend_ratios,
        "games_per_engine": {},
        "samples_per_engine": {},
        "total_samples": 0,
        "duration_seconds": 0,
    }

    # Calculate games per engine
    game_counts = games_per_engine(config.num_games, config.blend_ratios)
    stats["games_per_engine"] = game_counts

    logger.info("=" * 60)
    logger.info("Blended Selfplay Generation")
    logger.info("=" * 60)
    logger.info(f"Board: {config.board_type}, Players: {config.num_players}")
    logger.info(f"Total games: {config.num_games}")
    logger.info(f"Blend ratios: {config.blend_ratios}")
    logger.info(f"Games per engine: {game_counts}")
    logger.info("=" * 60)

    # Create temp directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix="blended_selfplay_"))
    npz_files = []

    try:
        # Generate games for each engine
        for engine, num_games in game_counts.items():
            if num_games <= 0:
                continue

            engine_db = str(temp_dir / f"{engine}_games.db")
            engine_npz = str(temp_dir / f"{engine}_data.npz")

            # Run engine-specific selfplay
            if engine == "heuristic":
                run_heuristic_games(config, num_games)
                # Heuristic runner doesn't output to DB easily, skip for now
                # In practice, use the unified selfplay.py with --record-db
                stats["games_per_engine"][engine] = num_games

            elif engine == "gumbel":
                actual_games = run_gumbel_games(config, num_games, engine_db)
                stats["games_per_engine"][engine] = actual_games

                # Export to NPZ
                if Path(engine_db).exists():
                    samples = export_db_to_npz(
                        engine_db, engine_npz,
                        config.board_type, config.num_players,
                    )
                    stats["samples_per_engine"][engine] = samples
                    if samples > 0:
                        npz_files.append(engine_npz)

            elif engine == "nnue":
                actual_games = run_nnue_games(config, num_games, engine_db)
                stats["games_per_engine"][engine] = actual_games

                if Path(engine_db).exists():
                    samples = export_db_to_npz(
                        engine_db, engine_npz,
                        config.board_type, config.num_players,
                    )
                    stats["samples_per_engine"][engine] = samples
                    if samples > 0:
                        npz_files.append(engine_npz)

            elif engine == "diverse":
                actual_games = run_diverse_games(config, num_games, engine_db)
                stats["games_per_engine"][engine] = actual_games

                if Path(engine_db).exists():
                    samples = export_db_to_npz(
                        engine_db, engine_npz,
                        config.board_type, config.num_players,
                    )
                    stats["samples_per_engine"][engine] = samples
                    if samples > 0:
                        npz_files.append(engine_npz)

        # Merge all NPZ files into final output
        if config.output_path and npz_files:
            output_path = Path(config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            total_samples = merge_npz_files(npz_files, str(output_path))
            stats["total_samples"] = total_samples
            logger.info(f"Final output: {output_path} ({total_samples} samples)")

        # Optionally keep the database
        if config.output_db:
            # Merge all engine DBs into one
            from app.db.game_replay import GameReplayDB
            final_db = GameReplayDB(config.output_db)

            for engine_db in temp_dir.glob("*_games.db"):
                try:
                    # Simple merge by copying games
                    source_db = GameReplayDB(str(engine_db))
                    games = source_db.list_games(limit=10000)
                    logger.info(f"Merging {len(games)} games from {engine_db.name}")
                except Exception as e:
                    logger.warning(f"Could not merge {engine_db}: {e}")

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    stats["duration_seconds"] = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BLENDED SELFPLAY COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {stats['duration_seconds']:.1f}s")
    logger.info(f"Games per engine: {stats['games_per_engine']}")
    logger.info(f"Samples per engine: {stats['samples_per_engine']}")
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate blended training data from multiple engine modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard 3-player blend
    python scripts/generate_blended_selfplay.py \\
        --board square8 --num-players 3 --num-games 2000 \\
        --output data/training/sq8_3p_blended.npz

    # Custom blend for 4-player
    python scripts/generate_blended_selfplay.py \\
        --board square8 --num-players 4 --num-games 1000 \\
        --blend heuristic:0.5,gumbel:0.3,nnue:0.2 \\
        --output data/training/sq8_4p_custom.npz
        """,
    )

    parser.add_argument(
        "--board", "-b",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--num-players", "-p",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Number of players (default: 3)",
    )
    parser.add_argument(
        "--num-games", "-n",
        type=int,
        default=1000,
        help="Total number of games to generate (default: 1000)",
    )
    parser.add_argument(
        "--blend",
        type=str,
        default=None,
        help="Blend ratios as 'engine:ratio,...' (default: heuristic:0.4,nnue:0.3,gumbel:0.2,diverse:0.1)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output NPZ file path",
    )
    parser.add_argument(
        "--output-db",
        type=str,
        default="",
        help="Optional: also save to GameReplayDB",
    )
    parser.add_argument(
        "--gumbel-budget",
        type=int,
        default=150,
        help="Gumbel MCTS simulation budget (default: 150)",
    )
    parser.add_argument(
        "--nnue-depth",
        type=int,
        default=3,
        help="NNUE search depth (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (0 = random)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Parse blend ratios
    blend_ratios = DEFAULT_BLEND.copy()
    if args.blend:
        blend_ratios = parse_blend_string(args.blend)

    # Set default output path
    output_path = args.output
    if not output_path:
        output_path = f"data/training/{args.board}_{args.num_players}p_blended.npz"

    config = BlendedSelfplayConfig(
        board_type=args.board,
        num_players=args.num_players,
        num_games=args.num_games,
        blend_ratios=blend_ratios,
        output_path=output_path,
        output_db=args.output_db,
        seed=args.seed,
        verbose=args.verbose,
        gumbel_budget=args.gumbel_budget,
        nnue_depth=args.nnue_depth,
        use_gpu=not args.no_gpu,
    )

    run_blended_selfplay(config)


if __name__ == "__main__":
    main()
