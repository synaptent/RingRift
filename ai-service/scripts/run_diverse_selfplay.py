#!/usr/bin/env python3
"""Diverse Selfplay Generator - Creates varied training data with multiple AI types.

This script generates training games using diverse AI matchups to improve
model robustness and generalization. Instead of heuristic vs heuristic,
it focuses on:

1. NNUE-guided AI vs various opponents
2. NN (Neural Network) based AI: Minimax, MCTS, Descent
3. Cross-AI matches: random/heuristic vs advanced AI
4. Asymmetric matchups for robust training

AI Types:
- heuristic-only: Fast baseline (avoid for training variety)
- nnue-guided: NNUE evaluation with search
- nn-minimax: Neural network with minimax search
- nn-mcts: Neural network with MCTS
- nn-descent: Neural network with gradient descent search
- random: Pure random moves (weak opponent for diversity)

Usage:
    # Run diverse selfplay on all configs
    python scripts/run_diverse_selfplay.py

    # Focus on specific config
    python scripts/run_diverse_selfplay.py --config hexagonal_2p

    # Specify number of games per matchup
    python scripts/run_diverse_selfplay.py --games-per-matchup 50

    # Run on specific GPU
    python scripts/run_diverse_selfplay.py --gpu 0
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import BoardType
from app.training.env import get_theoretical_max_moves
from app.utils.ramdrive import add_ramdrive_args, get_config_from_args, get_games_directory, RamdriveSyncer

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("run_diverse_selfplay", log_dir="logs")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [DiverseSelfplay] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# AI engine modes available
ENGINE_MODES = [
    "nnue-guided",      # Primary: NNUE evaluation
    "nn-minimax",       # Neural net + minimax search
    "nn-mcts",          # Neural net + MCTS
    "nn-descent",       # Neural net + descent search
    "heuristic-only",   # Baseline heuristic (use sparingly)
]

# Weak opponents for asymmetric training
WEAK_OPPONENTS = [
    "random",           # Pure random
    "heuristic-only",   # Basic heuristic
]

# Strong opponents
STRONG_OPPONENTS = [
    "nnue-guided",
    "nn-minimax",
    "nn-mcts",
    "nn-descent",
]


@dataclass
class MatchupConfig:
    """Configuration for a specific AI matchup."""
    player1_mode: str
    player2_mode: str
    description: str
    weight: float = 1.0  # Relative weight for game distribution


@dataclass
class DiverseSelfplayConfig:
    """Configuration for diverse selfplay generation."""
    board_type: str
    num_players: int
    games_per_matchup: int = 50
    max_moves: Optional[int] = None  # Auto-calculated from board_type/num_players
    output_dir: Optional[Path] = None
    gpu_id: int = 0
    batch_size: int = 32

    def __post_init__(self):
        """Auto-calculate max_moves if not specified."""
        if self.max_moves is None:
            # Map board_type string to BoardType enum
            board_type_map = {
                "square8": BoardType.SQUARE8,
                "square19": BoardType.SQUARE19,
                "hexagonal": BoardType.HEXAGONAL,
            }
            board_enum = board_type_map.get(self.board_type.lower())
            if board_enum:
                self.max_moves = get_theoretical_max_moves(board_enum, self.num_players)
                logger.info(f"Auto-calculated max_moves={self.max_moves} for {self.board_type} {self.num_players}p")
            else:
                # Fallback for unknown board types
                self.max_moves = 10000
                logger.warning(f"Unknown board type '{self.board_type}', using default max_moves=10000")

    # Matchup distribution weights
    # Higher weight = more games of this type
    matchup_weights: Dict[str, float] = field(default_factory=lambda: {
        "nnue_vs_nnue": 0.15,        # NNUE self-play
        "nn_vs_nn": 0.25,            # NN-based AI battles
        "nnue_vs_nn": 0.20,          # Cross NN/NNUE matches
        "strong_vs_weak": 0.25,       # Asymmetric for learning
        "heuristic_diverse": 0.15,    # Varied heuristic opponents
    })


def get_diverse_matchups(config: DiverseSelfplayConfig) -> List[MatchupConfig]:
    """Generate diverse matchup configurations based on weights."""
    matchups = []

    total_games = config.games_per_matchup * 10  # Total games to distribute

    # 1. NNUE self-play (baseline strong games)
    nnue_games = int(total_games * config.matchup_weights.get("nnue_vs_nnue", 0.15))
    if nnue_games > 0:
        matchups.append(MatchupConfig(
            player1_mode="nnue-guided",
            player2_mode="nnue-guided",
            description="NNUE self-play",
            weight=nnue_games
        ))

    # 2. NN-based AI battles (Minimax, MCTS, Descent)
    nn_games = int(total_games * config.matchup_weights.get("nn_vs_nn", 0.25))
    nn_modes = ["nn-minimax", "nn-mcts", "nn-descent"]
    games_per_nn_pair = nn_games // (len(nn_modes) * len(nn_modes))
    for mode1 in nn_modes:
        for mode2 in nn_modes:
            if games_per_nn_pair > 0:
                matchups.append(MatchupConfig(
                    player1_mode=mode1,
                    player2_mode=mode2,
                    description=f"{mode1} vs {mode2}",
                    weight=games_per_nn_pair
                ))

    # 3. Cross NNUE/NN matches
    cross_games = int(total_games * config.matchup_weights.get("nnue_vs_nn", 0.20))
    games_per_cross = cross_games // len(nn_modes)
    for nn_mode in nn_modes:
        if games_per_cross > 0:
            matchups.append(MatchupConfig(
                player1_mode="nnue-guided",
                player2_mode=nn_mode,
                description=f"NNUE vs {nn_mode}",
                weight=games_per_cross
            ))
            matchups.append(MatchupConfig(
                player1_mode=nn_mode,
                player2_mode="nnue-guided",
                description=f"{nn_mode} vs NNUE",
                weight=games_per_cross
            ))

    # 4. Strong vs Weak (asymmetric training)
    asym_games = int(total_games * config.matchup_weights.get("strong_vs_weak", 0.25))
    games_per_asym = asym_games // (len(STRONG_OPPONENTS) * len(WEAK_OPPONENTS) * 2)
    for strong in STRONG_OPPONENTS:
        for weak in WEAK_OPPONENTS:
            if games_per_asym > 0:
                # Strong as P1
                matchups.append(MatchupConfig(
                    player1_mode=strong,
                    player2_mode=weak,
                    description=f"{strong} vs {weak} (asymmetric)",
                    weight=games_per_asym
                ))
                # Strong as P2 (learning from different position)
                matchups.append(MatchupConfig(
                    player1_mode=weak,
                    player2_mode=strong,
                    description=f"{weak} vs {strong} (asymmetric)",
                    weight=games_per_asym
                ))

    # 5. Heuristic diversity (for coverage)
    heur_games = int(total_games * config.matchup_weights.get("heuristic_diverse", 0.15))
    games_per_heur = heur_games // 3
    if games_per_heur > 0:
        matchups.append(MatchupConfig(
            player1_mode="heuristic-only",
            player2_mode="random",
            description="Heuristic vs Random",
            weight=games_per_heur
        ))
        matchups.append(MatchupConfig(
            player1_mode="heuristic-only",
            player2_mode="nnue-guided",
            description="Heuristic vs NNUE",
            weight=games_per_heur
        ))
        matchups.append(MatchupConfig(
            player1_mode="heuristic-only",
            player2_mode="nn-mcts",
            description="Heuristic vs NN-MCTS",
            weight=games_per_heur
        ))

    return matchups


def run_matchup_games(
    config: DiverseSelfplayConfig,
    matchup: MatchupConfig,
    num_games: int
) -> Tuple[bool, int, str]:
    """Run games for a specific matchup configuration.

    Returns:
        Tuple of (success, games_completed, output_dir)
    """
    timestamp = int(time.time())
    run_id = f"{matchup.player1_mode}_vs_{matchup.player2_mode}_{timestamp}"
    output_dir = config.output_dir / f"diverse_{config.board_type}_{config.num_players}p" / run_id

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command based on AI types
    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "run_hybrid_selfplay.py"),
        "--board-type", config.board_type,
        "--num-players", str(config.num_players),
        "--num-games", str(num_games),
        "--max-moves", str(config.max_moves),
        "--output-dir", str(output_dir),
        "--engine-mode", matchup.player1_mode,  # P1 engine
        "--seed", str(random.randint(1, 1000000)),
    ]

    # For asymmetric matches, we need to specify P2's engine separately
    # This requires the selfplay script to support --p2-engine-mode
    if matchup.player1_mode != matchup.player2_mode:
        cmd.extend(["--p2-engine-mode", matchup.player2_mode])

    # GPU settings
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    logger.info(f"Starting: {matchup.description} ({num_games} games)")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600 * 2,  # 2 hour timeout
            cwd=str(AI_SERVICE_ROOT)
        )

        if result.returncode == 0:
            logger.info(f"Completed: {matchup.description}")
            return True, num_games, str(output_dir)
        else:
            logger.error(f"Failed: {matchup.description}")
            logger.error(f"Stderr: {result.stderr[:500]}")
            return False, 0, str(output_dir)

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout: {matchup.description}")
        return False, 0, str(output_dir)
    except Exception as e:
        logger.error(f"Error in {matchup.description}: {e}")
        return False, 0, str(output_dir)


async def run_diverse_selfplay(config: DiverseSelfplayConfig) -> Dict[str, Any]:
    """Run diverse selfplay generation."""
    results = {
        "config": f"{config.board_type}_{config.num_players}p",
        "started_at": datetime.now().isoformat(),
        "matchups_completed": 0,
        "total_games": 0,
        "matchup_results": [],
    }

    matchups = get_diverse_matchups(config)
    logger.info(f"Generated {len(matchups)} matchup configurations for {config.board_type}_{config.num_players}p")

    for matchup in matchups:
        num_games = int(matchup.weight)
        if num_games <= 0:
            continue

        success, games, output_dir = run_matchup_games(config, matchup, num_games)

        results["matchup_results"].append({
            "description": matchup.description,
            "games_requested": num_games,
            "games_completed": games,
            "success": success,
            "output_dir": output_dir,
        })

        if success:
            results["matchups_completed"] += 1
            results["total_games"] += games

    results["completed_at"] = datetime.now().isoformat()
    return results


def main():
    parser = argparse.ArgumentParser(description="Diverse Selfplay Generator")
    parser.add_argument("--config", type=str, help="Specific config to run (e.g., hexagonal_2p)")
    parser.add_argument("--board", type=str, help="Board type (square8, square19, hexagonal)")
    parser.add_argument("--players", type=int, help="Number of players (2, 3, 4)")
    parser.add_argument("--games-per-matchup", type=int, default=50, help="Games per matchup type")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--output-dir", type=str, help="Output directory (overrides --ram-storage)")
    parser.add_argument("--all-configs", action="store_true", help="Run all 9 configs")
    parser.add_argument("--priority-configs", action="store_true",
                        help="Run priority configs (least models first)")
    add_ramdrive_args(parser)  # Add --ram-storage, --sync-interval, --sync-target
    args = parser.parse_args()

    # Determine output directory: explicit path > ramdrive > default
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ramdrive_config = get_config_from_args(args)
        ramdrive_config.subdirectory = "selfplay/diverse"
        output_dir = get_games_directory(prefer_ramdrive=args.ram_storage, config=ramdrive_config)

    # Set up ramdrive sync if requested
    syncer = None
    if args.ram_storage and args.sync_interval > 0 and args.sync_target:
        syncer = RamdriveSyncer(
            source_dir=output_dir,
            target_dir=Path(args.sync_target),
            interval=args.sync_interval,
            patterns=["*.db", "*.jsonl", "*.npz"],
        )
        syncer.start()
        logger.info(f"Started ramdrive sync: {output_dir} -> {args.sync_target} every {args.sync_interval}s")

    configs_to_run = []

    if args.config:
        parts = args.config.rsplit("_", 1)
        configs_to_run.append((parts[0], int(parts[1].replace("p", ""))))
    elif args.board and args.players:
        configs_to_run.append((args.board, args.players))
    elif args.all_configs:
        for board in ["square8", "square19", "hexagonal"]:
            for players in [2, 3, 4]:
                configs_to_run.append((board, players))
    elif args.priority_configs:
        # Priority order based on model counts (least first)
        configs_to_run = [
            ("hexagonal", 2),   # 1 model
            ("hexagonal", 4),   # 1 model
            ("square19", 3),    # 4 models
            ("hexagonal", 3),   # 6 models
            ("square8", 3),     # 6 models
        ]
    else:
        # Default: run hexagonal configs (most underrepresented)
        configs_to_run = [
            ("hexagonal", 2),
            ("hexagonal", 3),
            ("hexagonal", 4),
        ]

    logger.info(f"Running diverse selfplay for {len(configs_to_run)} configurations")

    all_results = []
    for board_type, num_players in configs_to_run:
        config = DiverseSelfplayConfig(
            board_type=board_type,
            num_players=num_players,
            games_per_matchup=args.games_per_matchup,
            output_dir=output_dir,
            gpu_id=args.gpu,
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting: {board_type}_{num_players}p")
        logger.info(f"{'='*60}")

        results = asyncio.run(run_diverse_selfplay(config))
        all_results.append(results)

        logger.info(f"Completed {board_type}_{num_players}p: "
                    f"{results['total_games']} games, "
                    f"{results['matchups_completed']} matchups")

    # Stop syncer and perform final sync
    if syncer:
        logger.info("Stopping ramdrive syncer and performing final sync...")
        syncer.stop(final_sync=True)
        logger.info(f"Sync stats: {syncer.stats}")

    # Summary
    print("\n" + "=" * 60)
    print("DIVERSE SELFPLAY SUMMARY")
    print("=" * 60)
    total_games = sum(r["total_games"] for r in all_results)
    total_matchups = sum(r["matchups_completed"] for r in all_results)
    print(f"Total games generated: {total_games}")
    print(f"Total matchups completed: {total_matchups}")
    print(f"Output directory: {output_dir}")
    for r in all_results:
        print(f"  {r['config']}: {r['total_games']} games")


if __name__ == "__main__":
    main()
