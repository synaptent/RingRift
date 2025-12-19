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
- mcts: MCTS (neural net backed)
- gumbel-mcts: Gumbel AlphaZero-style MCTS
- policy-only: Direct policy network (no search)
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
from app.training.selfplay_config import SelfplayConfig, create_argument_parser
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

# AI engine modes available (must match EngineMode enum in selfplay_config.py)
ENGINE_MODES = [
    "nnue-guided",      # Primary: NNUE evaluation
    "nn-minimax",       # Neural net + minimax search
    "mcts",             # MCTS (neural net backed)
    "nn-descent",       # Neural net + descent search
    "gumbel-mcts",      # Gumbel MCTS (AlphaZero style)
    "policy-only",      # Direct NN policy without search
    "heuristic-only",   # Baseline heuristic
]

# Weak opponents for asymmetric training
WEAK_OPPONENTS = [
    "random",           # Pure random
    "heuristic-only",   # Basic heuristic
]

# Strong opponents (all NN-backed algorithms)
STRONG_OPPONENTS = [
    "nnue-guided",
    "nn-minimax",
    "mcts",
    "nn-descent",
    "gumbel-mcts",
    "policy-only",
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
    # Robust training: increase strong_vs_weak for better value semantics learning
    matchup_weights: Dict[str, float] = field(default_factory=lambda: {
        "nnue_vs_nnue": 0.10,        # NNUE self-play (reduced)
        "nn_vs_nn": 0.20,            # NN-based AI battles
        "nnue_vs_nn": 0.15,          # Cross NN/NNUE matches
        "strong_vs_weak": 0.35,       # Asymmetric for learning (INCREASED)
        "heuristic_diverse": 0.20,    # Varied heuristic opponents (INCREASED)
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

    # 2. NN-based AI battles (Minimax, MCTS, Descent, Gumbel)
    # NOTE: Minimax is too slow for large boards (sq19, hex) - takes minutes per move.
    # Use only MCTS and Descent for those boards.
    nn_games = int(total_games * config.matchup_weights.get("nn_vs_nn", 0.25))
    if config.board_type.lower() in ("square19", "hexagonal"):
        nn_modes = ["mcts", "nn-descent", "gumbel-mcts"]  # Skip minimax for large boards
    else:
        nn_modes = ["nn-minimax", "mcts", "nn-descent", "gumbel-mcts"]
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
    # Filter strong opponents to exclude minimax for large boards
    strong_opponents = [s for s in STRONG_OPPONENTS
                       if not (s == "nn-minimax" and config.board_type.lower() in ("square19", "hexagonal"))]
    asym_games = int(total_games * config.matchup_weights.get("strong_vs_weak", 0.25))
    games_per_asym = asym_games // (len(strong_opponents) * len(WEAK_OPPONENTS) * 2)
    for strong in strong_opponents:
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

    # 5. Heuristic and random vs NN diversity (for robust value learning)
    heur_games = int(total_games * config.matchup_weights.get("heuristic_diverse", 0.20))
    games_per_heur = heur_games // 8  # More matchup types now
    if games_per_heur > 0:
        # Heuristic vs various NN types
        matchups.append(MatchupConfig(
            player1_mode="heuristic-only",
            player2_mode="nnue-guided",
            description="Heuristic vs NNUE",
            weight=games_per_heur
        ))
        matchups.append(MatchupConfig(
            player1_mode="heuristic-only",
            player2_mode="mcts",
            description="Heuristic vs MCTS",
            weight=games_per_heur
        ))
        matchups.append(MatchupConfig(
            player1_mode="heuristic-only",
            player2_mode="gumbel-mcts",
            description="Heuristic vs Gumbel-MCTS",
            weight=games_per_heur
        ))
        matchups.append(MatchupConfig(
            player1_mode="heuristic-only",
            player2_mode="nn-descent",
            description="Heuristic vs NN-Descent",
            weight=games_per_heur
        ))
        # Random vs various NN types (clear value signal)
        matchups.append(MatchupConfig(
            player1_mode="random",
            player2_mode="nnue-guided",
            description="Random vs NNUE",
            weight=games_per_heur
        ))
        matchups.append(MatchupConfig(
            player1_mode="random",
            player2_mode="mcts",
            description="Random vs MCTS",
            weight=games_per_heur
        ))
        matchups.append(MatchupConfig(
            player1_mode="random",
            player2_mode="gumbel-mcts",
            description="Random vs Gumbel-MCTS",
            weight=games_per_heur
        ))
        matchups.append(MatchupConfig(
            player1_mode="random",
            player2_mode="nn-descent",
            description="Random vs NN-Descent",
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
    total_matchups = len(matchups)
    logger.info(
        "[diverse-selfplay] Starting %d matchups for %s_%dp",
        total_matchups,
        config.board_type,
        config.num_players,
    )
    start_time = time.time()

    for matchup_idx, matchup in enumerate(matchups, 1):
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

        # Progress logging with ETA
        elapsed = time.time() - start_time
        rate = matchup_idx / elapsed if elapsed > 0 else 0
        remaining = total_matchups - matchup_idx
        eta_seconds = remaining / rate if rate > 0 else 0
        pct = matchup_idx / total_matchups * 100
        status_str = "OK" if success else "FAILED"
        logger.info(
            "[diverse-selfplay] Matchup %d/%d (%.1f%%) %s | %.2f matchups/min | "
            "ETA: %.0fs | games: %d total",
            matchup_idx,
            total_matchups,
            pct,
            status_str,
            rate * 60,
            eta_seconds,
            results["total_games"],
        )

    elapsed_total = time.time() - start_time
    logger.info(
        "[diverse-selfplay] Completed %d/%d matchups in %.1fs | %d games total",
        results["matchups_completed"],
        total_matchups,
        elapsed_total,
        results["total_games"],
    )
    results["completed_at"] = datetime.now().isoformat()
    return results


def main():
    # Use unified argument parser from SelfplayConfig
    parser = create_argument_parser(
        description="Diverse Selfplay Generator",
        include_ramdrive=True,
        include_gpu=True,
    )
    # Add script-specific arguments
    parser.add_argument("--config", type=str, help="Specific config to run (e.g., hexagonal_2p)")
    parser.add_argument("--games-per-matchup", type=int, default=50, help="Games per matchup type")
    parser.add_argument("--all-configs", action="store_true", help="Run all 9 configs")
    parser.add_argument("--priority-configs", action="store_true",
                        help="Run priority configs (least models first)")
    # Note: ramdrive args added by create_argument_parser(include_ramdrive=True)
    # Add extra ramdrive args not in base parser
    parser.add_argument("--ram-storage", action="store_true", help="Use ramdrive storage")
    parser.add_argument("--sync-target", type=str, help="Target directory for ramdrive sync")
    parsed = parser.parse_args()

    # Create base SelfplayConfig from parsed args
    base_config = SelfplayConfig(
        board_type=parsed.board,
        num_players=parsed.num_players,
        num_games=parsed.games_per_matchup * 10,  # Estimated total
        output_dir=parsed.output_dir,
        use_gpu=not getattr(parsed, "no_gpu", False),
        gpu_device=getattr(parsed, "gpu_device", 0),
        seed=parsed.seed,
        source="run_diverse_selfplay.py",
        extra_options={
            "games_per_matchup": parsed.games_per_matchup,
            "all_configs": parsed.all_configs,
            "priority_configs": parsed.priority_configs,
            "config_key": parsed.config,
        },
    )

    # Map old args names for backward compatibility
    args = type("Args", (), {
        "config": parsed.config,
        "board": parsed.board,
        "players": parsed.num_players,
        "games_per_matchup": parsed.games_per_matchup,
        "gpu": base_config.gpu_device,
        "output_dir": parsed.output_dir,
        "all_configs": parsed.all_configs,
        "priority_configs": parsed.priority_configs,
        "ram_storage": getattr(parsed, "ram_storage", False),
        "sync_interval": base_config.sync_interval,  # From SelfplayConfig
        "sync_target": getattr(parsed, "sync_target", None),
    })()

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
